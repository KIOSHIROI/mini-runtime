import asyncio
import torch
from asyncio import Queue
from .request import Request
from .metrics import Metrics
from .kv_cache import KVCacheManager, BlockTable
from .config import BLOCK_SIZE, NUM_BLOCKS
from .backends.native_backend import NativeBackend

class Engine:
    def __init__(
        self,
        backend = None,
        max_batch_size: int = 4,
        request_timeout: float = 30.0,
        num_blocks: int = NUM_BLOCKS,
        block_size: int = BLOCK_SIZE,
        device = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.backend = backend
        
        self.waiting_queue = Queue()
        self.running_requests = []
        self.max_batch_size = max_batch_size
        self.request_timeout = request_timeout

        self.next_request_id = 0
        self.engine_task = None
        self.metrics = Metrics()
        self.block_size = block_size

        self.kv_manager = KVCacheManager(
            num_blocks=num_blocks,
            block_size=block_size,
            num_layers=backend.model.config.num_layers,
            num_kv_heads=backend.model.config.num_kv_heads,
            head_dim=backend.model.config.head_dim,
            device=device
        )
        
        self.backend.kv_manager = self.kv_manager 
        
    async def start(self):
        self.engine_task = asyncio.create_task(self.scheduler_loop())
    
    async def scheduler_loop(self):
        while True:
            await self.admit_requests()
            await self.prefill_new_requests()
            await self.decode_one_step()
    
    async def admit_requests(self):
        while len(self.running_requests) < self.max_batch_size:
            try:
                request: Request = self.waiting_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            
            request.start_time = asyncio.get_running_loop().time()
            block_table = BlockTable(self.block_size)
            can_allocate = self.kv_manager.allocate(block_table, request.prompt_tokens)
            
            if not can_allocate:
                self.waiting_queue.task_done()
                self.metrics.oom += 1
                break
            
            request.block_table = block_table
            self.running_requests.append(request)
            self.metrics.max_running_requests = max(
                self.metrics.max_running_requests, len(self.running_requests)
            )
            
            self.waiting_queue.task_done()
    
    async def prefill_new_requests(self):
        """_summary_
        处理 running_request 列表中的 Request
        将每个请求放到 asyncio 的线程池中
        由后端模型完成 Prefill 得到生成的第一个 token
        将该 token 标记为 request 生成的最后一个 token
        将 request 标记为 prefill_done
        
        """
        new_requests = [r for r in self.running_requests if not r.prefill_done]
        
        if not new_requests:
            return
        
        self.metrics.prefill_batches += 1
        
        for r in new_requests:
            first_token =await asyncio.to_thread(
                self.backend.prefill,
                r.request_id,
                r.prompt,
                r.block_table.block_ids
                ) # to_thread 避免阻塞事件循环

            r._last_token = first_token 
            r._generated_token_ids.append(first_token)
            r.generated_tokens = 1
            r.first_token_time = asyncio.get_running_loop().time()
            r.prefill_done = True
    
    async def decode_one_step(self):
        if not self.running_requests:
            await asyncio.sleep(0.001)
            return
        
        active = len(self.running_requests) 
               
        
        self.metrics.decode_steps += 1
        
        self.metrics.total_active_requests  += active  
          
        now = asyncio.get_running_loop().time()
                
        finished = []
        oom_requests = []

        for r in self.running_requests:
            total = r.prompt_tokens + r.generated_tokens
            if total > r.block_table.capacity:
                success = self.kv_manager.allocate(r.block_table, total)
                if not success:
                    oom_requests.append(r)

        for r in oom_requests:
            self.running_requests.remove(r)
            self.kv_manager.free(r.block_table)
            self.metrics.oom += 1
            if not r.future.done():
                r.future.set_result({
                    "request_id": r.request_id,
                    "error": "OOM",
                })
            else:
                self.metrics.cancelled += 1
                
        batched = [(r.request_id, r._last_token, r.block_table.block_ids) for r in self.running_requests]
        
        next_tokens = await asyncio.to_thread(
            self.backend.batch_decode,
            batched,
        )
        
        for r, next_token in zip(self.running_requests, next_tokens):                
            if r.first_token_time is None:
                r.first_token_time = now
                            
            r.generated_tokens += 1

            if next_token is not None:
                r._generated_token_ids.append(next_token)
            r._last_token = next_token

            if r.generated_tokens >= r.max_new_tokens or r._last_token is None:
                finished.append(r)
        
        for r in finished:
            self.finish_request(r, now)
            self.running_requests.remove(r)
        
    
    def finish_request(self, request: Request, finish_time: float):
        self.kv_manager.free(request.block_table)
        self.backend.release(request.request_id)
        if request.future.done():
            self.metrics.cancelled += 1
            return 
        
        queue_wait = request.start_time - request.submit_time
        service_time = finish_time - request.start_time
        ttft = request.first_token_time - request.submit_time 
        total = finish_time - request.submit_time 
        tpot = (
            (finish_time - request.first_token_time) / request.generated_tokens
            if request.generated_tokens else 0
        )
        
        request.future.set_result({
            "request_id": request.request_id,
            "ttft": ttft,
            "tpot": tpot,
            "total": total,
            "generated_tokens": request.generated_tokens,
            "queue_wait": queue_wait,
            "service_time": service_time,
        })
        self.metrics.success += 1
        self.metrics.total_ttft += ttft
        self.metrics.total_tpot += tpot
        self.metrics.total_latency += total
        self.metrics.total_output_tokens += request.generated_tokens
        self.metrics.total_queue_wait += queue_wait
        self.metrics.total_service_time += service_time
        
    async def submit(self, prompt: str, max_new_tokens: int = 128):
        """_summary_
        用户调用该函数提交一个生成请求
        
        """
        message = [{"role": "user", "content": prompt}]
        chat_text = self.backend.tokenizer.apply_chat_template(
            message, tokenize=False, add_generation_prompt=True
        )
        input_ids = self.backend.tokenizer.encode(chat_text)
        prompt_tokens = len(input_ids)
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        
        request = Request(
            request_id = self.next_request_id,
            prompt = prompt,
            submit_time = loop.time(),
            future = future,
            prompt_tokens = prompt_tokens,
            max_new_tokens = max_new_tokens,
            token_ids = input_ids,          # 保存完整 token 序列，用于 prefix matching
        )
        
        self.next_request_id += 1
        
        await self.waiting_queue.put(request)
        self.metrics.submitted += 1
        
        try:
            return await asyncio.wait_for(future, timeout=self.request_timeout)
        except asyncio.TimeoutError:
            future.cancel()
            self.metrics.timeout += 1
            return {
                "request_id": request.request_id,
                "error": "timeout",
            }
            
    def snapshot_metrics(self):
        success = self.metrics.success 
        decode_steps = self.metrics.decode_steps
        
        kv_snapshot = self.kv_manager.snapshot()
        
        return {
            "submitted": self.metrics.submitted,
            "success": self.metrics.success,
            "timeout": self.metrics.timeout,
            "cancelled": self.metrics.cancelled,
            "oom": self.metrics.oom,
            "prefill_batches": self.metrics.prefill_batches,
            "decode_steps": self.metrics.decode_steps,
            "avg_active_requests": (
                self.metrics.total_active_requests / decode_steps
                if decode_steps else 0
            ),
            "max_running_requests": self.metrics.max_running_requests,
            "avg_queue_wait": (
                self.metrics.total_queue_wait / success
                if success else 0
            ),
            "avg_service_time": (
                self.metrics.total_service_time / success
                if success else 0
            ),
            "avg_latency": (
                self.metrics.total_latency / success
                if success else 0
            ),
            "avg_ttft": (
                self.metrics.total_ttft / success
                if success else 0
            ),
            "avg_tpot": (
                self.metrics.total_tpot / success
                if success else 0            ),
            "output_tokens_per_sec": (
                self.metrics.total_output_tokens / self.metrics.total_service_time
                if self.metrics.total_service_time > 0 else 0
            ),
            "kv_cache": kv_snapshot,
        }
    async def shutdown(self):
        await self.waiting_queue.join()
        
        while self.running_requests:
            await asyncio.sleep(0.001)
            
        if self.engine_task:
            self.engine_task.cancel()
            await asyncio.gather(self.engine_task, return_exceptions=True)
            