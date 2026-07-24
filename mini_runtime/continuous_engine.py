import asyncio
import torch
from asyncio import Queue
from .request import Request
from .metrics import Metrics
from .kv_cache import KVCacheManager, BlockTable
from .prefix_cache import PrefixCache
from .config import BLOCK_SIZE, NUM_BLOCKS
from .backends.native_backend import NativeBackend, PrefillInput, BatchDecodeInput

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
        self.prefix_cache = PrefixCache(block_size=block_size)
        
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
            match_result = self.prefix_cache.match(request.token_ids)
            matched_blocks = match_result['matched_blocks']
            num_matched_tokens = match_result['num_matched_tokens']
            matched_offset = match_result['matched_offset']
            num_matched_blocks = len(matched_blocks)

            block_table = BlockTable(self.block_size)
            block_table.set_offset(matched_offset)
            # 1. 复用 matched blocks (运行请求引用 +1)
            for bid in matched_blocks:
                self.kv_manager.inc_ref(bid)
                block_table.append_block(bid)
            # 2. 为 remaining 分配新 block; OOM 时 evict prefix cache 重试
            if match_result['remaining_tokens']:
                can_allocate = self.kv_manager.allocate(block_table, len(request.token_ids))
                while not can_allocate:
                    evicted = self.prefix_cache.evict()
                    if evicted is None:
                        break
                    for bid in evicted:
                        self.kv_manager.dec_ref(bid)
                    can_allocate = self.kv_manager.allocate(block_table, len(request.token_ids))
                if not can_allocate:
                    # 回滚已复用的 matched blocks
                    for bid in matched_blocks:
                        self.kv_manager.dec_ref(bid)
                    self.waiting_queue.task_done()
                    self.metrics.oom += 1
                    break

            request.block_table = block_table
            request.num_matched_tokens = num_matched_tokens
            request.matched_offset = matched_offset
            request.num_matched_blocks = num_matched_blocks
            request.match_result = match_result
            self.running_requests.append(request)
            self.metrics.max_running_requests = max(
                self.metrics.max_running_requests, len(self.running_requests)
            )

            self.waiting_queue.task_done()
    
    async def prefill_new_requests(self):
        """对未 prefill 的请求做 prefix-aware prefill，并把新 block 插入 prefix cache。"""
        new_requests = [r for r in self.running_requests if not r.prefill_done]
        if not new_requests:
            return

        self.metrics.prefill_batches += 1

        for r in new_requests:
            inp = PrefillInput(
                request_id=r.request_id,
                token_ids=r.token_ids,
                block_ids=r.block_table.block_ids,
                block_offset=r.block_table.offset,
                skip_tokens=r.num_matched_tokens,
                num_cached_blocks=r.num_matched_blocks,
            )
            first_token = await asyncio.to_thread(self.backend.prefill, inp)
            # prefill 完成, 把新 block 插入 prefix cache (cache 持有, ref_count +1)
            new_cache_blocks = self.prefix_cache.insert(
                r.token_ids, list(r.block_table.block_ids), r.match_result)
            for bid in new_cache_blocks:
                self.kv_manager.inc_ref(bid)

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
            total = len(r.token_ids) + r.generated_tokens
            if total > r.block_table.capacity:
                success = self.kv_manager.allocate(r.block_table, total)
                while not success:
                    evicted = self.prefix_cache.evict()
                    if evicted is None:
                        break
                    for bid in evicted:
                        self.kv_manager.dec_ref(bid)
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

        batched = [BatchDecodeInput(
            request_id=r.request_id,
            token_id=r._last_token,
            block_ids=r.block_table.block_ids,
            block_offset=r.matched_offset,
        ) for r in self.running_requests]
        
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
        if request.block_table:
            self.kv_manager.free(request.block_table)
        self.backend.release(request.request_id)
        if request.future.done():
            self.metrics.cancelled += 1
            return

        queue_wait = request.start_time - request.submit_time
        service_time = finish_time - request.start_time
        ttft = (request.first_token_time - request.submit_time
                if request.first_token_time is not None else None)
        total = finish_time - request.submit_time
        tpot = (
            (finish_time - request.first_token_time) / request.generated_tokens
            if request.generated_tokens and request.first_token_time is not None else 0
        )

        request.future.set_result({
            "request_id": request.request_id,
            "ttft": ttft,
            "tpot": tpot,
            "total": total,
            "generated_tokens": request.generated_tokens,
            "queue_wait": queue_wait,
            "service_time": service_time,
            "output": self.backend.tokenizer.decode(request._generated_token_ids),
        })
        self.metrics.success += 1
        if ttft is not None:
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
        loop = asyncio.get_running_loop()
        future = loop.create_future()

        request = Request(
            request_id = self.next_request_id,
            prompt = prompt,
            token_ids = input_ids,
            max_new_tokens = max_new_tokens,
            submit_time = loop.time(),
            future = future,
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
        # 取消等待队列中的请求
        while not self.waiting_queue.empty():
            try:
                request = self.waiting_queue.get_nowait()
                if not request.future.done():
                    request.future.set_result({
                        "request_id": request.request_id,
                        "error": "cancelled",
                    })
                    self.metrics.cancelled += 1
                self.waiting_queue.task_done()
            except asyncio.QueueEmpty:
                break

        # 释放正在运行请求的资源
        for request in self.running_requests:
            if request.block_table:
                self.kv_manager.free(request.block_table)
            if not request.future.done():
                request.future.set_result({
                    "request_id": request.request_id,
                    "error": "cancelled",
                })
                self.metrics.cancelled += 1

        self.running_requests.clear()

        if self.engine_task:
            self.engine_task.cancel()
            await asyncio.gather(self.engine_task, return_exceptions=True)
            