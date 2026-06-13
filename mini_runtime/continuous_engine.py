import asyncio
from asyncio import Queue
from .request import Request
from .metrics import Metrics
from .kv_cache import KVCacheManager, BlockTable
from .config import BLOCK_SIZE, NUM_BLOCKS
class ContinuousBatchingEngine:
    def __init__(
        self,
        max_batch_size: int = 4,
        request_timeout: float = 30.0,
        prefill_time_per_token: float = 0.01,
        decode_time_per_token: float = 0.05,
        num_blocks: int = NUM_BLOCKS,
        block_size: int = BLOCK_SIZE,
    ):
        self.waiting_queue = Queue()
        self.running_requests = []
        self.max_batch_size = max_batch_size
        self.request_timeout = request_timeout
        self.prefill_time_per_token = prefill_time_per_token
        self.decode_time_per_token = decode_time_per_token
        self.next_request_id = 0
        self.engine_task = None
        self.metrics = Metrics()
        self.block_size = block_size

        self.kv_manager = KVCacheManager(
            num_blocks=num_blocks,
            block_size=block_size,
        )
        
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
        new_requests = [r for r in self.running_requests if not r.prefill_done]
        
        if not new_requests:
            return
        
        self.metrics.prefill_batches += 1
        
        prefill_time = max(r.prompt_tokens for r in new_requests) * \
        self.prefill_time_per_token
        await asyncio.sleep(prefill_time)
        
        for r in new_requests:
            r.prefill_done = True
    
    async def decode_one_step(self):
        if not self.running_requests:
            await asyncio.sleep(0.001)
            return
        
        active = len(self.running_requests) 
               
            
        await asyncio.sleep(self.decode_time_per_token)
        
        self.metrics.decode_steps += 1
        
        self.metrics.total_active_requests  += active  
          
        now = asyncio.get_running_loop().time()
                
        finished = []
        
        for r in self.running_requests:
            if r.first_token_time is None:
                r.first_token_time = now
                
            r.generated_tokens += 1

            if r.generated_tokens > r.block_table.capacity:
                can_allocate = self.kv_manager.allocate(r.block_table, r.generated_tokens)
                if not can_allocate:
                    self.metrics.oom += 1

            if r.generated_tokens >= r.max_new_tokens:
                finished.append(r)
        
        for r in finished:
            self.finish_request(r, now)
            self.running_requests.remove(r)
        
    
    def finish_request(self, request: Request, finish_time: float):
        self.kv_manager.free(request.block_table)
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
        
    async def submit(self, prompt: str, prompt_tokens: int = 32, max_new_tokens: int = 32):
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        
        request = Request(
            request_id = self.next_request_id,
            prompt = prompt,
            submit_time = loop.time(),
            future = future,
            prompt_tokens = prompt_tokens,
            max_new_tokens = max_new_tokens
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
            