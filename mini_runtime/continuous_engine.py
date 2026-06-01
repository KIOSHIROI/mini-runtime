import asyncio
from asyncio import Queue
from .request import Request
from .metrics import Metrics

class ContinuousBatchingEngine:
    def __init__(
        self,
        max_batch_size: int = 4,
        request_timeout: float = 30.0,
        prefill_time_per_token: float = 0.01,
        decode_time_per_token: float = 0.05,
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
                request = self.waiting_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
                
            request.start_time = asyncio.get_running_loop().time()
            self.running_requests.append(request)
            self.waiting_queue.task_done()
    
    async def prefill_new_requests(self):
        new_requests = [r for r in self.running_requests if not r.prefill_done]
        
        if not new_requests:
            return
        
        prefill_time = max(r.prompt_tokens for r in new_requests) * \
        self.prefill_time_per_token
        await asyncio.sleep(prefill_time)
        
        for r in new_requests:
            r.prefill_done = True
    
    async def decode_one_step(self):
        if not self.running_requests:
            await asyncio.sleep(0.001)
            return
        
        await asyncio.sleep(self.decode_time_per_token)
        now = asyncio.get_running_loop().time()
                
        finished = []
        
        for r in self.running_requests:
            if r.first_token_time is None:
                r.first_token_time = now
                
            r.generated_tokens += 1
        
            if r.generated_tokens >= r.max_new_tokens:
                finished.append(r)
        
        for r in finished:
            self.finish_request(r, now)
            self.running_requests.remove(r)
    
    def finish_request(self, request: Request, finish_time: float):
        if request.future.done():
            self.metrics.cancelled += 1
            return 
        
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
        })
        self.metrics.success += 1
        
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
    
    async def shutdown(self):
        await self.waiting_queue.join()
        
        while self.running_requests:
            await asyncio.sleep(0.001)
            
        if self.engine_task:
            self.engine_task.cancel()
            await asyncio.gather(self.engine_task, return_exceptions=True)
            