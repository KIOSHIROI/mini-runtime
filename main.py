import asyncio 
from asyncio import Queue
from dataclasses import dataclass 

@dataclass
class Request:
    request_id: int
    prompt: str
    submit_time: float
    future: asyncio.Future
    
    start_time: float | None = None
    finish_time: float | None = None
@dataclass
class Metrics:
    submitted: int = 0
    success: int = 0
    timeout: int = 0
    cancelled: int = 0
    batches: int = 0
    total_batch_size: int = 0
    total_queue_wait: float = 0.0
    total_service_time: float = 0.0
    total_latency: float = 0.0
    
class MiniRuntime:
    def __init__(
        self,
        max_batch_size: int = 4,
        batch_timeout: float = 0.5,
        request_timeout: float = 1.0,
        num_workers: int = 3,
     ):
        self.request_queue = asyncio.Queue()
        self.batch_queue = asyncio.Queue()
        self.max_batch_size = max_batch_size
        self.batch_timeout = batch_timeout
        self.request_timeout = request_timeout
        self.num_workers = num_workers
        self.scheduler_task = None
        self.worker_tasks = []  
        self.next_request_id = 0
        self.metrics = Metrics()
    
    def snapshot_metrics(self):
        success = self.metrics.success 
        batches = self.metrics.batches 
        
        return {
            "submitted": self.metrics.submitted,
            "success": self.metrics.success,
            "timeout": self.metrics.timeout,
            "cancelled": self.metrics.cancelled,
            "batches": self.metrics.batches,
            "avg_batch_size": (
                self.metrics.total_batch_size / batches
                if batches else 0
            ),
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
        }
    
    async def start(self):
        self.scheduler_task = asyncio.create_task(self.scheduler())
        self.worker_tasks = [
            asyncio.create_task(self.worker(i))
            for i in range(self.num_workers)
        ]
    
    async def submit(self, prompt: str):
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        request_id = self.next_request_id

        self.next_request_id += 1
        
        request = Request(
            request_id=request_id,
            prompt=prompt,
            submit_time=loop.time(),
            future=future
        )
        
        await self.request_queue.put(request)
        self.metrics.submitted += 1
        print(f"submit request-{request_id}, "
        f"queue size={self.request_queue.qsize()}")
        
        try:
            return await asyncio.wait_for(future, timeout=self.request_timeout)
        except asyncio.TimeoutError:
            future.cancel()
            self.metrics.timeout += 1
            return {
                'request_id': request_id,
                'status': 'timeout'
            }
        
    async def scheduler(self):
        while True:
            batch = []
            
            try:
                while len(batch) < self.max_batch_size:
                    request = await asyncio.wait_for(self.request_queue.get(), timeout=self.batch_timeout)
                    self.request_queue.task_done()
                    batch.append(request)
            except asyncio.TimeoutError:
                pass

            if batch:
                self.metrics.batches += 1
                self.metrics.total_batch_size += len(batch)
                await self.batch_queue.put(batch)
                
    async def worker(self, worker_id: int):
        loop = asyncio.get_running_loop()
        while True:
            batch = await self.batch_queue.get()
            print(f"worker-{worker_id} "
                f"processing batch "
                f"size={len(batch)}")
            
            start = loop.time()
            for request in batch:
                request.start_time = start
                
            await asyncio.sleep(3)
            
            finish = loop.time()
            
            for request in batch:
                request.finish_time = finish
            
                queue_wait = (
                    request.start_time
                    - request.submit_time
                )
                service_time = (
                    request.finish_time
                    - request.start_time
                )
                total_latency = (
                    request.finish_time
                    - request.submit_time
                )
                if not request.future.done():
                    request.future.set_result({
                        "request_id": request.request_id,
                        "output": f"response to {request.prompt}",
                        "wait": queue_wait,
                        "service": service_time,
                        "total": total_latency,
                    })
                    self.metrics.success += 1
                    self.metrics.total_queue_wait += queue_wait
                    self.metrics.total_service_time += service_time
                    self.metrics.total_latency += total_latency
                else:
                    self.metrics.cancelled += 1
            self.batch_queue.task_done()
        
    async def shutdown(self):
        await self.request_queue.join()
        await self.batch_queue.join()
        
        if self.scheduler_task:
            self.scheduler_task.cancel()

        for task in self.worker_tasks:
            task.cancel()
        
        await asyncio.gather(
            self.scheduler_task,
            *self.worker_tasks,
            return_exceptions=True
        )

async def run_benchmark(
    num_requests: int,
    concurrency: int,
    max_batch_size: int,
    num_workers: int,
    request_timeout: float,
):
    runtime = MiniRuntime(
        max_batch_size = max_batch_size,
        num_workers = num_workers,
        request_timeout = request_timeout,
    )
    
    await runtime.start()
    
    sem = asyncio.Semaphore(concurrency)
    
    async def one_request(i: int):
        async with sem:
            return await runtime.submit(f"request-{i}")
    start = asyncio.get_running_loop().time()
    
    tasks = [
        asyncio.create_task(one_request(i))
        for i in range(num_requests)
    ]
    
    results = await asyncio.gather(*tasks)

    end = asyncio.get_running_loop().time()
    
    await runtime.shutdown()
    
    metrics = runtime.snapshot_metrics() 
    metrics['duration'] = end - start
    metrics['throughput_rps'] = num_requests / (end - start) if (end - start) > 0 else 0
    
    return results, metrics

async def main():
    _, metrics = await run_benchmark(
        num_requests=20,
        concurrency=10,
        max_batch_size=1,
        num_workers=3,
        request_timeout=30.0,
    )
    print(metrics)

if __name__ == "__main__":
    asyncio.run(main())
    
        