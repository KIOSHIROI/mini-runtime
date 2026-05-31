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

class MiniRuntime:
    def __init__(
        self,
        max_batch_size: int = 4,
        batch_timeout: float = 0.5,
        request_timeout: float = 5.0,
        nums_workers: int = 3,
     ):
        self.request_queue = asyncio.Queue()
        self.batch_queue = asyncio.Queue()
        self.max_batch_size = max_batch_size
        self.batch_timeout = batch_timeout
        self.request_timeout = request_timeout
        self.nums_workers = nums_workers
        self.scheduler_task = None
        self.worker_tasks = []  
        self.next_request_id = 0
    
    async def start(self):
        self.scheduler_task = asyncio.create_task(self.scheduler())
        self.worker_tasks = [
            asyncio.create_task(self.worker(i))
            for i in range(self.nums_workers)
        ]
    
    async def submit(self, prompt: str):
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        request_id = self.next_request_id
        print(f"submit request-{request_id}, "
              f"queue size={self.request_queue.qsize()}")
        self.next_request_id += 1
        
        request = Request(
            request_id=request_id,
            prompt=prompt,
            submit_time=loop.time(),
            future=future
        )
        
        await self.request_queue.put(request)
        
        try:
            return await asyncio.wait_for(future, timeout=self.request_timeout)
        except asyncio.TimeoutError:
            future.cancel()
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
            self.batch_queue.task_done()
        
    async def shutdown(self):
        await self.request_queue.join()
        await self.batch_queue.join()
        
        if self.scheduler_task:
            self.scheduler_task.cancel()

        for task in self.worker_tasks:
            task.cancel()
async def wait_result(request_id: int, future: asyncio.Future):
    try:
        result = await asyncio.wait_for(future, timeout=REQUEST_TIMEOUT)
        return 'success', result
    except asyncio.TimeoutError:
        return 'timeout', {"request_id": request_id}
    except asyncio.CancelledError:
        return 'cancelled', {"request_id": request_id}

async def producer(request_queue: Queue):
    loop = asyncio.get_running_loop()
    wait_tasks = []
    
    for i in range(10):
        future = loop.create_future()
        request = Request(request_id=i, prompt=f"request-{i}", submit_time=loop.time(), future=future)
        
        await request_queue.put(request)
        wait_tasks.append(asyncio.create_task(wait_result(request.request_id, future)))
        
        print(f"submit {request.request_id}, "
        f"queue size={request_queue.qsize()}")
    
    results = await asyncio.gather(*wait_tasks)
    
    for status, result in results:
        if status == 'success':
            print(f"request {result['request_id']} completed, "
                  f"wait={result['wait']:.2f}s, "
                  f"service={result['service']:.2f}s, "
                  f"total={result['total']:.2f}s")
        else:
            print(f"request {result['request_id']} {status}")
    success = sum(1 for status, _ in results if status == 'success')
    timeout = sum(1 for status, _ in results if status == 'timeout')
    cancelled = sum(1 for status, _ in results if status == 'cancelled')
    
    print(f"Results - Success: {success}, Timeout: {timeout}, Cancelled: {cancelled}")


async def main():
    runtime = MiniRuntime()
    await runtime.start()
    
    tasks = [
        asyncio.create_task(runtime.submit(f"request-{i}"))
        for i in range(10)
    ]
    
    results = await asyncio.gather(*tasks)
    for result in results:
        print(result)
    
    await runtime.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
    
        