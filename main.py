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

# 架构:
# Producer -> request_queue ->
# Scheduler -> batch_queue -> Worker
MAX_BATCH_SIZE = 4
BATCH_TIMEOUT = 0.5
async def scheduler(
    request_queue: Queue,
    batch_queue: Queue,
):
    while True:
        batch = []
        
        try:
            while len(batch) < MAX_BATCH_SIZE:
                request = await asyncio.wait_for(request_queue.get(), timeout=BATCH_TIMEOUT)
                request_queue.task_done()
                batch.append(request)
        except asyncio.TimeoutError:
            pass

        if batch:
            await batch_queue.put(batch)

async def producer(request_queue: Queue):
    loop = asyncio.get_running_loop()
    futures = []
    for i in range(10):
        future = loop.create_future()
        request = Request(request_id=i, prompt=f"request-{i}", submit_time=loop.time(), future=future)
        
        await request_queue.put(request)
        futures.append(future)
        
        print(f"submit {request.request_id}, "
        f"queue size={request_queue.qsize()}")
    
    results = await asyncio.gather(*futures)
    for result in results:
        print(result)

    
async def worker(worker_id: int, batch_queue: Queue):
    loop = asyncio.get_running_loop()
    while True:
        batch = await batch_queue.get()
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
        batch_queue.task_done()

async def main():
    request_queue = asyncio.Queue()
    batch_queue = asyncio.Queue() 
    producer_task = asyncio.create_task(producer(request_queue))
    scheduler_task = asyncio.create_task(scheduler(request_queue, batch_queue))
    worker_tasks = [asyncio.create_task(worker(i, batch_queue)) for i in range(3)]
    
    await producer_task
    await request_queue.join()
    await batch_queue.join()
    
    scheduler_task.cancel()
    
    for task in worker_tasks:
        task.cancel()

if __name__ == "__main__":
    asyncio.run(main())
    
        