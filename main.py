import asyncio 
from asyncio import Queue
from dataclasses import dataclass 

@dataclass
class Request:
    request_id: int
    prompt: str
    submit_time: float
    
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
    for i in range(10):
        request = Request(request_id=i, prompt=f"request-{i}", submit_time=asyncio.get_event_loop().time())
        await request_queue.put(request)
        print(f"submit {request.request_id}, "
              f"queue size={request_queue.qsize()}")
        await asyncio.sleep(0.1)
    
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
            print(
                f"[{request.request_id}] "
                f"wait={queue_wait:.2f}s "
                f"service={service_time:.2f}s "
                f"total={total_latency:.2f}s"
            )
        batch_queue.task_done()

async def main():
    request_queue = asyncio.Queue()
    batch_queue = asyncio.Queue() 
    producer_task = asyncio.create_task(producer(request_queue))
    scheduler_task = asyncio.create_task(scheduler(request_queue, batch_queue))
    [asyncio.create_task(worker(i, batch_queue)) for i in range(3)]
    
    await producer_task
    await request_queue.join()
    await batch_queue.join()
    
    scheduler_task.cancel()
    
    
if __name__ == "__main__":
    asyncio.run(main())
    
        