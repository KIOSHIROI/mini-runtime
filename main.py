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

async def producer(queue: Queue):
    for i in range(10):
        request = Request(request_id=i, prompt=f"request-{i}", submit_time=asyncio.get_event_loop().time())
        await queue.put(request)
        print(f"submit {request.request_id}")
        await asyncio.sleep(1)
    
async def worker(worker_id: int, queue: Queue):
    loop = asyncio.get_running_loop()
    while True:
        request = await queue.get()
        request.start_time = loop.time()
        print(f"Worker-{worker_id} processing {request.request_id}")
        await asyncio.sleep(3)
        request.finish_time = loop.time()
        # print(f"Worker-{worker_id} finish {request}")
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
        queue.task_done()

async def main():
    queue = asyncio.Queue() 
    producer_task = asyncio.create_task(producer(queue))
    [asyncio.create_task(worker(i, queue)) for i in range(3)]
    
    await producer_task
    await queue.join()
    
    
if __name__ == "__main__":
    asyncio.run(main())
    
        