import asyncio 
from asyncio import Queue 


async def producer(queue: Queue):
    for i in range(10):
        await queue.put(f"request-{i}")
        print(f"submit {i}")
        await asyncio.sleep(1)
    
async def worker(worker_id: int, queue: Queue):
    while True:
        request = await queue.get()
        print(f"Worker-{worker_id} processing {request}")
        await asyncio.sleep(3)
        print(f"Worker-{worker_id} finish {request}")
        queue.task_done()

async def main():
    queue = asyncio.Queue() 
    producer_task = asyncio.create_task(producer(queue))
    [asyncio.create_task(worker(i, queue)) for i in range(3)]
    
    await producer_task
    await queue.join()
    
    
if __name__ == "__main__":
    asyncio.run(main())
    
        