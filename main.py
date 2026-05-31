import asyncio 
from asyncio import Queue 


async def producer(queue: Queue):
    for i in range(10):
        await queue.put(f"request-{i}")
        print(f"submit {i}")
        await asyncio.sleep(1)
    
async def worker(queue: Queue):
    while True:
        request = await queue.get()
        print(f"processing {request}")
        await asyncio.sleep(2)
        print(f"finish {request}")
        queue.task_done()

async def main():
    queue = asyncio.Queue() 
    producer_task = asyncio.create_task(producer(queue))
    worker_task = asyncio.create_task(worker(queue))
    
    await producer_task
    await worker_task
    
    
if __name__ == "__main__":
    asyncio.run(main())
    
        