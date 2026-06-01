import asyncio
from mini_runtime.continuous_engine import ContinuousBatchingEngine
from mini_runtime.request import Request
from mini_runtime.metrics import Metrics

async def main():
    engine = ContinuousBatchingEngine(max_batch_size=4)
    await engine.start()
    
    tasks = [
        asyncio.create_task(
            engine.submit(
                f"request-{i}",
                prompt_tokens=16,
                max_new_tokens=16,
            )
        )
        for i in range(10)
    ]
    
    results = await asyncio.gather(*tasks)
    for r in results:
        print(r)
        
    await engine.shutdown()
    
if __name__ == "__main__":
    asyncio.run(main())