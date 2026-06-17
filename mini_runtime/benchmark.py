from .runtime import MiniRuntime
from .continuous_engine import ContinuousBatchingEngine
from .workload import make_workload
from .backends.native_backend import NativeBackend
import asyncio
import csv

async def run_benchmark(
    num_requests: int,
    concurrency: int,
    max_batch_size: int,
    num_workers: int,
    request_timeout: float,
    workload_kind: str,
):
    runtime = MiniRuntime(
        max_batch_size = max_batch_size,
        num_workers = num_workers,
        request_timeout = request_timeout,
    )
    
    await runtime.start()
    
    sem = asyncio.Semaphore(concurrency)
    
    workload = make_workload(workload_kind, num_requests)
    
    async def one_request(i: int):
        item = workload[i]
        async with sem:
            return await runtime.submit(
                f"request-{i}",
                prompt_tokens=item["prompt_tokens"],
                max_new_tokens=item["max_new_tokens"],
            )
            
    start = asyncio.get_running_loop().time()
    
    tasks = [
        asyncio.create_task(one_request(i))
        for i in range(num_requests)
    ]
    
    results = await asyncio.gather(*tasks)

    end = asyncio.get_running_loop().time()
    
    await runtime.shutdown()
    
    metrics = runtime.snapshot_metrics() 
    metrics['workload'] = workload_kind
    metrics["max_batch_size"] = max_batch_size
    metrics["num_workers"] = num_workers
    metrics["concurrency"] = concurrency
    metrics['duration'] = end - start
    metrics['throughput_rps'] = num_requests / (end - start) if (end - start) > 0 else 0
    
    return results, metrics

async def run_continuous_benchmark(
    num_requests: int,
    concurrency: int,
    max_batch_size: int,
    request_timeout: float,
    workload_kind: str,
    backend: NativeBackend,
):
    engine = ContinuousBatchingEngine(
        backend = backend,
        max_batch_size=max_batch_size,
        request_timeout=request_timeout,
    )
    await engine.start()
    
    sem = asyncio.Semaphore(concurrency)
    workload = make_workload(workload_kind, num_requests)
    
    async def one_request(i: int):
        item = workload[i]
        async with sem:
            return await engine.submit(
                f"request-{i}",
                max_new_tokens=item["max_new_tokens"],
            )
    
    loop = asyncio.get_running_loop()
    start = loop.time()
    
    tasks = [
        asyncio.create_task(one_request(i))
        for i in range(num_requests)
    ]
    
    results = await asyncio.gather(*tasks)
    
    end = loop.time()
    
    await engine.shutdown()
    
    metrics = engine.snapshot_metrics()
    metrics['engine'] = "continuous"
    metrics['workload'] = workload_kind
    metrics["max_batch_size"] = max_batch_size
    metrics["concurrency"] = concurrency
    metrics['duration'] = end - start
    metrics['throughput_rps'] = (
        num_requests / (end - start) 
        if end > start else 0
    )
    
    return results, metrics

def write_metrics_csv(path: str, rows: list[dict]):
    if not rows:
        return

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)