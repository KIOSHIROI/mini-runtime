import asyncio
from mini_runtime.continuous_engine import ContinuousBatchingEngine
from mini_runtime.benchmark import run_benchmark ,run_continuous_benchmark, write_metrics_csv

async def main():
    configs = [
        (20, 10, 4, 30.0, "spso"),
        (20, 10, 4, 30.0, "lpso"),
        (20, 10, 4, 30.0, "splo"),
        (20, 10, 4, 30.0, "mixed"),
    ]
    
    rows = []
    for config in configs:
        results, metrics = await run_continuous_benchmark(*config)
        rows.append(metrics)
    
    for config in configs:
        results, metrics = await run_benchmark(
            num_requests=config[0],
            concurrency=config[1],
            max_batch_size=config[2],
            num_workers=3,
            request_timeout=config[3],
            workload_kind=config[4],
        )
        rows.append(metrics)
        
    # for metrics in rows:
    #     print(metrics)

    # Ensure CSV header covers all metric fields across rows.
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    for row in rows:
        for key in fieldnames:
            row.setdefault(key, "")

    write_metrics_csv("continuous_metrics.csv", rows)

if __name__ == "__main__":
    asyncio.run(main())