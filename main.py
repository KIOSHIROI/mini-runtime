import asyncio 
from mini_runtime.benchmark import run_benchmark, write_metrics_csv

async def main():
    all_metrics = []
    # n_r, concur, bs, n_w, to
    configs = [
        (20, 10, 4, 3, 30.0, "spso"),
        (20, 10, 4, 3, 30.0, "lpso"),
        (20, 10, 4, 3, 30.0, "splo"),
        (20, 10, 4, 3, 30.0, "mixed"),
    ]
    for config in configs:
        _, metrics = await run_benchmark(
            num_requests=config[0],
            concurrency=config[1],
            max_batch_size=config[2],
            num_workers=config[3],
            request_timeout=config[4],
            workload_kind=config[5],
        )
        all_metrics.append(metrics)

    for metrics in all_metrics:
        print(metrics)
       
    write_metrics_csv("benchmark_results.csv", all_metrics) 

if __name__ == "__main__":
    asyncio.run(main())
    
        