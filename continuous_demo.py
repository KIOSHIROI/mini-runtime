import asyncio
from mini_runtime.continuous_engine import ContinuousBatchingEngine
from mini_runtime.benchmark import run_benchmark ,run_continuous_benchmark, write_metrics_csv
from mini_runtime.backends.native_backend import NativeBackend
async def main():
    backend = NativeBackend("~/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct")
    configs = [
        (20, 10, 4, 30.0, "spso", backend),
        # (20, 10, 4, 30.0, "lpso", backend),
        # (20, 10, 4, 30.0, "splo", backend),
        # (20, 10, 4, 30.0, "mixed", backend),
    ]
    
    rows = []
    for config in configs:
        results, metrics = await run_continuous_benchmark(*config)
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