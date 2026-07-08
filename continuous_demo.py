import asyncio
import os
import csv
from datetime import datetime
from mini_runtime.backends.native_backend import NativeBackend
from mini_runtime.benchmark import run_benchmark, run_continuous_benchmark, write_metrics_csv

async def main():
    backend = NativeBackend("Qwen/Qwen2.5-0.5B-Instruct")

    name = input("实验名称: ").strip()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    csv_path = "continuous_metrics.csv"

    # 去重：检查已有 CSV 中的实验名，重名自动加 (n)
    if os.path.exists(csv_path):
        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            existing_names = [row.get("experiment_name", "") for row in reader]
        base = name
        n = 1
        while name in existing_names:
            name = f"{base}({n})"
            n += 1

    configs = [
        (20, 16, 16, 30.0, "spso", backend),
        # (20, 10, 4, 30.0, "lpso", backend),
        # (20, 10, 4, 30.0, "splo", backend),
        # (20, 10, 4, 30.0, "mixed", backend),
    ]

    rows = []
    for config in configs:
        results, metrics = await run_continuous_benchmark(*config)
        rows.append(metrics)

    # 加实验元信息
    for row in rows:
        row["experiment_name"] = name
        row["timestamp"] = timestamp

    write_metrics_csv(csv_path, rows, mode="a")

if __name__ == "__main__":
    asyncio.run(main())
