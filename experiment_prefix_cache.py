"""
Prefix Cache 验证实验:
  - 实验 A: 无缓存（每个请求用不同的 prompt）
  - 实验 B: 有缓存（所有请求用相同的 prompt）

通过对比 A 和 B 的吞吐/延迟，验证 Prefix Cache 的效果。
"""
import asyncio
from mini_runtime.backends.native_backend import NativeBackend
from mini_runtime.continuous_engine import Engine

PROMPT_SHARED = "介绍一下杭州这座城市，包括它的地理位置、历史文化、经济发展和旅游资源。"

PROMPT_UNIQUE = [
    "介绍一下杭州这座城市，包括它的地理位置、历史文化、经济发展和旅游资源。",
    "介绍一下北京这座城市，包括它的地理位置、历史文化、经济发展和旅游资源。",
    "介绍一下上海这座城市，包括它的地理位置、历史文化、经济发展和旅游资源。",
    "介绍一下深圳这座城市，包括它的地理位置、历史文化、经济发展和旅游资源。",
    "介绍一下广州这座城市，包括它的地理位置、历史文化、经济发展和旅游资源。",
    "介绍一下成都这座城市，包括它的地理位置、历史文化、经济发展和旅游资源。",
    "介绍一下武汉这座城市，包括它的地理位置、历史文化、经济发展和旅游资源。",
    "介绍一下南京这座城市，包括它的地理位置、历史文化、经济发展和旅游资源。",
    "介绍一下西安这座城市，包括它的地理位置、历史文化、经济发展和旅游资源。",
    "介绍一下重庆这座城市，包括它的地理位置、历史文化、经济发展和旅游资源。",
]


async def run_experiment(name: str, prompts: list[str],
                         max_batch_size: int = 4, concurrency: int = 4,
                         max_new_tokens: int = 16, request_timeout: float = 120.0):
    """每个实验创建独立的 Backend，避免状态污染"""
    backend = NativeBackend("Qwen/Qwen2.5-0.5B-Instruct")
    engine = Engine(
        backend=backend,
        max_batch_size=max_batch_size,
        request_timeout=request_timeout,
    )
    await engine.start()

    sem = asyncio.Semaphore(concurrency)

    async def one_request(i: int):
        async with sem:
            return await engine.submit(prompts[i], max_new_tokens=max_new_tokens)

    loop = asyncio.get_running_loop()
    start = loop.time()

    tasks = [asyncio.create_task(one_request(i)) for i in range(len(prompts))]
    results = await asyncio.gather(*tasks)

    end = loop.time()
    duration = end - start

    await engine.shutdown()

    metrics = engine.snapshot_metrics()
    metrics["experiment"] = name
    metrics["duration"] = duration
    metrics["throughput_rps"] = len(prompts) / duration if duration > 0 else 0
    metrics["num_requests"] = len(prompts)
    success = sum(1 for r in results if isinstance(r, dict) and "error" not in r)
    metrics["success_count"] = success

    kv = metrics.get("kv_cache", {})
    metrics["max_allocated_blocks"] = kv.get("max_allocated", 0)
    metrics["total_blocks"] = kv.get("total_blocks", 0)

    return metrics


async def main():
    print("=" * 60)
    print("Prefix Cache 验证实验")
    print("=" * 60)

    # ====== 实验 A: 无缓存 ======
    print("\n--- 实验 A: 无缓存（不同 prompt）---")
    metrics_a = await run_experiment(
        "A: 无缓存", PROMPT_UNIQUE[:10],
        max_batch_size=4, concurrency=4, max_new_tokens=16,
    )

    # ====== 实验 B: 有缓存 ======
    print("--- 实验 B: 有缓存（相同 prompt）---")
    prompts_shared = [PROMPT_SHARED] * 10
    metrics_b = await run_experiment(
        "B: 有缓存", prompts_shared,
        max_batch_size=4, concurrency=4, max_new_tokens=16,
    )

    # ====== 结果对比 ======
    print("\n" + "=" * 60)
    print("结果对比")
    print("=" * 60)

    for label, m in [("无缓存", metrics_a), ("有缓存", metrics_b)]:
        print(f"\n{label}:")
        print(f"  请求数:       {m['num_requests']}")
        print(f"  成功数:       {m.get('success_count', m.get('submitted', 0))}")
        print(f"  总耗时:       {m['duration']:.2f}s")
        print(f"  吞吐:         {m['throughput_rps']:.4f} req/s")
        print(f"  平均延迟:     {m.get('avg_latency', 0):.4f}s")
        print(f"  平均 TTFT:    {m.get('avg_ttft', 0):.4f}s")
        print(f"  平均 TPOT:    {m.get('avg_tpot', 0):.4f}s")
        print(f"  Prefill 批数:  {m.get('prefill_batches', 0)}")
        print(f"  Decode 步数:   {m.get('decode_steps', 0)}")
        print(f"  峰值 block 数: {m.get('max_allocated_blocks', 0)}/{m.get('total_blocks', 0)}")

    tp_a = metrics_a['throughput_rps']
    tp_b = metrics_b['throughput_rps']
    if tp_b > 0:
        ratio = tp_b / tp_a
        print(f"\n吞吐对比: 有缓存 / 无缓存 = {ratio:.2f}x")
        if ratio > 1:
            print(f"→ Prefix Cache 吞吐提升 {(ratio - 1) * 100:.1f}%")


if __name__ == "__main__":
    asyncio.run(main())
