import asyncio
from mini_runtime.continuous_engine import ContinuousBatchingEngine

async def run_scene(name: str, num_blocks: int, block_size: int,
                    num_requests: int, max_batch_size: int,
                    prompt_tokens: int, max_new_tokens: int,
                    request_timeout: float = 10.0):
    line = "─" * 56
    print(f"\n{line}")
    print(f"  {name}")
    print(f"  pool={num_blocks}B × {block_size}tok = {num_blocks * block_size}tok  "
          f"|  batch={max_batch_size}  |  reqs={num_requests}")
    print(f"  {prompt_tokens} prompt + {max_new_tokens} output "
          f"→ ceil({prompt_tokens + max_new_tokens}/{block_size}) "
          f"= {-( -(prompt_tokens + max_new_tokens) // block_size)} blocks/req")
    print(f"  max concurrent = {num_blocks // max(1, -( -(prompt_tokens + max_new_tokens) // block_size))} reqs")
    print(line)

    engine = ContinuousBatchingEngine(
        max_batch_size=max_batch_size,
        request_timeout=request_timeout,
        num_blocks=num_blocks,
        block_size=block_size,
    )
    await engine.start()

    async def one_request(i: int):
        return await engine.submit(
            f"req-{i}",
            prompt_tokens=prompt_tokens,
            max_new_tokens=max_new_tokens,
        )

    tasks = [asyncio.create_task(one_request(i)) for i in range(num_requests)]
    results = await asyncio.gather(*tasks)
    await engine.shutdown()

    m = engine.snapshot_metrics()
    kv = m["kv_cache"]

    print(f"  submitted={m['submitted']:>4}  success={m['success']:>4}  "
          f"timeout={m['timeout']:>4}  oom={m['oom']:>4}  cancelled={m['cancelled']}")
    print(f"  avg_ttft={m['avg_ttft']:.4f}s  avg_tpot={m['avg_tpot']:.4f}s  "
          f"output_tok/s={m['output_tokens_per_sec']:.1f}")
    print(f"  kv_cache: peak={kv['max_allocated']}/{kv['total_blocks']} blocks  "
          f"({kv['max_allocated'] / kv['total_blocks']:.0%})  "
          f"final_free={len(kv['free_blocks'])}")
    return m


async def main():
    # ── 场景一：单请求所需 block 超过整个池，admission 即 OOM ──
    # 10 blocks × 16 = 160 tokens 容量
    # 256 prompt → ceil(256/16) = 16 blocks → 第一个请求就超出池大小
    # 预期：success=0, oom≥8 (每轮 admission 尝试都失败)
    await run_scene(
        name="场景1: 单请求超池 — admission 直接 OOM",
        num_blocks=10, block_size=16,
        num_requests=8, max_batch_size=4,
        prompt_tokens=256, max_new_tokens=16,
    )

    # ── 场景二：批量 admission 填满池子，后续等待释放 ─────
    # 30 blocks × 16 = 480 tokens
    # 每请求: 32 prompt → 2 blocks 初始, decode 到 48 → 3 blocks 峰值
    # 30/3 = 10 个请求可同时跑，发 30 个请求
    # 预期：全部最终成功（老请求释放后新请求接上），max_allocated 接近 30
    await run_scene(
        name="场景2: 池满后等待 — block 释放后被复用",
        num_blocks=30, block_size=16,
        num_requests=30, max_batch_size=10,
        prompt_tokens=32, max_new_tokens=16,
    )

    # ── 场景三：大池子，全部畅通 ─────────────────────────
    # 200 blocks × 16 = 3200 tokens
    # 20 个 spso 请求 × 3 blocks = 60 << 200
    # 预期：全部成功，utilization < 5%
    await run_scene(
        name="场景3: 池远大于需求 — 全部畅通",
        num_blocks=200, block_size=16,
        num_requests=20, max_batch_size=4,
        prompt_tokens=32, max_new_tokens=16,
    )

    # ── 场景四：高并发 + 短超时 → 部分请求超时 ────────────
    # 20 blocks × 16 = 320 tokens
    # 每请求: 64 prompt + 0 output = 4 blocks, batch_size=10
    # 20/4 = 5 个请求同时跑, 100 个请求排队
    # timeout=3s, 每批 prefill = 64*0.01 = 0.64s
    # 100/5 = 20 批 * 0.64s = 12.8s > 3s → 排后面的请求超时
    await run_scene(
        name="场景4: 小池+高负载→大量超时",
        num_blocks=20, block_size=16,
        num_requests=80, max_batch_size=10,
        prompt_tokens=64, max_new_tokens=0,
        request_timeout=3.0,
    )

    print(f"\n{'─' * 56}")
    print("  All scenes done.")


if __name__ == "__main__":
    asyncio.run(main())
