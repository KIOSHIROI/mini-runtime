"""Baseline Benchmark: 遍历 batch/prompt/gen 组合，收集延迟和吞吐数据。"""
import asyncio
import time
import csv
import torch
from mini_runtime.backend.native import NativeBackend
from mini_runtime.engine import Engine


# ── prompt 构造 ──────────────────────────────────────────────

def build_prompt(backend: NativeBackend, target_tokens: int) -> str:
    """构造一个接近 target_tokens 长度的 prompt（含 chat template）。"""
    filler = "The quick brown fox jumps over the lazy dog. "
    prompt = filler
    while True:
        chat_text = backend.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True,
        )
        n = len(backend.tokenizer.encode(chat_text))
        if n >= target_tokens:
            return prompt
        prompt += filler


# ── 单次实验 ──────────────────────────────────────────────────

async def run_one(
    backend: NativeBackend,
    prompt: str,
    num_requests: int,
    max_new_tokens: int,
    max_batch_size: int,
    request_timeout: float,
) -> dict:
    # 清理 backend 状态（共享 backend 时避免上轮残留）
    backend._past_len.clear()
    backend._generated.clear()

    engine = Engine(
        backend=backend,
        max_batch_size=max_batch_size,
        request_timeout=request_timeout,
    )
    await engine.start()

    loop = asyncio.get_running_loop()
    t0 = loop.time()
    tasks = [
        asyncio.create_task(engine.submit(prompt, max_new_tokens=max_new_tokens))
        for _ in range(num_requests)
    ]
    results = await asyncio.gather(*tasks)
    elapsed = loop.time() - t0

    # 请求级指标
    ttfts = [r["ttft"] for r in results if "ttft" in r and r["ttft"] is not None]
    tpots = [r["tpot"] for r in results if "tpot" in r]
    gen_tokens = sum(r.get("generated_tokens", 0) for r in results)

    # 引擎级指标
    m = engine.snapshot_metrics()
    ep = engine.engine_profiler
    steps = ep.steps

    prefill_ms = sum(s.prefill_ms for s in steps) / len(steps) if steps else 0
    decode_ms = sum(s.decode_ms for s in steps) / len(steps) if steps else 0

    await engine.shutdown()
    backend.kv_manager = None  # 释放旧 Engine 的 KV cache 引用，避免 GPU 显存泄漏
    torch.cuda.empty_cache()

    return {
        "num_requests": num_requests,
        "max_batch_size": max_batch_size,
        "max_new_tokens": max_new_tokens,
        "prompt_tokens": len(backend.tokenizer.encode(
            backend.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False, add_generation_prompt=True,
            )
        )),
        "success": m["success"],
        "oom": m["oom"],
        "ttft_avg_ms": (sum(ttfts) / len(ttfts) * 1000) if ttfts else 0,
        "tpot_avg_ms": (sum(tpots) / len(tpots) * 1000) if tpots else 0,
        "throughput_tok_s": gen_tokens / elapsed if elapsed > 0 else 0,
        "elapsed_s": elapsed,
        "prefill_batches": m["prefill_batches"],
        "decode_steps": m["decode_steps"],
        "avg_prefill_ms": prefill_ms,
        "avg_decode_ms": decode_ms,
    }


# ── 主流程 ────────────────────────────────────────────────────

async def main():
    model_path = "Qwen/Qwen2.5-0.5B-Instruct"
    print(f"Loading model: {model_path} ...")
    backend = NativeBackend(model_path)
    prompt_tokens = len(backend.tokenizer.encode(
        backend.tokenizer.apply_chat_template(
            [{"role": "user", "content": "Hello"}],
            tokenize=False, add_generation_prompt=True,
        )
    )) - 1  # rough base overhead

    # ── 参数扫描 ──
    grid = [
        # (num_requests, max_batch_size, target_prompt, max_new_tokens)
        (4,  2,   128,   32),
        (4,  2,   512,   32),
        (4,  2,  2048,   32),
        (4,  4,   128,   64),
        (4,  4,   512,   64),
        (4,  4,  2048,   64),
        (8,  4,   128,   32),
        (8,  4,   512,   32),
        (8,  4,  2048,   32),
        (8,  8,   128,   64),
        (8,  8,   512,   64),
        (8,  8,  2048,   64),
        (16, 8,   128,   32),
        (16, 8,   512,   32),
        (16, 16,  128,   64),
        (16, 16,  512,   64),
    ]

    # 预构建 prompts（同长度复用）
    prompt_cache = {}

    rows = []
    for num_req, batch, target_prompt, max_tok in grid:
        if target_prompt not in prompt_cache:
            print(f"Building prompt ~{target_prompt} tokens ...")
            prompt_cache[target_prompt] = build_prompt(backend, target_prompt)
        prompt = prompt_cache[target_prompt]

        print(f"  [{num_req} reqs, batch={batch}, prompt~{target_prompt}, gen={max_tok}] ", end="", flush=True)
        t_start = time.perf_counter()
        row = await run_one(backend, prompt, num_req, max_tok, batch, request_timeout=60.0)
        row["wall_s"] = time.perf_counter() - t_start
        rows.append(row)
        print(f"→ TTFT={row['ttft_avg_ms']:.0f}ms  TPOT={row['tpot_avg_ms']:.0f}ms  "
              f"throughput={row['throughput_tok_s']:.1f} tok/s  "
              f"prefill={row['avg_prefill_ms']:.1f}ms  decode={row['avg_decode_ms']:.1f}ms")

    # ── 打印表格 ──
    header = f"{'reqs':>4} {'batch':>5} {'prompt':>7} {'gen':>4}  {'TTFT':>7} {'TPOT':>7} {'tok/s':>8} {'prefill':>8} {'decode':>7} {'oom':>3}"
    sep = "-" * len(header)
    print("\n" + sep)
    print(header)
    print(sep)
    for r in rows:
        print(f"{r['num_requests']:>4} {r['max_batch_size']:>5} {r['prompt_tokens']:>7} {r['max_new_tokens']:>4}  "
              f"{r['ttft_avg_ms']:>6.0f}ms {r['tpot_avg_ms']:>6.0f}ms {r['throughput_tok_s']:>7.1f} "
              f"{r['avg_prefill_ms']:>7.1f}ms {r['avg_decode_ms']:>6.1f}ms {r['oom']:>3}")
    print(sep)

    # ── 写 CSV ──
    fieldnames = ["num_requests", "max_batch_size", "prompt_tokens", "max_new_tokens",
                  "ttft_avg_ms", "tpot_avg_ms", "throughput_tok_s", "elapsed_s", "wall_s",
                  "avg_prefill_ms", "avg_decode_ms", "prefill_batches", "decode_steps",
                  "success", "oom"]
    with open("baseline_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved {len(rows)} rows → baseline_results.csv")


if __name__ == "__main__":
    asyncio.run(main())
