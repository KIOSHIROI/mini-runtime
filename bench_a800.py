"""A800 Benchmark: expanded grid for 80GB GPU.

Usage:
    python bench_a800.py
    python bench_a800.py --small   # 快速冒烟测试（仅 8 组）
    python bench_a800.py --large   # 完整扫描（56 组）
"""

import asyncio
import time
import csv
import argparse
import torch
from mini_runtime.backends.native_backend import NativeBackend
from mini_runtime.continuous_engine import Engine


# ── prompt 构造 ──────────────────────────────────────────────

def build_prompt(backend: NativeBackend, target_tokens: int) -> str:
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

    ttfts = [r["ttft"] for r in results if "ttft" in r and r["ttft"] is not None]
    tpots = [r["tpot"] for r in results if "tpot" in r]
    gen_tokens = sum(r.get("generated_tokens", 0) for r in results)

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


# ── Grid 定义 ─────────────────────────────────────────────────

def get_grid(mode: str) -> list[tuple[int, int, int, int]]:
    """返回 (num_requests, max_batch_size, target_prompt, max_new_tokens) 列表"""
    if mode == "small":
        return [
            (4,  4,    128,   64),
            (4,  4,   2048,   64),
            (8,  8,    512,  128),
            (8,  8,   4096,  128),
            (16, 16,   128,   64),
            (16, 16,  2048,   64),
            (32, 16,   512,  256),
            (32, 32,  4096,  256),
        ]
    elif mode == "medium":
        return [
            # 小 batch，扫 prompt 长度
            (8,  8,    128,  128),
            (8,  8,    512,  128),
            (8,  8,   2048,  128),
            (8,  8,   4096,  128),
            (8,  8,   8192,  128),
            # 中 batch，扫 prompt 长度
            (16, 16,   128,  256),
            (16, 16,   512,  256),
            (16, 16,  2048,  256),
            (16, 16,  4096,  256),
            (16, 16,  8192,  256),
            # 大 batch
            (32, 32,   128,  128),
            (32, 32,   512,  128),
            (32, 32,  2048,  128),
            (32, 32,  4096,  128),
            # 请求数 >> batch，测试排队
            (32,  8,   512,  128),
            (64,  8,   512,  128),
            (32, 16,   512,  256),
            (64, 16,   512,  256),
            # 高并发 + 长 prompt
            (32, 32,  4096,  256),
            (64, 32,  2048,  128),
        ]
    else:
        # large — 全覆盖
        grid = []
        for num_req in [8, 16, 32, 64]:
            for batch in [4, 8, 16, 32]:
                if batch > num_req:
                    continue
                for prompt_len in [128, 512, 2048, 4096, 8192]:
                    for gen_len in [64, 128, 256]:
                        grid.append((num_req, batch, prompt_len, gen_len))
        return grid


# ── 主流程 ────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--small", action="store_true")
    parser.add_argument("--medium", action="store_true")
    parser.add_argument("--large", action="store_true")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    args = parser.parse_args()

    if args.small:
        mode = "small"
    elif args.medium:
        mode = "medium"
    elif args.large:
        mode = "large"
    else:
        mode = "medium"  # 默认 medium

    grid = get_grid(mode)
    print(f"Grid mode: {mode} ({len(grid)} experiments)")
    print(f"Loading model: {args.model} ...")
    backend = NativeBackend(args.model)

    # 预构建 prompts（同长度复用）
    prompt_cache = {}

    rows = []
    for num_req, batch, target_prompt, max_tok in grid:
        if target_prompt not in prompt_cache:
            print(f"Building prompt ~{target_prompt} tokens ...")
            prompt_cache[target_prompt] = build_prompt(backend, target_prompt)
        prompt = prompt_cache[target_prompt]

        label = f"[{num_req:>2} reqs, batch={batch:>2}, prompt~{target_prompt:>4}, gen={max_tok:>3}]"
        print(f"  {label} ", end="", flush=True)
        t_start = time.perf_counter()
        row = await run_one(backend, prompt, num_req, max_tok, batch, request_timeout=120.0)
        row["wall_s"] = time.perf_counter() - t_start
        rows.append(row)
        print(f"→ TTFT={row['ttft_avg_ms']:.0f}ms  TPOT={row['tpot_avg_ms']:.0f}ms  "
              f"throughput={row['throughput_tok_s']:.1f} tok/s  "
              f"prefill={row['avg_prefill_ms']:.1f}ms  decode={row['avg_decode_ms']:.1f}ms  "
              f"OOM={row['oom']}")

    # ── 打印表格 ──
    header = (f"{'reqs':>4} {'batch':>5} {'prompt':>7} {'gen':>4}  "
              f"{'TTFT':>7} {'TPOT':>7} {'tok/s':>8} {'prefill':>8} {'decode':>7} {'oom':>3}")
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
    csv_name = f"bench_a800_{mode}.csv"
    fieldnames = ["num_requests", "max_batch_size", "prompt_tokens", "max_new_tokens",
                  "ttft_avg_ms", "tpot_avg_ms", "throughput_tok_s", "elapsed_s", "wall_s",
                  "avg_prefill_ms", "avg_decode_ms", "prefill_batches", "decode_steps",
                  "success", "oom"]
    with open(csv_name, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved {len(rows)} rows → {csv_name}")


if __name__ == "__main__":
    asyncio.run(main())
