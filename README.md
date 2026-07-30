# Mini-Runtime

A lightweight LLM inference engine built from scratch, inspired by vLLM and SGLang.

## Features

- **Continuous Batching** — prefill + decode in a single scheduler loop
- **Batched Prefill** — multiple prefill chunks processed in one forward pass
- **Chunked Prefill** — long prompts split into bounded chunks to avoid blocking
- **Paged KV Cache** — block-based memory management with reference counting
- **Prefix Cache** — radix-tree based prefix sharing with LRU eviction
- **Native Qwen2.5 Backend** — PyTorch-native model runner with GQA + RoPE
- **Profiling** — per-step engine profiler, per-module latency breakdown, memory snapshots

## Architecture

```
Engine
 ├── admit_requests()          → prefix_cache.match + kv_cache.allocate
 ├── prefill_step()            → batch_prefill (batched forward)
 └── decode_one_step()         → batch_decode (batched forward)
      │
      └── Backend (NativeBackend)
           ├── batch_prefill()    — padded batch + attention mask
           ├── batch_decode()     — single-token batch decode
           └── Model (Qwen2)
                ├── Attention (GQA + RoPE)
                ├── MLP
                └── RMSNorm
```

## Project Structure

```
mini-runtime/
├── mini_runtime/
│   ├── engine.py              # Continuous batching engine
│   ├── request.py             # Request state machine
│   ├── configs/               # Runtime configuration
│   ├── cache/
│   │   ├── kv_cache.py        # BlockPool, KVCacheManager, BlockTable
│   │   └── prefix_cache.py    # Radix-tree prefix cache
│   ├── backend/
│   │   └── native.py          # Native Qwen2.5 backend
│   ├── model/qwen2/           # Model implementation
│   ├── profiler.py            # Engine/Memory/Module profilers
│   ├── metrics.py             # Request-level metrics
│   ├── workload.py            # Workload generators
│   └── benchmark.py           # Benchmark runner
├── benchmarks/scenarios/       # Benchmark scripts
├── tests/                     # Test suite
├── examples/                  # Usage examples
├── tools/                     # Debugging utilities
├── archive/                   # Historical prototypes
└── scripts/                   # Setup scripts
```

## Quick Start

```bash
# Install dependencies
pip install torch transformers safetensors huggingface-hub

# Run end-to-end test
PYTHONPATH=. python tests/test_e2e.py

# Run all tests
PYTHONPATH=. python tests/test_e2e.py
PYTHONPATH=. python tests/test_offset.py
PYTHONPATH=. python tests/test_evict.py
PYTHONPATH=. python tests/test_refcount.py

# Run baseline benchmark (grid scan)
PYTHONPATH=. python benchmarks/scenarios/baseline.py

# Run prefix cache experiment
PYTHONPATH=. python benchmarks/scenarios/prefix_cache.py
```

## Benchmark

```python
import asyncio
from mini_runtime.backend.native import NativeBackend
from mini_runtime.benchmark import run_continuous_benchmark

async def main():
    backend = NativeBackend("Qwen/Qwen2.5-0.5B-Instruct")
    results, metrics = await run_continuous_benchmark(
        num_requests=20, concurrency=4, max_batch_size=8,
        request_timeout=60.0, workload_kind="mixed", backend=backend,
    )
    print(metrics)

asyncio.run(main())
```

## Development

- Core engine pipeline: `engine.py` — don't add model-specific logic here
- New backends: implement `batch_prefill / batch_decode / release` in `backend/`
- New models: add under `model/` with their own `config.py`
- KV cache optimizations: extend `cache/kv_cache.py`
- Profiling: use `ModuleProfiler.record()` for new timing points
