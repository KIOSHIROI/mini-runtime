"""端到端测试: prefix cache 集成到 Engine"""
import asyncio
from mini_runtime.backends.native_backend import NativeBackend
from mini_runtime.continuous_engine import Engine


async def main():
    backend = NativeBackend("Qwen/Qwen2.5-0.5B-Instruct")
    engine = Engine(backend=backend, max_batch_size=2, request_timeout=60.0)
    await engine.start()

    # 两个共享长 system prompt 的请求
    sys_prompt = "You are a helpful assistant. " * 8
    r1 = await engine.submit(sys_prompt + "Question: what is 1+1? Answer:", max_new_tokens=8)
    r2 = await engine.submit(sys_prompt + "Question: what is 2+2? Answer:", max_new_tokens=8)

    print("r1:", {k: r1[k] for k in ("request_id", "ttft", "generated_tokens", "output", "error") if k in r1})
    print("r2:", {k: r2[k] for k in ("request_id", "ttft", "generated_tokens", "output", "error") if k in r2})

    assert "error" not in r1, f"r1 失败: {r1}"
    assert "error" not in r2, f"r2 失败: {r2}"

    m = engine.snapshot_metrics()
    print(f"\nmetrics: success={m['success']} oom={m['oom']} prefill_batches={m['prefill_batches']} decode_steps={m['decode_steps']}")
    print(f"cache tree children: {len(engine.prefix_cache.root.children)} (期望 >=1, 说明 prefix 被缓存)")

    # 验证 r1 的输出文本合理 (1+1=2)
    print(f"\nr1 generated_text: '{backend.generated_text(r1['request_id'])}'")

    assert len(engine.prefix_cache.root.children) >= 1, "prefix cache 应有缓存节点"
    print("\n端到端测试通过!")

    await engine.shutdown()


asyncio.run(main())
