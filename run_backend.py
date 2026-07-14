from mini_runtime.backends.native_backend import NativeBackend
from mini_runtime.kv_cache import KVCacheManager, BlockTable

def main():
    backend = NativeBackend("Qwen/Qwen2.5-0.5B-Instruct")
    config = backend.model.config

    # 初始化 KV cache 管理器
    block_size = 16
    num_blocks = 128  # 16 × 128 = 2048 tokens，足够 prompt + 生成
    kv_manager = KVCacheManager(
        num_blocks=num_blocks,
        block_size=block_size,
        num_layers=config.num_layers,
        num_kv_heads=config.num_kv_heads,
        head_dim=config.head_dim,
        device=backend.device
    )
    backend.kv_manager = kv_manager

    prompt = "介绍一下杭州"

    # 先 tokenize 获取 prompt 长度，用于分配 block
    messages = [{"role": "user", "content": prompt}]
    chat_text = backend.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    input_ids = backend.tokenizer(chat_text, return_tensors="pt").input_ids.to(backend.device)
    num_prompt_tokens = input_ids.size(1)

    max_new_tokens = 100

    # 分配 block（prompt + 预期输出）
    block_table = BlockTable(block_size)
    if not kv_manager.allocate(block_table, num_prompt_tokens + max_new_tokens):
        print("block 分配失败！")
        return

    # Prefill
    block_ids = list(block_table.block_ids)
    token = backend.prefill(0, prompt, block_ids)

    # 逐 token 生成
    for _ in range(max_new_tokens):
        results = backend.batch_decode([(0, token, block_ids)])
        token = results[0]
        if token is None:
            break
        print(backend.tokenizer.decode([token]), end="", flush=True)

    backend.release(0)
    print()

if __name__ == "__main__":
    main()
