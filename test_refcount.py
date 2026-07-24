"""测试阶段 1b: 统一 ref_count
验证 cache 持有的 block 不会被运行请求的 free 提前释放。
"""
import torch
from mini_runtime.kv_cache import KVCacheManager, BlockTable
from mini_runtime.prefix_cache import PrefixCache

BS = 16
manager = KVCacheManager(num_blocks=16, block_size=BS, num_layers=2,
                         num_kv_heads=2, head_dim=4, device=torch.device('cpu'))
cache = PrefixCache(block_size=BS)


def ref(bid):
    return manager.blocks[bid].ref_count


def simulate_admit(token_ids):
    """模拟 Engine 协调: match → inc_ref matched → allocate remaining → prefill → insert → inc_ref new"""
    result = cache.match(token_ids)
    bt = BlockTable(BS)
    for bid in result['matched_blocks']:
        manager.inc_ref(bid)          # 运行请求复用
        bt.append_block(bid)
    if result['remaining_tokens']:
        manager.allocate(bt, len(token_ids))
    # 模拟 prefill: 确保 block 有存储（真实场景由 backend 写入 KV）
    for bid in bt.block_ids:
        manager.pool._ensure_block(bid)
    new_cache_blocks = cache.insert(token_ids, list(bt.block_ids), result)
    for bid in new_cache_blocks:
        manager.inc_ref(bid)          # cache 持有新 block
    return bt, result


print("=== 测试 1: 全新 insert, ref_count = 运行(1) + cache(1) = 2 ===")
tokens_a = list(range(48))
bt_a, _ = simulate_admit(tokens_a)
blocks_a = list(bt_a.block_ids)
print(f"  A 的 block: {blocks_a}")
for b in blocks_a:
    print(f"  block {b}: ref_count={ref(b)} (期望 2)")
    assert ref(b) == 2
    assert manager.pool._blocks[b] is not None

print("\n=== 测试 2: A free(运行请求释放), ref_count=1, 数据保留 ===")
manager.free(bt_a)
for b in blocks_a:
    print(f"  block {b}: ref_count={ref(b)} (期望 1), is_free={manager.blocks[b].is_free} (期望 False)")
    assert ref(b) == 1
    assert not manager.blocks[b].is_free
    assert manager.pool._blocks[b] is not None
print("  PASS: cache 持有的 block 没被 free 提前释放")

print("\n=== 测试 3: B 完全 match 命中 A, ref_count=2 ===")
tokens_b = list(range(48))
bt_b, res_b = simulate_admit(tokens_b)
print(f"  B matched_blocks={res_b['matched_blocks']}, matched_tokens={res_b['matched_tokens']}")
assert res_b['matched_blocks'] == blocks_a
assert res_b['matched_tokens'] == 48
for b in blocks_a:
    print(f"  block {b}: ref_count={ref(b)} (期望 2)")
    assert ref(b) == 2

print("\n=== 测试 4: B free, ref_count=1 ===")
manager.free(bt_b)
for b in blocks_a:
    assert ref(b) == 1
    assert not manager.blocks[b].is_free
print("  PASS: 仍 cache 持有")

print("\n=== 测试 5: C 部分匹配分裂 (common=20, aligned=16) ===")
tokens_c = list(range(20)) + [99] * 10   # 30 token
bt_c, res_c = simulate_admit(tokens_c)
blocks_c = list(bt_c.block_ids)
print(f"  C matched_blocks={res_c['matched_blocks']}, matched_tokens={res_c['matched_tokens']}")
print(f"  C block_table: {blocks_c}")
assert res_c['matched_blocks'] == [blocks_a[0]]      # [0]
assert res_c['matched_tokens'] == 16
# block 0: cache(common_node,1) + C运行(1) = 2
print(f"  block 0 (复用, common_node): ref_count={ref(0)} (期望 2)")
assert ref(0) == 2
# block 1,2: cache(remaining_from_child,1) = 1, 从 child 转移不变
print(f"  block 1 (remaining): ref_count={ref(1)} (期望 1)")
print(f"  block 2 (remaining): ref_count={ref(2)} (期望 1)")
assert ref(1) == 1
assert ref(2) == 1
# block 3: C运行(1) + cache(new_branch,1) = 2
new_block = blocks_c[-1]
print(f"  block {new_block} (新, new_branch): ref_count={ref(new_block)} (期望 2)")
assert ref(new_block) == 2

print("\n=== 测试 6: C free, block 0→1, new_block→1, block 1,2 仍 1 ===")
manager.free(bt_c)
print(f"  block 0: {ref(0)} (期望 1)")
print(f"  block 1: {ref(1)} (期望 1)")
print(f"  block 2: {ref(2)} (期望 1)")
print(f"  block {new_block}: {ref(new_block)} (期望 1)")
assert ref(0) == 1 and ref(1) == 1 and ref(2) == 1 and ref(new_block) == 1
for b in [0, 1, 2, new_block]:
    assert manager.pool._blocks[b] is not None
print("  PASS: 所有 cache 持有的 block 数据保留")

print("\n=== 测试 7: 分裂后 D match 命中 C 的分支 (跨 common_node + new_branch) ===")
tokens_d = list(range(20)) + [99] * 10
bt_d, res_d = simulate_admit(tokens_d)
print(f"  D matched_blocks={res_d['matched_blocks']}, matched_tokens={res_d['matched_tokens']}")
assert res_d['matched_blocks'] == [0, new_block]
assert res_d['matched_tokens'] == 30
print(f"  block 0: {ref(0)} (期望 2), block {new_block}: {ref(new_block)} (期望 2)")
assert ref(0) == 2 and ref(new_block) == 2
manager.free(bt_d)
assert ref(0) == 1 and ref(new_block) == 1
print("  PASS: 跨节点 match 命中 + ref_count 正确")

print(f"\n{'='*45}\n阶段 1b 测试全部通过! ref_count 统一正确")
