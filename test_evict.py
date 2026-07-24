"""测试阶段 1c: evict 机制 (LRU 叶子驱逐 + dec_ref)"""
import torch
from mini_runtime.kv_cache import KVCacheManager, BlockTable
from mini_runtime.prefix_cache import PrefixCache

BS = 16
manager = KVCacheManager(16, BS, 2, 2, 4, torch.device('cpu'))
cache = PrefixCache(BS)


def admit_and_release(token_ids):
    """模拟请求: admit + prefill + insert + free(只剩 cache 引用)"""
    result = cache.match(token_ids)
    bt = BlockTable(BS)
    for bid in result['matched_blocks']:
        manager.inc_ref(bid); bt.append_block(bid)
    if result['remaining_tokens']:
        manager.allocate(bt, len(token_ids))
    for bid in bt.block_ids:
        manager.pool._ensure_block(bid)
    new_blocks = cache.insert(token_ids, list(bt.block_ids), result)
    for bid in new_blocks:
        manager.inc_ref(bid)
    blocks = list(bt.block_ids)   # free 会 clear block_table, 先记录
    manager.free(bt)              # 请求结束，只剩 cache 引用
    return blocks


print("=== 测试 1: 插入 A 和 B (不同前缀), 各自 cache 引用 = 1 ===")
a_blocks = admit_and_release(list(range(48)))         # block 0,1,2
b_blocks = admit_and_release(list(range(100, 148)))   # block 3,4,5
print(f"  A blocks={a_blocks} ref={[manager.blocks[b].ref_count for b in a_blocks]} (期望 [1,1,1])")
print(f"  B blocks={b_blocks} ref={[manager.blocks[b].ref_count for b in b_blocks]} (期望 [1,1,1])")
assert all(manager.blocks[b].ref_count == 1 for b in a_blocks + b_blocks)

print("\n=== 测试 2: match A 刷新其 last_access, B 仍是旧的 ===")
cache.match(list(range(48)))   # A 命中, last_access 更新

print("\n=== 测试 3: evict 应驱逐 B (LRU 最旧叶子) ===")
evicted = cache.evict()
print(f"  evicted={evicted}")
assert set(evicted) == set(b_blocks), f"应驱逐 B 的 block, got {evicted}"

print("\n=== 测试 4: 对 evicted block dec_ref, ref_count→0, is_free=True ===")
for bid in evicted:
    manager.dec_ref(bid)
print(f"  B ref={[manager.blocks[b].ref_count for b in b_blocks]} (期望 [0,0,0])")
print(f"  B is_free={[manager.blocks[b].is_free for b in b_blocks]} (期望 [True,True,True])")
assert all(manager.blocks[b].ref_count == 0 for b in b_blocks)
assert all(manager.blocks[b].is_free for b in b_blocks)

print("\n=== 测试 5: A 不受影响, 仍 cache 持有 ===")
print(f"  A ref={[manager.blocks[b].ref_count for b in a_blocks]} (期望 [1,1,1])")
assert all(manager.blocks[b].ref_count == 1 for b in a_blocks)
assert all(not manager.blocks[b].is_free for b in a_blocks)

print("\n=== 测试 6: B 被驱逐后, 再 match B 命中 0 (树中已删) ===")
res = cache.match(list(range(100, 148)))
print(f"  B match: matched_blocks={res['matched_blocks']} (期望 [])")
assert res['matched_blocks'] == []

print("\n=== 测试 7: A 仍可 match 命中 ===")
res = cache.match(list(range(48)))
print(f"  A match: matched_blocks={res['matched_blocks']} matched_tokens={res['matched_tokens']}")
assert res['matched_blocks'] == a_blocks
assert res['matched_tokens'] == 48

print(f"\n{'='*40}\n阶段 1c evict 测试全部通过!")
