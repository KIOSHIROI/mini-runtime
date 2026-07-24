"""测试 BlockPool.read_layer 的 offset 支持"""
import torch
from mini_runtime.kv_cache import BlockPool, BlockTable

def make_pool(num_blocks=4, num_layers=1, num_kv_heads=2, block_size=16, head_dim=4):
    return BlockPool(num_blocks, num_layers, num_kv_heads, block_size, head_dim, torch.device('cpu'))

def fill_block(pool, block_id, start_token, num_kv_heads=2, head_dim=4):
    """往 block_id 写入 16 个位置的 KV，让 K[0,0,pos,0] = start_token + pos 作为特征"""
    pool._ensure_block(block_id)
    for pos in range(16):
        for layer in range(pool.num_layers):
            k, v = pool._blocks[block_id][layer]
            k[0, 0, pos, 0] = start_token + pos
            v[0, 0, pos, 0] = start_token + pos

def check(name, got, expect):
    ok = got == expect
    print(f"  {'PASS' if ok else 'FAIL'}: {name} got={got} expect={expect}")
    return ok

print("=== 测试 1: 单 block, offset=0 (向后兼容) ===")
pool = make_pool()
fill_block(pool, 0, start_token=16)   # block 0 = token 16-31
K, V = pool.read_layer(0, [[0]], [16], 16, [0])
all_ok = True
for i in range(16):
    all_ok &= check(f"pos{i}", K[0,0,i,0].item(), 16 + i)

print("\n=== 测试 2: 单 block, offset=4 → 读 token 20-31 ===")
K, V = pool.read_layer(0, [[0]], [12], 12, [4])
for i in range(12):
    all_ok &= check(f"pos{i}", K[0,0,i,0].item(), 20 + i)
# 确认返回长度是 12
all_ok &= check("返回长度", K.shape[2], 12)

print("\n=== 测试 3: 多 block + offset (block0 offset=4 + block1 全部) ===")
pool2 = make_pool()
fill_block(pool2, 0, start_token=16)   # block 0 = token 16-31
fill_block(pool2, 1, start_token=32)   # block 1 = token 32-47
# 原分支从 token 20 开始: block 0 位置 4-15 (token 20-31) + block 1 位置 0-15 (token 32-47) = 28 token
K, V = pool2.read_layer(0, [[0, 1]], [28], 28, [4])
for i in range(12):   # 前 12 个来自 block 0 offset
    all_ok &= check(f"pos{i}(block0)", K[0,0,i,0].item(), 20 + i)
for i in range(16):   # 后 16 个来自 block 1
    all_ok &= check(f"pos{12+i}(block1)", K[0,0,12+i,0].item(), 32 + i)

print("\n=== 测试 4: offset_list=None 向后兼容 ===")
K, V = pool2.read_layer(0, [[0, 1]], [32], 32)   # 不传 offset_list
for i in range(16):
    all_ok &= check(f"pos{i}", K[0,0,i,0].item(), 16 + i)
for i in range(16):
    all_ok &= check(f"pos{16+i}", K[0,0,16+i,0].item(), 32 + i)

print("\n=== 测试 5: batch 多请求 (一个有 offset, 一个没有) ===")
K, V = pool2.read_layer(0, [[0, 1], [0]], [28, 16], 28, [4, 0])
# 请求0: offset=4, past_len=28 → token 20-47
all_ok &= check("req0 pos0", K[0,0,0,0].item(), 20)
all_ok &= check("req0 pos12", K[0,0,12,0].item(), 32)
# 请求1: offset=0, past_len=16 → token 16-31, pad 到 28
all_ok &= check("req1 pos0", K[1,0,0,0].item(), 16)
all_ok &= check("req1 pos15", K[1,0,15,0].item(), 31)
all_ok &= check("req1 pad pos16", K[1,0,16,0].item(), 0)   # padding 应为 0

print("\n=== 测试 6: BlockTable capacity 含 offset ===")
bt = BlockTable(16)
bt.append_block(0)
bt.append_block(1)
print(f"  offset=0 时 capacity = {bt.capacity} (期望 32)")
all_ok &= check("capacity offset=0", bt.capacity, 32)
bt.set_offset(4)
print(f"  offset=4 时 capacity = {bt.capacity} (期望 28)")
all_ok &= check("capacity offset=4", bt.capacity, 28)

print(f"\n{'='*40}\n结果: {'全部通过' if all_ok else '有失败'}")
