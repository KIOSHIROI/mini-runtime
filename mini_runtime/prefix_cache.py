from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import math

@dataclass
class RadixNode:
    token_ids: list[int] = field(default_factory=list)  # 从父节点到当前节点的完整 token 序列
    block_ids: list[int] = field(default_factory=list)  # 对应的物理 block（容量 >= len(token_ids)）
    children: dict[int, "RadixNode"] = field(default_factory=dict)  # 第一个 token -> 子节点
    ref_count: int = 0  # 引用计数，用于 GC

    @property
    def capacity(self) -> int:
        """block_ids 能容纳的 token 数量"""
        return len(self.block_ids) * BLOCK_SIZE

BLOCK_SIZE = 16  # 全局 block size

class PrefixCache:
    def __init__(self, block_size: int = 16):
        self.root = RadixNode()
        self.block_size = block_size
        self.root.ref_count = 1  # root 节点永不释放

    def match(self, token_ids: list[int]) -> dict:
        """
        查找 token_ids 的最长前缀匹配。

        返回:
            {
                'matched_blocks': list[int],  # 可复用的 block_ids
                'matched_tokens': int,        # 实际匹配的 token 数量
                'remaining_tokens': list[int], # 未匹配的 token 序列
                'node': RadixNode,             # 最后匹配的节点
                'need_split': bool,            # 是否需要分裂
                'split_pos': int,              # 分裂位置（仅当 need_split=True）
                'child_to_split': RadixNode,   # 需要分裂的子节点（仅当 need_split=True）
            }
        """
        collected_blocks = []
        matched_tokens = 0
        node = self.root
        pos = 0

        while pos < len(token_ids):
            first_token = token_ids[pos]
            if first_token not in node.children:
                break

            child = node.children[first_token]
            edge = child.token_ids

            # 计算 token_ids[pos:] 和 edge 的公共前缀长度
            common = 0
            max_common = min(len(edge), len(token_ids) - pos)
            while common < max_common and token_ids[pos + common] == edge[common]:
                common += 1

            if common == 0:
                # 第一个 token 就不匹配，停止
                break

            if common == len(edge):
                # 完全匹配这条边，继续向下
                # 只有完全匹配整条边，才能复用对应的 block_ids
                collected_blocks.extend(child.block_ids)
                matched_tokens += common
                pos += common
                node = child
            else:
                # 部分匹配，需要在 common 处分裂
                # 只收集完全匹配的 block（按 block_size 对齐）
                matched_block_count = common // self.block_size
                matched_blocks = child.block_ids[:matched_block_count]
                collected_blocks.extend(matched_blocks)

                actual_matched_tokens = matched_block_count * self.block_size

                return {
                    'matched_blocks': collected_blocks,
                    'matched_tokens': actual_matched_tokens,
                    'remaining_tokens': token_ids[actual_matched_tokens:],
                    'node': node,
                    'need_split': True,
                    'split_pos': common,  # token 级的分裂位置
                    'child_to_split': child,
                }

        # 完全退出循环，没有遇到需要分裂的情况
        return {
            'matched_blocks': collected_blocks,
            'matched_tokens': matched_tokens,
            'remaining_tokens': token_ids[matched_tokens:],
            'node': node,
            'need_split': False,
            'split_pos': 0,
            'child_to_split': None,
        }

    def insert(
        self,
        token_ids: list[int],
        block_ids: list[int],
        match_result: dict,
    ):
        """
        根据 match 结果插入新的请求。

        Args:
            token_ids: 完整的 token 序列
            block_ids: 为此请求分配的所有 block
            match_result: match() 返回的结果
        """
        node = match_result['node']
        remaining_tokens = match_result['remaining_tokens']

        if match_result['need_split']:
            # 需要分裂现有节点
            child_to_split = match_result['child_to_split']
            split_pos = match_result['split_pos']

            edge = child_to_split.token_ids

            # 公共前缀部分（会被两个分支共享）
            common_node = RadixNode(
                token_ids=edge[:split_pos],
                block_ids=child_to_split.block_ids[:split_pos // self.block_size],
                ref_count=1,
            )

            # 原节点的剩余部分
            remaining_from_child = RadixNode(
                token_ids=edge[split_pos:],
                block_ids=child_to_split.block_ids[split_pos // self.block_size:],
                ref_count=child_to_split.ref_count,
            )
            remaining_from_child.children = child_to_split.children

            # 新分支
            num_new_blocks = len(block_ids) - len(match_result['matched_blocks'])
            new_branch = RadixNode(
                token_ids=remaining_tokens,
                block_ids=block_ids[-num_new_blocks:] if num_new_blocks > 0 else [],
                ref_count=1,
            )

            # 重新挂接
            common_node.children[edge[split_pos]] = remaining_from_child
            if remaining_tokens:
                common_node.children[remaining_tokens[0]] = new_branch

            node.children[edge[0]] = common_node

        else:
            # 不需要分裂，直接添加新分支
            if not remaining_tokens:
                return

            num_needed_blocks = len(block_ids) - len(match_result['matched_blocks'])
            new_node = RadixNode(
                token_ids=remaining_tokens,
                block_ids=block_ids[-num_needed_blocks:] if num_needed_blocks > 0 else [],
                ref_count=1,
            )
            node.children[remaining_tokens[0]] = new_node

    def decrement_ref(self, token_ids: list[int]) -> list[int]:
        """
        减少 token_ids 对应路径的引用计数，返回可以释放的 block_ids。

        这是一个简化版本，实际实现需要考虑引用计数和 GC。
        """
        # TODO: 实现引用计数和 GC
        return []


if __name__ == "__main__":
    cache = PrefixCache(block_size=16)

    print("=== 测试 1: 第一个请求，树为空 ===")
    tokens = list(range(48))  # 48 tokens, 需要 3 个 block
    result = cache.match(tokens)
    print(f"match: matched_blocks={result['matched_blocks']}, matched_tokens={result['matched_tokens']}")
    print(f"remaining tokens: {len(result['remaining_tokens'])}")

    # 分配 3 个 block
    cache.insert(tokens, [100, 101, 102], result)
    print(f"insert 后 root.children keys: {list(cache.root.children.keys())}")

    print("\n=== 测试 2: 完全相同的请求 ===")
    result2 = cache.match(tokens)
    print(f"match: matched_blocks={result2['matched_blocks']}, matched_tokens={result2['matched_tokens']}")
    print(f"remaining tokens: {len(result2['remaining_tokens'])}")

    print("\n=== 测试 3: 前缀匹配（前 32 tokens） ===")
    tokens_partial = list(range(32))
    result3 = cache.match(tokens_partial)
    print(f"match: matched_blocks={result3['matched_blocks']}, matched_tokens={result3['matched_tokens']}")
    print(f"remaining tokens: {len(result3['remaining_tokens'])}")

    print("\n=== 测试 4: 部分匹配导致分裂 ===")
    tokens_diverge = list(range(20)) + [99] * 10
    result4 = cache.match(tokens_diverge)
    print(f"match: matched_blocks={result4['matched_blocks']}, matched_tokens={result4['matched_tokens']}")
    print(f"remaining tokens: {len(result4['remaining_tokens'])}, need_split={result4['need_split']}")

    # 插入新分支
    cache.insert(tokens_diverge, [100, 101, 200], result4)

    # 验证树结构
    child0 = cache.root.children[0]
    print(f"\n树结构验证:")
    print(f"root.children[0] token_ids length: {len(child0.token_ids)}")
    print(f"root.children[0] children keys: {list(child0.children.keys())}")

    print("\n=== 测试 5: 非 block_size 对齐的情况 ===")
    cache2 = PrefixCache(block_size=16)
    tokens_unaligned = list(range(18))  # 18 tokens，需要 2 个 block（但第二个 block 只有 2 个 token）
    result5 = cache2.match(tokens_unaligned)
    print(f"match: matched_tokens={result5['matched_tokens']}, remaining={len(result5['remaining_tokens'])}")

    cache2.insert(tokens_unaligned, [100, 101], result5)
    result5_2 = cache2.match(tokens_unaligned)
    print(f"再次 match: matched_tokens={result5_2['matched_tokens']}, remaining={len(result5_2['remaining_tokens'])}")
    print(f"预期: matched_tokens=18, remaining=0")