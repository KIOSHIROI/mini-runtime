from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class RadixNode:
    token_ids: list[int] = field(default_factory=list)   # 从父节点到当前节点的完整 token 序列
    block_ids: list[int] = field(default_factory=list)   # 对应的物理 block（仅完整 block）
    children: dict[int, "RadixNode"] = field(default_factory=dict)  # 第一个 token -> 子节点
    ref_count: int = 0   # 引用计数，用于 GC
    offset: int = 0      # 第一个 block 的起始位置（分裂产生的废位）
    last_access: int = 0 # LRU 逻辑时钟，最近被 match 命中的时间


class PrefixCache:
    """
    基于 Radix Tree 的 prefix cache。

    核心设计（block 级复用 + block 内偏移）:
      - KV cache 以 block 为最小复用单位，只复用完整 block
      - 节点的 block_ids 只含完整 block，末尾不足 block 的 token 不持有 block
      - 分裂点用 token 级的 common（保证两分支首 token 不同，不冲突）
      - common_node 对齐到 block 边界 (token_ids = edge[:aligned])
      - 原分支 (remaining_from_child) 从 common 开始，带 offset = common % block_size
        跳过边界 block 前面的废位
      - B 分支从 aligned 开始，重新 prefill token[aligned:]，无 offset
    """

    def __init__(self, block_size: int = 16):
        self.root = RadixNode()
        self.block_size = block_size
        self.root.ref_count = 1  # root 节点永不释放
        self._clock = 0  # LRU 逻辑时钟

    def _tick(self) -> int:
        self._clock += 1
        return self._clock

    def match(self, token_ids: list[int]) -> dict:
        """
        查找 token_ids 的最长前缀匹配。

        返回:
            matched_blocks:       可复用的完整 block_ids
            num_matched_tokens:  实际复用的 token 数（总是 block_size 整数倍，即 aligned）
            remaining_tokens:     未命中、需要 prefill 的 token
            matched_offset:       复用部分第一个 block 的 offset（请求 block_table 的 offset）
            node:                 最后匹配到的节点（insert 的起点）
            need_split:           是否需要分裂现有节点
            split_pos / child_to_split / aligned / floor_blocks / split_offset: 分裂信息
        """
        collected_blocks: list[int] = []
        num_matched_tokens = 0
        matched_offset = 0
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
                break

            if common == len(edge):
                # 完全匹配这条边，复用它的全部 block，继续向下
                child.last_access = self._tick()
                if not collected_blocks:
                    matched_offset = child.offset   # 第一个命中节点决定请求的 offset
                collected_blocks.extend(child.block_ids)
                num_matched_tokens += common
                pos += common
                node = child
            else:
                # 部分匹配：在 token 级 common 处分裂，但只复用对齐到 block 的完整 block
                child.last_access = self._tick()
                aligned = (common // self.block_size) * self.block_size
                floor_blocks = common // self.block_size
                split_offset = common - aligned    # = common % block_size，原分支的 offset

                matched_blocks = child.block_ids[:floor_blocks]
                if not collected_blocks and matched_blocks:
                    matched_offset = child.offset   # 第一个命中且复用了 block，继承 child 的 offset
                collected_blocks.extend(matched_blocks)

                total_matched = num_matched_tokens + aligned
                return {
                    'matched_blocks': collected_blocks,
                    'num_matched_tokens': total_matched,
                    'remaining_tokens': token_ids[total_matched:],
                    'matched_offset': matched_offset,
                    'node': node,
                    'need_split': True,
                    'split_pos': common,
                    'child_to_split': child,
                    'aligned': aligned,
                    'floor_blocks': floor_blocks,
                    'split_offset': split_offset,
                }

        # 没有遇到需要分裂的情况（完全匹配若干条边后结束，或一开始就不匹配）
        return {
            'matched_blocks': collected_blocks,
            'num_matched_tokens': num_matched_tokens,
            'remaining_tokens': token_ids[num_matched_tokens:],
            'matched_offset': matched_offset,
            'node': node,
            'need_split': False,
            'split_pos': 0,
            'child_to_split': None,
            'aligned': 0,
            'floor_blocks': 0,
            'split_offset': 0,
        }

    def insert(self, token_ids: list[int], block_ids: list[int], match_result: dict) -> list[int]:
        """
        根据 match 结果把请求的 block 插入树。

        Args:
            token_ids: 完整的 token 序列（仅用于文档，实际不直接用）
            block_ids: 为此请求分配的所有 block（matched 复用的 + 新 prefill 的）
            match_result: match() 返回的结果

        Returns:
            新增 cache 引用的 block 列表（调用者需对这些 block 调 inc_ref）。
            复用的 matched_blocks 不在此列（它们已在 cache 中，引用数不变）。
            分裂时 child 的 block 只是转移到 common_node/remaining，引用数不变，也不在此列。
        """
        node = match_result['node']
        remaining_tokens = match_result['remaining_tokens']
        matched_blocks = match_result['matched_blocks']

        if match_result['need_split']:
            child_to_split = match_result['child_to_split']
            edge = child_to_split.token_ids
            aligned = match_result['aligned']
            floor_blocks = match_result['floor_blocks']
            split_offset = match_result['split_offset']

            # 公共前缀节点：对齐 block，只持有完整 block，继承 child 的 offset
            common_node = RadixNode(
                token_ids=edge[:aligned],
                block_ids=child_to_split.block_ids[:floor_blocks],
                offset=child_to_split.offset,
                ref_count=1,
                last_access=child_to_split.last_access,
            )

            # 原节点的剩余部分：从 common 开始，含边界 block，offset 跳过废位
            remaining_from_child = RadixNode(
                token_ids=edge[aligned + split_offset:],   # = edge[common:]
                block_ids=child_to_split.block_ids[floor_blocks:],
                offset=split_offset,
                ref_count=child_to_split.ref_count,
                last_access=child_to_split.last_access,
            )
            remaining_from_child.children = child_to_split.children

            # B 的新分支：从 aligned 开始，重新 prefill，无 offset
            num_new_blocks = len(block_ids) - len(matched_blocks)
            new_branch = RadixNode(
                token_ids=remaining_tokens,
                block_ids=block_ids[-num_new_blocks:] if num_new_blocks > 0 else [],
                offset=0,
                ref_count=1,
                last_access=self._tick(),
            )

            # 重新挂接：两分支首 token 不同（原=edge[common], B=remaining_tokens[0]）
            common_node.children[edge[aligned + split_offset]] = remaining_from_child
            if remaining_tokens:
                common_node.children[remaining_tokens[0]] = new_branch
            node.children[edge[0]] = common_node
            return list(new_branch.block_ids)

        else:
            # 不分裂，直接挂新分支
            if not remaining_tokens:
                return []
            num_new_blocks = len(block_ids) - len(matched_blocks)
            new_node = RadixNode(
                token_ids=remaining_tokens,
                block_ids=block_ids[-num_new_blocks:] if num_new_blocks > 0 else [],
                offset=0,
                ref_count=1,
                last_access=self._tick(),
            )
            node.children[remaining_tokens[0]] = new_node
            return list(new_node.block_ids)

    def evict(self) -> list[int] | None:
        """
        LRU evict 一个叶子节点，返回其 block_ids（调用者需对这些 block 调 dec_ref）。
        无叶子可 evict 时返回 None。只 evict 叶子（无 children 的节点），保证不破坏树结构。
        """
        result = self._find_lru_leaf(self.root, None, 0)
        if result is None:
            return None
        parent, key, leaf = result
        del parent.children[key]
        return list(leaf.block_ids)

    def _find_lru_leaf(self, node: "RadixNode", parent, key: int):
        """在 node 子树找 last_access 最小的叶子。返回 (parent, key, leaf)；无叶子返回 None。"""
        if parent is not None and not node.children:
            return (parent, key, node)
        best = None
        for k, child in list(node.children.items()):
            candidate = self._find_lru_leaf(child, node, k)
            if candidate is not None:
                if best is None or candidate[2].last_access < best[2].last_access:
                    best = candidate
        return best


if __name__ == "__main__":
    cache = PrefixCache(block_size=16)

    print("=== 测试 1: 第一个请求，树为空 ===")
    tokens = list(range(48))   # 48 tokens, 3 个 block
    result = cache.match(tokens)
    print(f"match: matched_blocks={result['matched_blocks']}, num_matched_tokens={result['num_matched_tokens']}, offset={result['matched_offset']}")
    print(f"remaining tokens: {len(result['remaining_tokens'])}")
    cache.insert(tokens, [100, 101, 102], result)
    print(f"insert 后 root.children keys: {list(cache.root.children.keys())}")

    print("\n=== 测试 2: 完全相同的请求 ===")
    result2 = cache.match(tokens)
    print(f"match: matched_blocks={result2['matched_blocks']}, num_matched_tokens={result2['num_matched_tokens']}, remaining={len(result2['remaining_tokens'])}")

    print("\n=== 测试 3: 前缀匹配（前 32 tokens，对齐 block）===")
    tokens_partial = list(range(32))
    result3 = cache.match(tokens_partial)
    print(f"match: matched_blocks={result3['matched_blocks']}, num_matched_tokens={result3['num_matched_tokens']}, remaining={len(result3['remaining_tokens'])}")

    print("\n=== 测试 4: 部分匹配导致分裂（公共前缀 20，不对齐 block）===")
    tokens_diverge = list(range(20)) + [99] * 10   # 30 tokens
    result4 = cache.match(tokens_diverge)
    print(f"match: matched_blocks={result4['matched_blocks']}, num_matched_tokens={result4['num_matched_tokens']}")
    print(f"remaining tokens: {len(result4['remaining_tokens'])}, need_split={result4['need_split']}, split_offset={result4['split_offset']}")
    print(f"预期: matched_blocks=[100], num_matched_tokens=16, remaining=14, split_offset=4")

    # B 复用 [100]，重新 prefill 14 token 到 1 个新 block [200]
    cache.insert(tokens_diverge, [100, 200], result4)

    # 验证树结构
    common_node = cache.root.children[0]
    remaining = common_node.children[20]
    new_branch = common_node.children[16]
    print(f"\n树结构验证:")
    print(f"common_node: token_ids_len={len(common_node.token_ids)}, block_ids={common_node.block_ids}, offset={common_node.offset}")
    print(f"  预期: token_ids_len=16, block_ids=(100,101,102)[:1]=(100,), offset=0")
    print(f"remaining_from_child (key=20): token_ids_len={len(remaining.token_ids)}, block_ids={remaining.block_ids}, offset={remaining.offset}")
    print(f"  预期: token_ids_len=28, block_ids=(101,102), offset=4")
    print(f"new_branch (key=16): token_ids_len={len(new_branch.token_ids)}, block_ids={new_branch.block_ids}, offset={new_branch.offset}")
    print(f"  预期: token_ids_len=14, block_ids=(200,), offset=0")
    print(f"common_node children keys: {list(common_node.children.keys())} (预期 [20, 16])")

    print("\n=== 测试 5: 分裂后再次 match B（验证 B 能命中自己的分支）===")
    result5 = cache.match(tokens_diverge)
    print(f"match: matched_blocks={result5['matched_blocks']}, num_matched_tokens={result5['num_matched_tokens']}, remaining={len(result5['remaining_tokens'])}")
    print(f"预期: matched_blocks=[100,200], num_matched_tokens=30, remaining=0")

    print("\n=== 测试 6: 非 block_size 对齐的首次插入（18 tokens）===")
    cache2 = PrefixCache(block_size=16)
    tokens_unaligned = list(range(18))
    result6 = cache2.match(tokens_unaligned)
    print(f"首次 match: num_matched_tokens={result6['num_matched_tokens']}, remaining={len(result6['remaining_tokens'])}")
    cache2.insert(tokens_unaligned, [100, 101], result6)
    result6_2 = cache2.match(tokens_unaligned)
    print(f"再次 match: num_matched_tokens={result6_2['num_matched_tokens']}, remaining={len(result6_2['remaining_tokens'])}")
    print(f"预期: num_matched_tokens=18, remaining=0")
