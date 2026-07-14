from __future__ import annotations
from dataclasses import dataclass, field

@dataclass
class RadixNode:
    token_ids: list[int] = field(default_factory=list) # 从父节点到当前节点的 token 序列
    block_ids: list[int] = field(default_factory=list) # 此序列对应的物理 block
    children: dict[int, "RadixNode"] = field(default_factory=dict) # 第一个 token -> 子节点

class PrefixCache:
    def __init__(self, block_size: int):
        self.root = RadixNode()
        self.block_size = block_size
    
    def match(self, token_ids: list[int]) -> tuple[list[int], int]:
        collected_blocks = []
        pos = 0
        node = self.root 
        
        while pos < len(token_ids):
            first_token = token_ids[pos]
            if first_token not in node.children:
                break 
        
            child = node.children[first_token]
            edge = child.token_ids
            
            # 比较 token_ids[pos:] 和 edge。算公共前缀长度 common
            common = 0
            while common < len(edge) and pos + common < len(token_ids) and token_ids[pos + common] == edge[common]:
                common += 1

            
            if common == len(edge):
                # 完全匹配，继续向下
                collected_blocks.extend(child.block_ids)
                pos += common
                node = child
            else:
                # 部分匹配，停止
                matched_blocks = child.block_ids[:common // self.block_size]  # 只收集匹配的部分
                collected_blocks.extend(matched_blocks)                
                break
            
        match_pos = len(collected_blocks) * self.block_size
        return (collected_blocks, match_pos)
    
if __name__ == "__main__":
    cache = PrefixCache(block_size=16)
    child = RadixNode(token_ids=list(range(32)),
                      block_ids=[100, 101],
                      )
    cache.root.children[0] = child 
    
    blocks, pos = cache.match([999, 888])
    print(f"未命中: blocks={blocks}, pos={pos}")
   # 预期: blocks=[], pos=0