from dataclasses import dataclass 
import math 
from collections import deque
@dataclass 
class KVBlock:
    block_id: int 
    ref_count: int = 0
    is_free: bool = True 

class BlockTable:
    def __init__(self, block_size):
        self.block_size = block_size 
        self._block_ids = []
    
    def append_block(self, physical_block_id: int) -> None:
        self._block_ids.append(physical_block_id)
    
    @property
    def num_blocks(self) -> int:
        return len(self._block_ids)

    @property
    def capacity(self) -> int:
        return self.num_blocks * self.block_size
    
    @property
    def block_ids(self) -> tuple[int, ...]:
        return tuple(self._block_ids)
    
    def clear(self) -> None:
        self._block_ids.clear()
    
class KVCacheManager:
    def __init__(self, num_blocks: int, block_size: int):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.max_allocated = 0
        self.blocks = [KVBlock(i) for i in range(num_blocks)]
        self.free_blocks_ids = deque(range(num_blocks))
        self.used_blocks_ids = []

    def allocate(self, block_table: BlockTable, num_tokens: int) -> bool:
        num_need = math.ceil(num_tokens / self.block_size)
        num_need -= block_table.num_blocks
        if len(self.free_blocks_ids) >= num_need:
            for i in range(num_need):
                block_id = self.free_blocks_ids.popleft()
                self.blocks[block_id].ref_count += 1
                self.blocks[block_id].is_free = False 
                self.used_blocks_ids.append(block_id)
                block_table.append_block(block_id)
            self.max_allocated = max(self.max_allocated, len(self.used_blocks_ids))
            return True 
        return False
    
    def free(self, block_table: BlockTable) -> None:
        for block_id in block_table.block_ids:
            self.blocks[block_id].ref_count -= 1
            if self.blocks[block_id].ref_count == 0:
                self.used_blocks_ids.remove(block_id)
                self.free_blocks_ids.append(block_id)
                self.blocks[block_id].is_free = True 
        
        block_table.clear()
        
    
    @property 
    def utilization(self) -> float:
        return len(self.used_blocks_ids) / self.num_blocks
    
    @property
    def allocated_blocks(self) -> int:
        return len(self.used_blocks_ids)
    
    def snapshot(self) -> dict:
        return {
            'allocated_blocks': self.allocated_blocks,
            'utilization': self.utilization,
            'max_allocated': self.max_allocated,
            'total_blocks': self.num_blocks,
            'free_blocks': list(self.free_blocks_ids),
            'used_blocks': list(self.used_blocks_ids),
        }

            
        
        
    