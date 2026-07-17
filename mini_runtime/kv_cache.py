from dataclasses import dataclass 
import math 
import torch
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
    
class KVCacheManager: # 管理层，监控谁占用了哪些 block，负责分配和回收 block
    def __init__(self, num_blocks: int, block_size: int, num_layers: int, num_kv_heads: int, head_dim: int, device: torch.device):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.max_allocated = 0
        self.blocks = [KVBlock(i) for i in range(num_blocks)]
        self.free_blocks_ids = deque(range(num_blocks))
        self.used_blocks_ids = []
        self.pool = BlockPool(num_blocks, num_layers, num_kv_heads, block_size, head_dim, device)

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
                self.pool.release(block_id)

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

class BlockPool: # 存储层，负责实际存储 KV 数据，提供读写接口
    """_summary_
    需要提供的能力：
    - prefill 完成后把 KV 按 block 切分写入
    - decode 前 从 _kv 中读取 KV 并 pad
    - decode 后 写入单个新 token 的 KV 到对应 block 位置
    - release 时释放至 None，归还 block
    """
    def __init__(self, num_blocks, num_layers, num_kv_heads, block_size, head_dim, device):
        self._blocks = [None] * num_blocks
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.block_size = block_size
        self.head_dim = head_dim
        self.device = device
    def release(self, block_id):
        if self._blocks[block_id] is None:
            return

        # 显式释放 GPU 内存
        if self.device.type == "cuda":
            for layer_kv in self._blocks[block_id]:
                for tensor in layer_kv:
                    if tensor is not None:
                        del tensor
            torch.cuda.empty_cache()

        self._blocks[block_id] = None
        
    def _ensure_block(self,block_id):
        if self._blocks[block_id] is None:
            self._blocks[block_id] = [(
                torch.zeros((1, self.num_kv_heads, self.block_size, self.head_dim), device=self.device),
                torch.zeros((1, self.num_kv_heads, self.block_size, self.head_dim), device=self.device),
            ) for _ in range(self.num_layers)]  
            
    def write_block(self, block_id, kv_layers):
        """
        懒初始化 tensor storage. 每个 block 的生命周期中仅调用一次 
        """
        self._ensure_block(block_id)
        for layer_idx ,(K_chunk, V_chunk)in enumerate(kv_layers):
            chunk_len = K_chunk.shape[2]
            bk, bv = self._blocks[block_id][layer_idx]
            bk[:, :, :chunk_len, :] = K_chunk
            bv[:, :, :chunk_len, :] = V_chunk

    def write_blocks(self, block_ids, kv_layers): # prefill 
        token_len = kv_layers[0][0].shape[2]
        for i, block_id in enumerate(block_ids):
            start = i * self.block_size 
            end = min(start + self.block_size, token_len)
            chunk = [(k[:, :, start:end, :], v[:, :, start:end, :]) 
                     for k, v in kv_layers]
            self.write_block(block_id, chunk)
            
    
    def write_token(self, block_id, kv_layers, pos_in_block): # decode
        self._ensure_block(block_id)  # decode 可能写入新分配的 block
        for layer_idx, (K_chunk, V_chunk) in enumerate(kv_layers):
            bk, bv = self._blocks[block_id][layer_idx]
            bk[:, :, pos_in_block, :] = K_chunk[:, :, 0, :]
            bv[:, :, pos_in_block, :] = V_chunk[:, :, 0, :]
    
    def read_layer(self, layer_idx, block_ids_list, past_len_list, max_past_len):
        B = len(block_ids_list)
        K_list = []
        V_list = []
        for block_ids, past_len in zip(block_ids_list, past_len_list):
            # 跳过未初始化的 block（刚分配、还没写入数据的）
            K_parts = [self._blocks[bid][layer_idx][0]
                       for bid in block_ids if self._blocks[bid] is not None]
            V_parts = [self._blocks[bid][layer_idx][1]
                       for bid in block_ids if self._blocks[bid] is not None]
            # 空列表防护：batch_size 可能为 0 或首次 prefill 前
            K_req = torch.cat(K_parts, dim=2) if K_parts else \
                torch.zeros(1, self.num_kv_heads, 0, self.head_dim, device=self.device)
            V_req = torch.cat(V_parts, dim=2) if V_parts else \
                torch.zeros(1, self.num_kv_heads, 0, self.head_dim, device=self.device)
            
            K_req = K_req[:, :, :past_len, :]
            V_req = V_req[:, :, :past_len, :]
            
            pad_len = max_past_len - past_len
            if pad_len > 0:
                K_req = torch.cat([K_req, torch.zeros((1, self.num_kv_heads, pad_len, self.head_dim), device=self.device)], dim=2)
                V_req = torch.cat([V_req, torch.zeros((1, self.num_kv_heads, pad_len, self.head_dim), device=self.device)], dim=2)
                
            K_list.append(K_req)
            V_list.append(V_req)
        
        K_batched = torch.cat(K_list, dim=0)
        V_batched = torch.cat(V_list, dim=0)
        return K_batched, V_batched

                
            
            

        
        