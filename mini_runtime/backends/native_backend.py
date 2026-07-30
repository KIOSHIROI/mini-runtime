import os
import time
import torch
from transformers import AutoTokenizer
from dataclasses import dataclass

from ..model.qwen2_model import Qwen2Model
from ..model.config import Qwen2Config
from ..model.loader import load_qwen2_weights
from ..kv_cache import KVCacheManager
from ..profiler import ModuleProfiler


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@dataclass
class PrefillInput:
    """Backend 定义的 prefill 输入。Engine 负责从 Request 构造此对象。"""
    request_id: int
    token_ids: list[int]           # 完整的 prompt token 序列
    block_ids: tuple[int, ...]     # 此请求持有的所有 block
    block_offset: int = 0          # 第一个 block 的起始偏移
    skip_tokens: int = 0           # 跳过前 N 个 token（已在 cache 中）
    num_cached_blocks: int = 0     # 前 N 个 block 来自 cache
    chunk_start: int = 0          # 此 chunk 在完整 prompt 中的起始位置
    chunk_end: int | None = None     # 此 chunk 在完整 prompt 中的结束位置（不含），None 表示完整 prefill
    is_last_chunk: bool = True      # 是否为最后一个 chunk（决定是否返回 first token）

@dataclass
class BatchDecodeInput:
    """Backend 定义的 decode 输入。"""
    request_id: int
    token_id: int                  # 上一个生成的 token
    block_ids: tuple[int, ...]     # 此请求持有的所有 block
    block_offset: int = 0          # 第一个 block 的起始偏移

class NativeBackend:
    def __init__(self, model_path: str = "Qwen/Qwen2.5-0.5B-Instruct", device: torch.device = DEVICE):
        model_path = os.path.expanduser(model_path)
        self.kv_manager = None
        # 如果本地路径但没有 tokenizer 文件 → 在 snapshots/ 下找
        if os.path.isdir(model_path) and not os.path.isfile(
            os.path.join(model_path, "tokenizer.json")
        ):
            snapshot_dir = os.path.join(model_path, "snapshots")
            if os.path.isdir(snapshot_dir):
                subdirs = sorted(os.listdir(snapshot_dir))
                if subdirs:
                    model_path = os.path.join(snapshot_dir, subdirs[0])

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = Qwen2Model(Qwen2Config())
        self.device = device
        load_qwen2_weights(self.model, model_path, device)
        self.model.eval()
        self.model.to(device)

        # 纯推理环境，全局禁用 autograd 避免计算图积累占用显存
        torch.set_grad_enabled(False)
        self._past_len: dict[int, int] = {}
        self._generated: dict[int, list[int]] = {}  # request_id → 已生成token id
        
        self.module_profiler = ModuleProfiler()
        self.model.module_profiler = self.module_profiler

    def prefill(self, inp: PrefillInput) -> int | None:
        """prefix-aware prefill。 chunk_end=None 时做完整 prefill，否则做 chunked prefill"""
        if inp.chunk_end is not None:
            return self._prefill_chunk(inp)
        else:
            return self._prefill_full(inp)
        
    def _prefill_full(self, inp: PrefillInput) -> int:
        pool = self.kv_manager.pool
        token_ids = inp.token_ids
        matched_tokens = inp.skip_tokens
        num_matched_blocks = inp.num_cached_blocks
        matched_offset = inp.block_offset
        block_ids = list(inp.block_ids)
        remaining = token_ids[matched_tokens:]

        # 全部命中缓存：无需 prefill，直接 decode 一步获取首 token
        if not remaining:
            matched_blocks = block_ids[:num_matched_blocks]
            matched_kv = []
            for layer_idx in range(pool.num_layers):
                K, V = pool.read_layer(layer_idx, [matched_blocks], [matched_tokens],
                                        matched_tokens, [matched_offset])
                matched_kv.append((K, V))
            last_token = token_ids[-1]
            input_ids = torch.tensor([[last_token]], device=self.device)
            position_ids = torch.tensor([[matched_tokens]], device=self.device)
            logits, _ = self.model(input_ids, position_ids, past_key_values=matched_kv)
            self._past_len[inp.request_id] = matched_tokens
            next_token = logits[0, -1, :].argmax().item()
            self._generated[inp.request_id] = [next_token]
            return next_token

        input_ids = torch.tensor([remaining], device=self.device)
        position_ids = torch.arange(matched_tokens, len(token_ids),
                                    device=self.device).unsqueeze(0)

        # 1. 读取 matched 的 KV 作为 past_key_values (复用 prefix cache)
        if matched_tokens > 0:
            matched_blocks = block_ids[:num_matched_blocks]
            matched_kv = []
            for layer_idx in range(pool.num_layers):
                K, V = pool.read_layer(layer_idx, [matched_blocks], [matched_tokens],
                                        matched_tokens, [matched_offset])
                matched_kv.append((K, V))
            # attention_mask: remaining attend to matched(全部) + remaining(causal)
            q_len = len(remaining)
            kv_len = matched_tokens + q_len
            attn_mask = torch.ones(1, 1, q_len, kv_len, device=self.device, dtype=torch.bool)
            causal = torch.tril(torch.ones(q_len, q_len, device=self.device, dtype=torch.bool))
            attn_mask[:, :, :, matched_tokens:] = causal
        else:
            matched_kv = None
            attn_mask = None

        # 2. forward remaining (带 matched KV)
        logits, past_key_values = self.model(
            input_ids, position_ids, past_key_values=matched_kv, attention_mask=attn_mask)

        # 3. 只把 remaining 的 KV 写入新 block (matched 的已在 cache)
        new_block_ids = block_ids[num_matched_blocks:]
        remaining_kv = [(k[:, :, matched_tokens:, :], v[:, :, matched_tokens:, :])
                        for k, v in past_key_values]
        if new_block_ids:
            pool.write_blocks(new_block_ids, remaining_kv)

        self._past_len[inp.request_id] = len(token_ids)
        next_token = logits[0, -1, :].argmax().item()
        self._generated[inp.request_id] = [next_token]
        return next_token
    
    def _prefill_chunk(self, inp: PrefillInput) -> int | None:
        mp = self.module_profiler
        
        pool = self.kv_manager.pool 
        token_ids = inp.token_ids 
        chunk_start = inp.chunk_start
        chunk_end = inp.chunk_end 
        block_ids = list(inp.block_ids)
        block_offset = inp.block_offset
        
        past_len = chunk_start 
        chunk_tokens = token_ids[chunk_start:chunk_end]
        chunk_len = len(chunk_tokens)
                
        # 1. 全缓存命中
        if chunk_len == 0:
            past_kv = []
            # 读 past KV
            t0 = time.perf_counter()
            for layer_idx in range(pool.num_layers):
                K, V = pool.read_layer(layer_idx, [block_ids], [past_len],
                                        past_len, [block_offset])
                past_kv.append((K, V)) 
            mp.record("kv_head", mp.elapsed(t0))
            
            input_ids = torch.tensor([[token_ids[-1]]], device=self.device)
            position_ids = torch.tensor([[past_len]], device=self.device)
            
            # forward
            logits, _ = self.model(input_ids, position_ids, past_key_values=past_kv)
            
            self._past_len[inp.request_id] = past_len
            next_token = logits[0, -1, :].argmax().item()
            self._generated[inp.request_id] = [next_token]
            # 全命中无KV写入，直接返回 next_token
            return next_token
        
        input_ids = torch.tensor([chunk_tokens], device=self.device)
        position_ids = torch.arange(past_len, chunk_end, device=self.device).unsqueeze(0)
        
        if past_len > 0:
            past_kv = []
            # 读 past KV
            t0 = time.perf_counter()
            for layer_idx in range(pool.num_layers):
                K, V = pool.read_layer(layer_idx, [block_ids], [past_len],
                                        past_len, [block_offset])
                past_kv.append((K, V))
            mp.record("kv_head", mp.elapsed(t0))
            
            q_len = chunk_len
            kv_len = past_len + chunk_len
            attn_mask = torch.ones(1, 1, q_len, kv_len, device=self.device, dtype=torch.bool)
            causal = torch.tril(torch.ones(q_len, q_len, device=self.device, dtype=torch.bool))
            attn_mask[:, :, :, past_len:] = causal
        else:
            past_kv = None
            attn_mask = None
        
        # forward
        logits, past_key_values = self.model(
            input_ids, position_ids, past_key_values=past_kv, attention_mask=attn_mask)
        
        chunk_kv = [(k[:, :, past_len:, :], v[:, :, past_len:, :]) for k, v in past_key_values]
        # 写 KV
        t0 = time.perf_counter()
        pool.write_chunk_kv(block_ids, chunk_kv, token_start=chunk_start, block_offset=block_offset)
        mp.record("kv_write", mp.elapsed(t0))
        
        self._past_len[inp.request_id] = chunk_end
        
        if inp.is_last_chunk:
            next_token = logits[0, -1, :].argmax().item()
            self._generated[inp.request_id] = [next_token]
            return next_token
        else:
            return None
    def batch_prefill(self, inputs: list[PrefillInput]) -> list:
        mp = self.module_profiler
        pool = self.kv_manager.pool 
        
        results = [None] * len(inputs)
        
        chunk_inputs = []
        chunk_indices = []
        r_block_ids_list = []
        past_lens = []
        chunk_lens = []

        offsets = []

        for idx, inp in enumerate(inputs):
            chunk_len = inp.chunk_end - inp.chunk_start
            if chunk_len == 0:
                past_kv = []
                for layer_idx in range(pool.num_layers):
                    K, V = pool.read_layer(
                        layer_idx, [inp.block_ids], [inp.chunk_start],
                        inp.chunk_start, [inp.block_offset]
                    )
                    past_kv.append((K, V))

                input_ids = torch.tensor([[inp.token_ids[-1]]], device=self.device)
                position_ids = torch.tensor([[inp.chunk_start]], device=self.device)
                logits, _ = self.model(input_ids, position_ids, past_key_values=past_kv)

                next_token = logits[0, -1, :].argmax().item()
                self._past_len[inp.request_id] = inp.chunk_start
                self._generated[inp.request_id] = [next_token]
                results[idx] = next_token
                
            if chunk_len > 0:
                chunk_inputs.append(inp)
                r_block_ids_list.append(inp.block_ids)
                chunk_indices.append(idx)
                past_len = inp.chunk_start
                past_lens.append(past_len)
                chunk_lens.append(chunk_len)
                offsets.append(inp.block_offset)
        
        B = len(chunk_inputs) 
        if B == 0:
            return results
        
        max_past_len = max(past_lens) if past_lens else 0
        max_chunk_len = max(chunk_lens) if chunk_lens else 0
        
        batched_kvs = []
        
        t0 = time.perf_counter()
        for layer_idx in range(pool.num_layers):
            K_batch, V_batch = pool.read_layer(
                layer_idx, r_block_ids_list, past_lens, max_past_len, offsets
            )
            batched_kvs.append((K_batch, V_batch))
        mp.record("kv_head", mp.elapsed(t0))
            
        input_ids = torch.zeros(B, max_chunk_len, dtype=torch.long, device=self.device)
        position_ids = torch.zeros(B, max_chunk_len, dtype=torch.long, device=self.device)
        
        for i, inp in enumerate(chunk_inputs):
            cl = chunk_lens[i]
            chunk_tokens = inp.token_ids[inp.chunk_start:inp.chunk_end]
            input_ids[i, :cl] = torch.tensor(chunk_tokens, device=self.device)
            position_ids[i, :cl] = torch.arange(inp.chunk_start, inp.chunk_end, device=self.device)
        
        attn_mask = torch.zeros(B, 1, max_chunk_len, max_past_len + max_chunk_len, device=self.device, dtype=torch.bool)
        
        for i in range(B):
            pl = past_lens[i]
            cl = chunk_lens[i]
            if pl > 0:
                attn_mask[i, 0, :cl, :pl] = True
            if cl > 0:
                causal = torch.tril(torch.ones(cl, cl, device=self.device, dtype=torch.bool))
                attn_mask[i, 0, :cl, max_past_len:max_past_len + cl] = causal
            
        
        # forward
        logits, new_kvs = self.model(
            input_ids, position_ids,
            past_key_values=batched_kvs,
            attention_mask=attn_mask,
        )
        
        t0 = time.perf_counter()
        for i, inp in enumerate(chunk_inputs):
            cl = chunk_lens[i]
            
            chunk_kv = []
            for k, v in new_kvs:
                chunk_k = k[i:i+1, :, max_past_len:max_past_len + cl, :]
                chunk_v = v[i:i+1, :, max_past_len:max_past_len + cl, :]
                chunk_kv.append((chunk_k, chunk_v))
        
            pool.write_chunk_kv(list(inp.block_ids), chunk_kv, token_start=inp.chunk_start, block_offset=inp.block_offset)
        mp.record("kv_write", mp.elapsed(t0))
        
        for j, (inp, idx) in enumerate(zip(chunk_inputs, chunk_indices)):
            self._past_len[inp.request_id] = inp.chunk_end
            
            if inp.is_last_chunk:
                next_token = logits[j, -1, :].argmax().item()
                self._generated[inp.request_id] = [next_token]
                results[idx] = next_token
                
        return results
                
    def batch_decode(self, inputs: list[BatchDecodeInput]) -> list:
        """返回 [next_token_id, ...]"""
        mp = self.module_profiler
        
        B = len(inputs)
        pool = self.kv_manager.pool

        past_lens = []
        r_block_ids_list = []
        offsets = []
        request_positions = []
        for inp in inputs:
            past_len = self._past_len[inp.request_id]
            past_lens.append(past_len)
            r_block_ids_list.append(inp.block_ids)
            offsets.append(inp.block_offset)
            request_positions.append(past_len)

        max_past_len = max(past_lens) if past_lens else 0
        # 从 BlockPool 拼 KV (带 offset)
        batched_kvs = []
        
        # 读 KV
        t0 = time.perf_counter()
        for layer_idx in range(pool.num_layers):
            K_batch, V_batch = pool.read_layer(
                layer_idx, r_block_ids_list, past_lens, max_past_len, offsets
            )
            batched_kvs.append((K_batch, V_batch))
        mp.record("kv_head", mp.elapsed(t0))
        
        # attenion_mask：decode 时 query 只有 1 个新 token，需 attend 到过去所有 token + 自己
        attention_mask = torch.ones(B, 1, 1, max_past_len + 1, device=self.device).bool() # 
        for i in range(B):
            # 只 mask 掉不同请求之间的 padding 位
            if past_lens[i] < max_past_len:
                attention_mask[i, :, :, past_lens[i]:max_past_len] = False

        input_ids = torch.tensor([[inp.token_id] for inp in inputs], device=self.device)
        position_ids = torch.tensor([[pos] for pos in request_positions], device=self.device)

        # forward
        logits, new_kvs = self.model(
            input_ids, position_ids,
            past_key_values=batched_kvs,
            attention_mask=attention_mask,
        )

        next_tokens = []
        for i, inp in enumerate(inputs):
            next_token = logits[i, -1, :].argmax().item()
            if next_token == self.tokenizer.eos_token_id:
                next_tokens.append(None)
            else:
                next_tokens.append(next_token)
                self._generated[inp.request_id].append(next_token)
                # 更新 KV cache
                past_len = past_lens[i]
                token_kv = [
                    (k[i: i+1, :, -1:, :], v[i:i+1, :, -1:, :])
                    for k, v in new_kvs
                ]
                block_idx = (inp.block_offset + past_len) // pool.block_size
                pos_in_block = (inp.block_offset + past_len) % pool.block_size
                
                # 写 KV
                t0 = time.perf_counter()
                pool.write_token(inp.block_ids[block_idx], token_kv, pos_in_block)
                mp.record("kv_write", mp.elapsed(t0))
                
                self._past_len[inp.request_id] = past_len + 1

        return next_tokens
        
        
    def release(self, request_id: int):
        self._generated.pop(request_id, None)
        self._past_len.pop(request_id, None)

    def generated_text(self, request_id: int) -> str:
        return self.tokenizer.decode(self._generated.get(request_id, []))