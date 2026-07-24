import torch
from transformers import AutoTokenizer
from ..model.qwen2_model import Qwen2Model
from ..model.config import Qwen2Config
from ..model.loader import load_qwen2_weights
from ..kv_cache import KVCacheManager
from dataclasses import dataclass
import os

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


        self._past_len: dict[int, int] = {}
        self._generated: dict[int, list[int]] = {}  # request_id → 已生成token id

    def prefill(self, inp: PrefillInput) -> int:
        """prefix-aware prefill。只计算 token_ids[skip_tokens:]，复用匹配部分的 KV。"""
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
    
    def batch_decode(self, inputs: list[BatchDecodeInput]) -> list:
        """返回 [next_token_id, ...]"""
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
        for layer_idx in range(pool.num_layers):
            K_batch, V_batch = pool.read_layer(
                layer_idx, r_block_ids_list, past_lens, max_past_len, offsets
            )
            batched_kvs.append((K_batch, V_batch))
        # attenion_mask：decode 时 query 只有 1 个新 token，需 attend 到过去所有 token + 自己
        attention_mask = torch.ones(B, 1, 1, max_past_len + 1, device=self.device).bool()
        for i in range(B):
            # 只 mask 掉不同请求之间的 padding 位
            if past_lens[i] < max_past_len:
                attention_mask[i, :, :, past_lens[i]:max_past_len] = False

        input_ids = torch.tensor([[inp.token_id] for inp in inputs], device=self.device)
        position_ids = torch.tensor([[pos] for pos in request_positions], device=self.device)

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
                pool.write_token(inp.block_ids[block_idx], token_kv, pos_in_block)
                self._past_len[inp.request_id] = past_len + 1

        return next_tokens
        
        
    def release(self, request_id: int):
        self._generated.pop(request_id, None)
        self._past_len.pop(request_id, None)

    def generated_text(self, request_id: int) -> str:
        return self.tokenizer.decode(self._generated.get(request_id, []))