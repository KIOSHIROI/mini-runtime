import torch
from transformers import AutoTokenizer
from ..model.qwen2_model import Qwen2Model
from ..model.config import Qwen2Config
from ..model.loader import load_qwen2_weights
from ..kv_cache import KVCacheManager
import os

class NativeBackend:
    def __init__(self, model_path: str = "Qwen/Qwen2.5-0.5B-Instruct"):
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
        load_qwen2_weights(self.model, model_path)
        self.model.eval()

        self._past_len: dict[int, int] = {}
        self._generated: dict[int, list[int]] = {}  # request_id → 已生成token id

    def prefill(self, request_id: int,  prompt: str, block_ids: list[int]) -> int:
        """_summary_
        将 prompt 编码为 token_id 序列，构建 position_id 序列
        调用模型计算得到 logits 和 KV cache
        保存 KV cache 和 prompt 长度
        返回生成的第一个 token_id
        """
        # 1. 编码 prompt（用对话模板，Qwen2.5-Instruct 必须有）
        messages = [{"role": "user", "content": prompt}]
        chat_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        input_ids = self.tokenizer(chat_text, return_tensors="pt").input_ids
        # 2. 构造 position_ids: [[0, 1, 2, ..., L-1]]
        position_ids = torch.arange(input_ids.size(1)).unsqueeze(0)
        # 3. 调用 model(input_ids, position_ids)

        logits, past_key_values = self.model(input_ids, position_ids)

        # 4. 保存 KV cache 和 prompt_len
        # self._kv[request_id] = past_key_values
        self.kv_manager.pool.write_blocks(block_ids, past_key_values)
        
        self._past_len[request_id] = input_ids.size(1)
        # 5. 取 logits[0, -1, :] 做 argmax → next_token
        next_token = logits[0, -1, :].argmax().item()
        # 6. self._generated[request_id] = [next_token]
        self._generated[request_id] = [next_token]
        # 7. return next_token
        return next_token
    
    def batch_decode(self, requests: list[tuple[int, int, int]]) -> list:
        """_summary_
        requests: [(request_id, token_id, block_ids), ...]
        return [next_token_id, ...]
        """
        B = len(requests)
        pool = self.kv_manager.pool
        
        # 收集 past_len / position / block_ids
        past_lens = []
        r_block_ids_list = []
        request_positions = []
        for request_id, token_id, block_ids in requests:
            past_len = self._past_len[request_id]
            past_lens.append(past_len)
            r_block_ids_list.append(block_ids)
            request_positions.append(past_len)
        
        max_past_len = max(past_lens) if past_lens else 0
        # 从 BlockPool 拼 KV
        batched_kvs = []
        for layer_idx in range(pool.num_layers):
            K_batch, V_batch = pool.read_layer(
                layer_idx, r_block_ids_list, past_lens, max_past_len
            )
            batched_kvs.append((K_batch, V_batch))
        # attenion_mask
        attention_mask = torch.ones(B, 1, 1, max_past_len + 1).bool()  # [B, 1, 1, total_len]
        for i in range(B):
            attention_mask[i, :, :, :past_lens[i]] = False
            attention_mask[i, :, :, max_past_len] = False
            
        input_ids = torch.tensor([[token_id] for _, token_id, _ in requests])
        position_ids = torch.tensor([[pos] for pos in request_positions])

        logits, new_kvs = self.model(
            input_ids, position_ids, 
            past_key_values=batched_kvs, 
            attention_mask=attention_mask
            )
        
        next_tokens = []

        for i, (request_id, token_id, block_ids) in enumerate(requests):
            next_token = logits[i, -1, :].argmax().item()
            if next_token == self.tokenizer.eos_token_id:
                next_tokens.append(None)
            else:
                next_tokens.append(next_token)
                self._generated[request_id].append(next_token)
                # 更新 KV cache
                past_len = past_lens[i]
                token_kv = [
                    (k[i: i+1, :, -1:, :], v[i:i+1, :, -1:, :])
                    for k, v in new_kvs
                ]
                block_idx = past_len // pool.block_size
                pos_in_block = past_len % pool.block_size
                pool.write_token(block_ids[block_idx], token_kv, pos_in_block)
                self._past_len[request_id] = past_len + 1

        return next_tokens
        
        
    def release(self, request_id: int):
        self._generated.pop(request_id, None)
        self._past_len.pop(request_id, None)

    def generated_text(self, request_id: int) -> str:
        return self.tokenizer.decode(self._generated.get(request_id, []))