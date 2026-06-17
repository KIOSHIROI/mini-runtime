import torch
from transformers import AutoTokenizer
from ..model.qwen2_model import Qwen2Model
from ..model.config import Qwen2Config
from ..model.loader import load_qwen2_weights

import os

class NativeBackend:
    def __init__(self, model_path: str = "Qwen/Qwen2.5-0.5B-Instruct"):
        model_path = os.path.expanduser(model_path)

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

        self._kv: dict[int, list] = {}          # request_id → 24 层 KVcache
        self._generated: dict[int, list[int]] = {}  # request_id → 已生成token id
        self._prompt_len: dict[int, int] = {}       # request_id → prompt长度

    def prefill(self, request_id: int, prompt: str) -> int:
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
        self._kv[request_id] = past_key_values
        self._prompt_len[request_id] = input_ids.size(1)
        # 5. 取 logits[0, -1, :] 做 argmax → next_token
        next_token = logits[0, -1, :].argmax().item()
        # 6. self._generated[request_id] = [next_token]
        self._generated[request_id] = [next_token]
        # 7. return next_token
        return next_token

    def decode(self, request_id: int, token_id: int) -> int | None:
        # 1. 拿历史 KV
        past_kv = self._kv.get(request_id, None)
        # 2. 计算当前绝对位置: position = self._prompt_len[rid] +len(self._generated[rid])
        position = self._prompt_len.get(request_id, 0) + len(self._generated.get(request_id, [])) - 1
        # 3. input_ids = [[token_id]], position_ids = [[position]]
        input_ids = torch.tensor([[token_id]])
        position_ids = torch.tensor([[position]])
        # 4. 调用 model(input_ids, position_ids, past_key_values=past_kv)
        logits, past_key_values = self.model(input_ids, position_ids, past_key_values=past_kv)
        # 5. 更新 KV cache
        self._kv[request_id] = past_key_values
        # 6. 取 logits[0, -1, :] 做 argmax → next_token
        next_token = logits[0, -1, :].argmax().item()
        # 7. 如果 next_token == tokenizer.eos_token_id: return None
        if next_token == self.tokenizer.eos_token_id:
            return None
        # 8. self._generated[request_id].append(next_token)
        self._generated[request_id].append(next_token)
        
        # 9. return next_token
        return next_token
    
    def batch_decode(self, requests: list[tuple[int, int]]) -> list:
        """_summary_
        requests: [(request_id, token_id), ...]
        return [next_token_id, ...]
        """
        B = len(requests)
        request_kvs = []
        request_positions = []
        max_past_len = 0
        for request_id, token_id in requests:
            request_kvs.append(self._kv.get(request_id, None))
            position = self._prompt_len.get(request_id, 0) + self._generated.get(request_id, []).__len__() - 1
            request_positions.append(position)
            past_lens = self._kv[request_id][0][0].shape[2] # [layer0:(K_0 [batch, num_head, len, head_dim], V_0 ...) ...]
            max_past_len = max(max_past_len, past_lens)

        # pad KV cache
        pad_kv_caches = []
        for kv in request_kvs:
            pad_kv = kv[:]
            for i in range(len(pad_kv)):
                k, v = pad_kv[i]
                pad_len = max_past_len - k.size(2)
                if pad_len > 0:
                    k_pad = torch.zeros(k.size(0), k.size(1), pad_len, k.size(3))
                    v_pad = torch.zeros(v.size(0), v.size(1), pad_len, v.size(3))
                    k = torch.cat([k, k_pad], dim=2)
                    v = torch.cat([v, v_pad], dim=2)
                pad_kv[i] = (k, v)
            pad_kv_caches.append(pad_kv)

        batched_kvs = []
        num_layers = len(pad_kv_caches[0])
        for layer_idx in range(num_layers):
            Ks = torch.cat([pad_kv_caches[r][layer_idx][0] for r in range(B)], dim=0)
            Vs = torch.cat([pad_kv_caches[r][layer_idx][1] for r in range(B)], dim=0)
            batched_kvs.append((Ks, Vs))

        attention_mask = torch.ones(len(requests), 1, 1, max_past_len + 1).bool()  # [B, 1, 1, total_len]
        for i, (request_id, token_id) in enumerate(requests):
            past_len = self._kv[request_id][0][0].shape[2]
            attention_mask[i, :, :, :past_len] = False
            attention_mask[i, :, :, max_past_len] = False
        input_ids = torch.tensor([[token_id] for _, token_id in requests])
        position_ids = torch.tensor([[pos] for pos in request_positions])

        logits, new_kvs = self.model(input_ids, position_ids, past_key_values=batched_kvs, attention_mask=attention_mask)
        next_tokens = []

        for i, (request_id, token_id) in enumerate(requests):
            next_token = logits[i, -1, :].argmax().item()
            if next_token == self.tokenizer.eos_token_id:
                next_tokens.append(None)
            else:
                next_tokens.append(next_token)
                self._generated[request_id].append(next_token)
                # 更新 KV cache
                new_kv = []
                for layer_kv in new_kvs:
                    k, v = layer_kv
                    past_len = self._kv[request_id][0][0].shape[2]
                    new_k = k[i:i+1, :, :past_len+1, :]
                    new_v = v[i:i+1, :, :past_len+1, :]
                    new_kv.append((new_k, new_v))
                self._kv[request_id] = new_kv

        return next_tokens
        
        
    def release(self, request_id: int):
        self._kv.pop(request_id, None)
        self._generated.pop(request_id, None)
        self._prompt_len.pop(request_id, None)

    def generated_text(self, request_id: int) -> str:
        return self.tokenizer.decode(self._generated.get(request_id, []))