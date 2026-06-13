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
    def release(self, request_id: int):
        self._kv.pop(request_id, None)
        self._generated.pop(request_id, None)
        self._prompt_len.pop(request_id, None)

    def generated_text(self, request_id: int) -> str:
        return self.tokenizer.decode(self._generated.get(request_id, []))