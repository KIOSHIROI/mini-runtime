from mini_runtime.model.qwen2_model import Qwen2Model
from mini_runtime.model.config import Qwen2Config
from mini_runtime.model.loader import load_qwen2_weights

model = Qwen2Model(Qwen2Config())
load_qwen2_weights(model, "~/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct")
print("loaded successfully")
print(f"params: {sum(p.numel() for p in model.parameters()):,}")