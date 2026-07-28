import torch
from mini_runtime.model.qwen2_model import Qwen2Model
from mini_runtime.model.config import Qwen2Config
from mini_runtime.model.loader import load_qwen2_weights

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = Qwen2Model(Qwen2Config())
load_qwen2_weights(model, "Qwen/Qwen2.5-0.5B-Instruct", DEVICE)
print("loaded successfully")
print(f"params: {sum(p.numel() for p in model.parameters()):,}")