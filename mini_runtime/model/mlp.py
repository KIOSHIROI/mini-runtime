import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import Qwen2Config
class MLP(nn.Module):
    def __init__(self, config: Qwen2Config):
        """
        使用 SwiGLU 
        GLU: 门控激活函数 gate_function(W1x) * W2x
        """
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate, bias=False)
        self.down_proj = nn.Linear(self.intermediate, self.hidden_size, bias=False)
    
    def forward(self, x):
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        return self.down_proj(F.silu(gate) * up)