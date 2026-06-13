import torch
import torch.nn as nn
from .config import Qwen2Config
from .attention import Attention
from .mlp import MLP
from .rms_norm import RMSNorm

class TransformerBlock(nn.Module):
    def __init__(self, config: Qwen2Config):
        super().__init__()
        self.attention = Attention(config)
        self.mlp = MLP(config)
        self.norm1 = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.norm2 = RMSNorm(config.hidden_size, config.rms_norm_eps)
        
    def forward(self, x, position_ids, past_kv):
        """
        Pre_Norm 架构 
        残差连接
        """
        x_attn, K_cache, V_cache = self.attention(self.norm1(x), position_ids, past_kv)
        x = x + x_attn
        
        x = x + self.mlp(self.norm2(x))
        return x, K_cache, V_cache