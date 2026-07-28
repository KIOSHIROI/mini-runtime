import torch
import time
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
        
    def forward(self, x, position_ids, past_kv, attention_mask):
        """
        Pre_Norm 架构 
        残差连接
        """
        t0 = time.perf_counter()
        x_attn, new_k, new_v = self.attention(self.norm1(x), position_ids, past_kv, attention_mask)
        attn_ms = (time.perf_counter() - t0) * 1000

        x = x + x_attn

        t0 = time.perf_counter()
        x = x + self.mlp(self.norm2(x))
        mlp_ms = (time.perf_counter() - t0) * 1000

        return x, new_k, new_v, attn_ms, mlp_ms