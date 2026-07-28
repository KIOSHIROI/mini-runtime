import torch
import time
import torch.nn as nn
from .config import Qwen2Config
from .transformer_block import TransformerBlock
from .rms_norm import RMSNorm

class Qwen2Model(nn.Module):
    def __init__(self, config: Qwen2Config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.profiler = None  # backend 注入
        
    def forward(self, input_ids, position_ids, past_key_values=None, attention_mask=None):
        mp = self.profiler
        t0 = time.perf_counter() if mp else 0.0
        
        x = self.embed_tokens(input_ids)
        if mp: mp.record("forward/embedding", mp.elapsed(t0))
        
        attn_total = 0.0
        mlp_total = 0.0
        
        present_key_values = []
        for i, layer in enumerate(self.layers):
            past_kv = None if past_key_values is None else past_key_values[i]
            x, new_k, new_v, attn_ms, mlp_ms = layer(x, position_ids, past_kv, attention_mask)
            present_key_values.append((new_k, new_v)) # [layers, kv, num_heads, seq_len, head_dim]
            
            attn_total += attn_ms
            mlp_total += mlp_ms
        
        if mp: 
            mp.record("forward/attention", attn_total)
            mp.record("forward/mlp", mlp_total)
        
        t0 = time.perf_counter() if mp else 0.0
        x = self.norm(x) # (batch, seq_len, hidden_size)
        logits = self.lm_head(x[:, -1:, :]) # (batch, 1, vocab_size), -1: 保留 seq_len 维度
        if mp: mp.record("forward/norm+lm_head", mp.elapsed(t0))
        
        return logits, present_key_values