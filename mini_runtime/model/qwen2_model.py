import torch
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
    
    def forward(self, input_ids, position_ids, past_key_values=None, attention_mask=None):
        x = self.embed_tokens(input_ids)
        
        present_key_values = []
        for i, layer in enumerate(self.layers):
            past_kv = None if past_key_values is None else past_key_values[i]
            
            x, new_k, new_v = layer(x, position_ids, past_kv, attention_mask)
            
            present_key_values.append((new_k, new_v)) # [layers, kv, num_heads, seq_len, head_dim]
        
        x = self.norm(x) # (batch, seq_len, hidden_size)
        
        logits = self.lm_head(x[:, -1:, :]) # (batch, 1, vocab_size), -1: 保留 seq_len 维度
        
        return logits, present_key_values