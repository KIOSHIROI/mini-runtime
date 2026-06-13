import torch
from .config import Qwen2Config
import torch.nn as nn
from .rotary import precompute_inv_cis, apply_rotary_pos_emb
from .rms_norm import RMSNorm

class Attention(nn.Module):
    def __init__(self, config: Qwen2Config):
        super().__init__()
        self.config = config
        self.head_dim = config.head_dim 
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_kv_heads
        self.num_kv_groups = config.num_key_value_groups
        self.hidden_size = config.hidden_size
        
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * config.head_dim, bias=True)
        self.k_proj = nn.Linear(config.hidden_size, config.num_kv_heads * config.head_dim, bias=True)
        self.v_proj = nn.Linear(config.hidden_size, config.num_kv_heads * config.head_dim, bias=True)
        self.o_proj = nn.Linear(config.num_attention_heads * config.head_dim, config.hidden_size, bias=False)
        
        self.freqs_cis = precompute_inv_cis(self.head_dim, self.config.max_position_embeddings, self.config.rope_theta)

    def forward(self, x, position_ids,past_kv):
        batch, seq_len, _ = x.shape
        
        Q = self.q_proj(x).view(-1, seq_len, self.num_heads, self.head_dim).transpose(1, 2) # batch, num_heads, seq_len, head_dim
        K = self.k_proj(x).view(-1, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2) # batch, num_kv_heads, seq_len, head_dim
        V = self.v_proj(x).view(-1, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2) # batch, num_kv_heads, seq_len, head_dim
        Q = apply_rotary_pos_emb(Q, self.freqs_cis, position_ids)
        K = apply_rotary_pos_emb(K, self.freqs_cis, position_ids)
        
        if past_kv is not None:
            K = torch.cat([past_kv[0], K], dim=2)
            V = torch.cat([past_kv[1], V], dim=2)
        
        K_cache = K[:]
        V_cache = V[:]
         
        K = K.repeat_interleave(self.num_kv_groups, dim=1)
        V = V.repeat_interleave(self.num_kv_groups, dim=1)

        attn_output = nn.functional.scaled_dot_product_attention(
            Q, K, V, 
            is_causal=(Q.shape[2]>1)
            )

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        output = self.o_proj(attn_output)
        
        return output, K_cache, V_cache
    
        