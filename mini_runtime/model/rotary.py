import torch

def precompute_inv_cis(head_dim, max_position, theta):
    inv = 1 / (theta ** (torch.arange(0, head_dim, 2) / head_dim))
    t = torch.arange(max_position)
    freqs = torch.einsum('i,j->ij', t, inv) # max_position, head_dim // 2
    cos_cached = freqs.cos()
    sin_cached = freqs.sin()
    
    return cos_cached, sin_cached

def apply_rotary_pos_emb(x, freqs_cis, position_ids):
    cos_cached, sin_cached = freqs_cis
    cos = cos_cached[position_ids].unsqueeze(1)
    sin = sin_cached[position_ids].unsqueeze(1)
    x1, x2 = x.chunk(2, dim=-1)
    x_rotated_1 = x1 * cos - x2 * sin
    x_rotated_2 = x1 * sin + x2 * cos
    x_out = torch.cat((x_rotated_1, x_rotated_2), dim=-1)
    return x_out.type_as(x)
    
    
# 旧版本：使用复数实现的版本，计算教慢，内存开销大
# def precompute_inv_cis(head_dim, max_position, theta):
#     inv = 1 / (theta ** (torch.arange(0, head_dim, 2) / head_dim))
#     t = torch.arange(max_position)
#     freqs = torch.einsum('i,j->ij', t, inv) # max_position, head_dim // 2
#     return torch.polar(torch.ones_like(freqs), freqs) # r * e**(cos + i sin)

# def apply_rotary_pos_emb(x, freqs_cis, position_ids):
#     """
#     position_ids: (batch, seq_len)
#     """
#     shape = x.shape
#     x_reshaped = x.float().reshape(*shape[:-1], 2, -1) # batch, seq_len, num_heads, head_dim // 2, 2
#     x_complex = torch.complex(x_reshaped[..., 0, :], x_reshaped[..., 1, :])
#     freqs_cis = freqs_cis[position_ids] # (batch, seq_len, head_dim // 2) 涉及Pytorch高级索引规则
#     freqs_cis = freqs_cis.unsqueeze(1) # (batch, 1, seq_len, head_dim // 2)
#     x_rotated = x_complex * freqs_cis
#     x_out = torch.stack([x_rotated.real, x_rotated.imag], dim=-2).flatten(-2)
#     return x_out.type_as(x)
