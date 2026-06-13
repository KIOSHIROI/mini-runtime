class Qwen2Config:
    hidden_size: int = 896
    num_layers: int = 24 
    num_attention_heads: int = 14
    num_kv_heads: int = 2 
    head_dim: int = 64 
    intermediate_size: int = 4864 
    vocab_size: int = 151936
    max_position_embeddings: int = 32768
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    tie_word_embeddings: bool = True
    @property 
    def num_key_value_groups(self) -> int:
        return self.num_attention_heads // self.num_kv_heads