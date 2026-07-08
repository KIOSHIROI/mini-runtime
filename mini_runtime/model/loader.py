import os
import glob

import torch
from safetensors.torch import load_file
from huggingface_hub import snapshot_download
from .qwen2_model import Qwen2Model

def load_qwen2_weights(model: Qwen2Model, model_path: str, device: torch.device) -> None:
    expanded = os.path.expanduser(model_path)
    if os.path.isdir(expanded):
        model_path = expanded
    else:
        model_path = snapshot_download(model_path)

    files = sorted(glob.glob(os.path.join(model_path, "*.safetensors")))
    if not files:
        # HF cache 结构: snapshots/<hash>/model.safetensors
        snapshot_dir = os.path.join(model_path, "snapshots")
        if os.path.isdir(snapshot_dir):
            files = sorted(glob.glob(os.path.join(snapshot_dir, "*", "*.safetensors")))
    if not files:
        raise FileNotFoundError(f"No .safetensors found in {model_path}")
    
    state_dict = {}
    for file in files:
        state_dict.update(load_file(file)) # load_file() -> dict[str, Tensor]

    def _map_key(hf_key: str) -> str:
        key = hf_key.replace("model.", "")
        key = key.replace("self_attn", "attention")
        key = key.replace("input_layernorm", "norm1")
        key = key.replace("post_attention_layernorm", "norm2")
        return key 

    mapped_state_dict = {} 
    for hf_key, value in state_dict.items():
        mapped_key = _map_key(hf_key)
        mapped_state_dict[mapped_key] = value
    
    if "lm_head.weight" not in mapped_state_dict:
        mapped_state_dict["lm_head.weight"] = mapped_state_dict["embed_tokens.weight"].clone()
        
    final_state_dict = {
        key: tensor.to(torch.float16)
        for key, tensor in mapped_state_dict.items()
    }
    
    model.to(device)
    model.load_state_dict(final_state_dict, strict=True)
