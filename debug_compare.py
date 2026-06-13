"""Layer-by-layer comparison: our Qwen2Model vs HF original."""
import torch
import os

# ---- 路径 ----
BASE = os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct")
SNAPSHOT = os.path.join(BASE, "snapshots")
MODEL_PATH = os.path.join(SNAPSHOT, sorted(os.listdir(SNAPSHOT))[0])

from transformers import AutoModelForCausalLM, AutoTokenizer
print("Loading models...")
hf = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16)
hf.eval()
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

from mini_runtime.model.qwen2_model import Qwen2Model
from mini_runtime.model.config import Qwen2Config
from mini_runtime.model.loader import load_qwen2_weights
our = Qwen2Model(Qwen2Config())
load_qwen2_weights(our, MODEL_PATH)
our.eval()

# ---- 输入 ----
chat = tokenizer.apply_chat_template(
    [{"role": "user", "content": "一句话介绍杭州"}],
    tokenize=False, add_generation_prompt=True,
)
inputs = tokenizer(chat, return_tensors="pt")
input_ids = inputs["input_ids"]
position_ids = torch.arange(input_ids.size(1)).unsqueeze(0)

# ---- Layer-by-layer hooks ----
our_outputs = {}
hf_outputs = {}

def hook_factory(name):
    def hook(module, inp, out):
        if isinstance(out, tuple):
            out = out[0]
        if isinstance(out, torch.Tensor):
            our_outputs[name] = out.detach().clone()
    return hook

def hf_hook_factory(name):
    def hook(module, inp, out):
        if isinstance(out, tuple):
            out = out[0]
        if isinstance(out, torch.Tensor):
            hf_outputs[name] = out.detach().clone()
    return hook

# Hook key points in our model
our.embed_tokens.register_forward_hook(hook_factory("embed"))
our.layers[0].norm1.register_forward_hook(hook_factory("norm1"))
our.layers[0].attention.o_proj.register_forward_hook(hook_factory("attn_out"))
our.layers[0].norm2.register_forward_hook(hook_factory("norm2"))
our.layers[0].mlp.down_proj.register_forward_hook(hook_factory("mlp_out"))

# Hook HF model
hf.model.embed_tokens.register_forward_hook(hf_hook_factory("embed"))
hf.model.layers[0].input_layernorm.register_forward_hook(hf_hook_factory("norm1"))
hf.model.layers[0].self_attn.o_proj.register_forward_hook(hf_hook_factory("attn_out"))
hf.model.layers[0].post_attention_layernorm.register_forward_hook(hf_hook_factory("norm2"))
hf.model.layers[0].mlp.down_proj.register_forward_hook(hf_hook_factory("mlp_out"))

# ---- 前向传播 ----
print("Running forward pass...")
with torch.no_grad():
    _, _ = our(input_ids, position_ids)
    hf_out = hf(input_ids, position_ids=position_ids)

# ---- 对比 ----
print("\n" + "=" * 50)
print("Layer-by-layer comparison (layer 0)")
print("=" * 50)

def cmp(name):
    our_val = our_outputs.get(name)
    hf_val = hf_outputs.get(name)
    if our_val is None or hf_val is None:
        print(f"  {name:>12}: missing data")
        return
    diff = (our_val.float() - hf_val.float()).abs().max().item()
    status = "✓" if diff < 0.01 else ("⚠" if diff < 0.1 else "✗")
    print(f"  {name:>12}: diff={diff:.6f}  {status}")

cmp("embed")
cmp("norm1")
cmp("attn_out")
cmp("norm2")
cmp("mlp_out")

# ---- Final logits comparison ----
print(f"\n  Final logits diff (layer 24):")
our_logits = our_outputs.get("mlp_out")
hf_logits = hf_outputs.get("mlp_out")
# Full model output
print(f"  (Hook captures layer 0 only; full model logits below)")

# ---- 完整模型输出对比 ----
print("\n" + "=" * 50)
print("Full model output")
print("=" * 50)

with torch.no_grad():
    our_logits_full, _ = our(input_ids, position_ids)
    hf_out_full = hf(input_ids, position_ids=position_ids)
    hf_logits_full = hf_out_full.logits

# Compare last-token logits
our_last = our_logits_full[0, -1, :]
hf_last = hf_logits_full[0, -1, :]

diff_full = (our_last.float() - hf_last.float()).abs().max().item()
print(f"  Last-token logits max diff: {diff_full:.6f}")

# Top-5 tokens
our_top5 = our_last.topk(5)
hf_top5 = hf_last.topk(5)
print(f"  Our top-5:  {[(tokenizer.decode([t]), our_top5.values[i].item()) for i, t in enumerate(our_top5.indices.tolist())]}")
print(f"  HF  top-5:  {[(tokenizer.decode([t]), hf_top5.values[i].item()) for i, t in enumerate(hf_top5.indices.tolist())]}")

# Check if argmax matches
our_argmax = our_last.argmax().item()
hf_argmax = hf_last.argmax().item()
print(f"  Argmax match: {'YES' if our_argmax == hf_argmax else 'NO (our=' + tokenizer.decode([our_argmax]) + ', hf=' + tokenizer.decode([hf_argmax]) + ')'}")

print("\n✓ All checks complete")
