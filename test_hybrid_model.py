from transformers import AutoTokenizer
from qwen3_model import Qwen3WithLinearAttention, SUPPORTED_ATTENTION_VARIANTS
import torch

model_path = "Qwen3-1.7B"
weights_base_path = "linear_attention_checkpoints"

layer_attention_types = (
    ["gla"] * 10 +           # Layers 0-9: GLA
    ["rwkv7"] * 10 +         # Layers 10-19: RWKV7
    ["full_attention"] * 8    # Layers 20-27: Full self-attention
)

tokenizer = AutoTokenizer.from_pretrained(model_path)

model = Qwen3WithLinearAttention.from_pretrained(
    base_model_path=model_path,
    layer_attention_types=layer_attention_types,
    weights_base_path=weights_base_path,
    rwkv7_config_path="linear_attn/rwkv7_config.json",
    gla_config_path="linear_attn/gla_config.json",
)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
model.eval()

text = "The square root of 144 is"

inputs = tokenizer(text, return_tensors="pt").to(device)
with torch.no_grad():
    outputs = model(**inputs)

with torch.no_grad():
    generated = model.generate(
        inputs["input_ids"],
        max_new_tokens=50,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )

generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
print(f"Generated text: {generated_text}")
