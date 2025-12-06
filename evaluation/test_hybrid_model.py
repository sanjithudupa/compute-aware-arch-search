import os
import sys

# Add project root to Python path to allow imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from transformers import AutoTokenizer
from models.qwen3_model import Qwen3WithLinearAttention
import torch

model = Qwen3WithLinearAttention.from_config_json(
    config_path="configs/hybrid_model_configs/top10.json",
)

tokenizer = AutoTokenizer.from_pretrained("Qwen3-1.7B")

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
model.eval()

text = "The square root of 144 is"

inputs = tokenizer(text, return_tensors="pt").to(device)

with torch.no_grad():
    generated = model.generate(
        inputs["input_ids"],
        max_new_tokens=50,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
print(f"Generated text: {generated_text}")
