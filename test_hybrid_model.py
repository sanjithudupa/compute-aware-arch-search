from transformers import AutoTokenizer
from qwen3_model import Qwen3WithLinearAttention
import torch

model = Qwen3WithLinearAttention.from_config_json(
    config_path="qwen3hybridconfig.json",
)

tokenizer = AutoTokenizer.from_pretrained("Qwen3-1.7B")

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
