import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE = "cuda" # :(

MODEL_NAME = "Qwen/Qwen3-1.7B"
SAVE_DIR = MODEL_NAME.split('/')[-1]  # Save to "Qwen3-1.7B" in root directory

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,
    device_map=DEVICE
)

tokenizer.save_pretrained(SAVE_DIR)
model.save_pretrained(SAVE_DIR)

print(f"Downloaded {MODEL_NAME} to {SAVE_DIR}")