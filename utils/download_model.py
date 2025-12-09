import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE = "cpu" # :(

MODEL_NAMES = [
    # "Qwen/Qwen3-8B",
    "Qwen/Qwen3-1.7B",
]

for model_name in MODEL_NAMES:
    save_dir = model_name.split('/')[-1]  # Save to "Qwen3-8B" or "Qwen3-1.7B" in root directory

    print(f"Downloading {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map=DEVICE
    )

    tokenizer.save_pretrained(save_dir)
    model.save_pretrained(save_dir)

    print(f"Downloaded {model_name} to {save_dir}")
