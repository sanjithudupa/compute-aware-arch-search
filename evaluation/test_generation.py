"""
Simple script to test if the trained hybrid model can generate coherent English.
Run on RunPod where the model weights are stored.
"""
import os
import sys

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
from transformers import AutoTokenizer
from safetensors.torch import load_file

# Import the custom model
from qwen3_model import Qwen3WithLinearAttention


def test_generation(device: str = "cuda"):
    print(f"Using device: {device}")
    
    model_path = "/workspace/compute-aware-arch-search"
    config_path = "configs/hybrid_model_configs/top10_gla.json"
    
    print(f"Loading model with config: {config_path}")
    
    # Load model architecture from config
    model = Qwen3WithLinearAttention.from_config_json(config_path=config_path)
    
    # Load the trained weights from safetensors
    safetensor_files = sorted([f for f in os.listdir(model_path) if f.endswith('.safetensors')])
    print(f"Loading weights from: {safetensor_files}")
    
    state_dict = {}
    for sf_file in safetensor_files:
        sf_path = os.path.join(model_path, sf_file)
        state_dict.update(load_file(sf_path))
    
    # Load weights (strict=False in case of minor mismatches)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"Missing keys: {len(missing)}")
    if unexpected:
        print(f"Unexpected keys: {len(unexpected)}")
    
    model = model.to(device).to(torch.float16)
    print("Model loaded successfully!")
    
    model.eval()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen3-1.7B")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Test prompts - simple ones to check basic English capability
    test_prompts = [
        "Hello, my name is",
        "The capital of France is",
        "1 + 1 =",
        "Once upon a time",
        "The sky is",
    ]
    
    print("\n" + "=" * 60)
    print("GENERATION TEST")
    print("=" * 60)
    
    for prompt in test_prompts:
        print(f"\nPrompt: '{prompt}'")
        print("-" * 40)
        
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                max_new_tokens=30,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Show just the generated part
        new_text = generated_text[len(prompt):]
        print(f"Generated: {prompt}{new_text}")
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_generation(device=device)

