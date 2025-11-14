
from transformers import AutoTokenizer, AutoConfig
from qwen3_model import Qwen3ForCausalLM, Qwen3ForNAS
import torch
from fla.layers import MultiScaleRetention
from transformers import Trainer
 

model_path = "Qwen3-1.7B"

tokenizer = AutoTokenizer.from_pretrained(model_path)

import json
config = AutoConfig.from_pretrained(model_path)
with open(f"{model_path}/config.json", "r") as f:
    config_dict = json.load(f)
    if "attention_hook_idx" in config_dict:
        config.attention_hook_idx = config_dict["attention_hook_idx"]
    else:
        print("attention_hook_idx not found in config.json, setting default to 2")
        config.attention_hook_idx = 2

# model = Qwen3ForCausalLM.from_pretrained(
#     model_path,
#     config=config,
#     torch_dtype=torch.float32
# )
model = Qwen3ForNAS.from_pretrained(model_path, config=config, torch_dtype=torch.float32)
model = model.to("cpu")
input_ids = tokenizer("hello how are you?", return_tensors="pt").input_ids
result = model(input_ids=input_ids)
prev_hidden, current_hidden = result
print(f"prev_hidden_states (before layer {config.attention_hook_idx}): {prev_hidden.shape}")
print(f"current_hidden_states (after layer {config.attention_hook_idx}): {current_hidden.shape}")
print(prev_hidden[-1][-1][-1])
print(current_hidden[-1][-1][-1])


#needed to pip install triton and fla  - going to start needing to use gpu for this
linear_attention_layers = [MultiScaleRetention(hidden_size=2048, num_heads=16).to(device='cpu', dtype=torch.float32)]

optimizer = torch.optim.Adam(retnet_model.parameters(), lr=1e-4)
loss_fn = torch.nn.MSELoss()

for layer in linear_attention_layers:

