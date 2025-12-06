import os
import sys

# Add project root to Python path to allow imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from transformers import AutoModelForCausalLM, AutoTokenizer
from models.linear_attn.simple import SimpleBlock
from copy import deepcopy
from utils.benchmark import benchmark_model
import torch

model_path = "models/qwen3-1.7b"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cpu")
model = model.to(torch.float32)
print('loaded model')

results = benchmark_model(model, tokenizer)
print('original model throughput: ', results)

# simple test that tries a couple different granularities of distributing the linear blocks and possible throughput improvements

TEST_GRANULARITIES = [8, 4, 2, 1]

results_dict = {0: results}

for granularity in TEST_GRANULARITIES:    
    linear_model = deepcopy(model)
    for i in range(0, 28, granularity):
        linear_block = SimpleBlock(hidden_dim=2048).to(torch.float32)
        linear_model.model.layers[i].self_attn = linear_block

    num_linear_blocks = 28 // granularity
    print(f'linearized model with {num_linear_blocks} blocks')
    results = benchmark_model(linear_model, tokenizer) # btw this takes a long ass time, you can reduce batch size or sequence lenght to speed it up
    print(f'linearized model throughput: {results} tokens/sec')
    
    results_dict[num_linear_blocks] = results

print('results: ', results_dict)

"""
example output from when I ran it (but some things are weird bc why is the throughput not changing as expected??)
loaded model
original model throughput:  327.4795039528691
linearized model with 3 blocks
linearized model throughput: 269.4920818506612 tokens/sec
linearized model with 7 blocks
linearized model throughput: 330.8355892460407 tokens/sec
linearized model with 14 blocks
linearized model throughput: 338.8972176912656 tokens/sec
linearized model with 28 blocks
linearized model throughput: 319.70413197785155 tokens/sec
results:  {0: 327.4795039528691, 3: 269.4920818506612, 7: 330.8355892460407, 14: 338.8972176912656, 28: 319.70413197785155}
"""