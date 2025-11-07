from transformers import AutoModelForCausalLM, AutoTokenizer
from linear_attn.simple import SimpleBlock
from copy import deepcopy
from benchmark import benchmark_model
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
example output from when I ran it (i think it fucked up on the last two so i have to run it again)
loaded model
original model throughput:  348.89910080205584
linearized model with 3 blocks
linearized model throughput: 280.5871691087895 tokens/sec
linearized model with 7 blocks
linearized model throughput: 283.1005132271336 tokens/sec
linearized model with 14 blocks
linearized model throughput: 14.940665790337397 tokens/sec
linearized model with 28 blocks
linearized model throughput: 341.3334594409137 tokens/sec
results:  {0: 348.89910080205584, 3: 280.5871691087895, 7: 283.1005132271336, 14: 14.940665790337397, 28: 341.3334594409137}

clearly something is very wrong because the throughput is going down but the skeleton is there
"""