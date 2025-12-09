import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
import os
import time
import pandas as pd

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from qwen3_model.modeling_qwen3 import Qwen3WithLinearAttention

RESULTS_PATH = os.path.join(project_root, "evaluation", "throughput_results.csv")
if not os.path.exists(RESULTS_PATH):
    df = pd.DataFrame(columns=["config_name", "test_start_timestamp", "n_tokens", "time", "invalidated"])
    df.to_csv(RESULTS_PATH, index=False)
else:
    df = pd.read_csv(RESULTS_PATH)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(67) # 67 !!!!!!!!

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
config_path = os.path.join(project_root, "configs", "hybrid_model_configs")
config_names = os.listdir(config_path)

for i in range(len(config_names)):
    config_name = config_names[i]
    if (not torch.cuda.is_available()) and (config_name != "control.json"):
        print(f'{config_name} not running on GPU, skipping')
        continue
    
    formatted_config_name = config_name.split(".")[0]

    print(f'loading model: {config_name}')
    model_path = os.path.join(config_path, config_name)
    model = Qwen3WithLinearAttention.from_config_json(config_path=model_path).to(DEVICE)
    model.eval()

    print(f'loaded model: {formatted_config_name}')

    test_timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    N = [50, 300, 3000]

    for n in N:
        # generate a sequence of 300 random tokens, measure the time it takes to generate
        print(f'{formatted_config_name} generating {n} random tokens')
        inputs = torch.randint(0, model.config.vocab_size, (1, n)).to(DEVICE)
        attention_mask = torch.ones_like(inputs).to(DEVICE)

        start_time = time.perf_counter()
        outputs = model(inputs, attention_mask=attention_mask)
        end_time = time.perf_counter()
        
        print(f'{formatted_config_name} time to generate {n} random tokens: {end_time - start_time} seconds')
        new_row = {
            "config_name": formatted_config_name,
            "test_start_timestamp": test_timestamp,
            "n_tokens": n,
            "time": end_time - start_time,
            "invalidated": False
        }
        df.loc[len(df)] = new_row
        df.to_csv(RESULTS_PATH, index=False)