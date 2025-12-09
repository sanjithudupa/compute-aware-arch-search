import torch
import sys
import os
import time
import pandas as pd

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from qwen3_model.modeling_qwen3 import Qwen3WithLinearAttention

RESULTS_PATH = os.path.join(project_root, "evaluation", "timing_results.csv")
if not os.path.exists(RESULTS_PATH):
    df = pd.DataFrame(columns=["config_name", "test_start_timestamp", "context_length", "n_generated", "time", "ttft", "invalidated"])
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

    CONTEXT_LENGTHS = [50, 300, 3000]
    MAX_NEW_TOKENS = 150

    for context_len in CONTEXT_LENGTHS:
        inputs = torch.randint(0, model.config.vocab_size, (1, context_len)).to(DEVICE)
        attention_mask = torch.ones_like(inputs).to(DEVICE)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = time.perf_counter()
        _ = model.generate(inputs, attention_mask=attention_mask, max_new_tokens=1, do_sample=False, use_cache=True)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        ttft = time.perf_counter() - start_time

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = time.perf_counter()
        generated = model.generate(inputs, attention_mask=attention_mask, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, use_cache=True)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        total_time = time.perf_counter() - start_time
        n_generated = generated.shape[1] - context_len
        
        print(f'{formatted_config_name} context={context_len}, generated={n_generated}, ttft={ttft:.4f}s, total={total_time:.4f}s')
        new_row = {
            "config_name": formatted_config_name,
            "test_start_timestamp": test_timestamp,
            "context_length": context_len,
            "n_generated": n_generated,
            "time": total_time,
            "ttft": ttft,
            "invalidated": False
        }
        df.loc[len(df)] = new_row
        df.to_csv(RESULTS_PATH, index=False)