
# note that these are 1-indexed
GLA_LOSSES = {
    3: 0.0003,
    9: 0.1059,
    10: 0.1186,
    12: 0.1672,
    11: 0.1686,
    7: 0.2129,
    8: 0.2396,
    1: 0.3459,
    2: 0.3585,
    5: 0.3773,
    4: 0.3856,
    6: 0.4688
}

# note these are 1-indexed
RWKV7_LOSSES = {
    1: 0.14544,
    3: 0.15398,
    21: 0.16705,
    22: 0.16833,
    20: 0.18859,
    2: 0.19184,
    19: 0.20476,
    18: 0.2346,
    4: 0.26388,
    17: 0.27872,
    5: 0.31018,
    16: 0.364450,
    11: 0.42294,
    6: 0.43012,
    15: 0.44016,
    7: 0.447,
    14: 0.4526,
    12: 0.47817,
    13: 0.513,
    8: 0.56123,
    9: 0.69808,
    10: 0.70282
}

ALLOWED_RANGES = {
    "gla": range(1, 11),
    "rwkv7": range(11, 23)
}

import json
import os

def generate_config_for_percentile(percentile, output_dir="hybrid_model_configs"):
    num_layers = 28
    layer_types = ["full_attention"] * num_layers
    
    selected_gla = []
    selected_rwkv7 = []
    
    filtered_gla = [(k, v) for k, v in GLA_LOSSES.items() if k in ALLOWED_RANGES["gla"]]
    filtered_rwkv7 = [(k, v) for k, v in RWKV7_LOSSES.items() if k in ALLOWED_RANGES["rwkv7"]]
    
    if percentile == 10:
        top_gla_count = max(1, int(len(filtered_gla) * 0.1))
        top_rwkv7_count = max(1, int(len(filtered_rwkv7) * 0.1))
    elif percentile == 25:
        top_gla_count = max(1, int(len(filtered_gla) * 0.25))
        top_rwkv7_count = max(1, int(len(filtered_rwkv7) * 0.25))
    elif percentile == 50:
        top_gla_count = max(1, int(len(filtered_gla) * 0.5))
        top_rwkv7_count = max(1, int(len(filtered_rwkv7) * 0.5))
    else:
        raise ValueError("Percentile must be 10, 25, or 50")
    
    sorted_gla = sorted(filtered_gla, key=lambda x: x[1])[:top_gla_count]
    sorted_rwkv7 = sorted(filtered_rwkv7, key=lambda x: x[1])[:top_rwkv7_count]
    
    for layer_idx, _ in sorted_gla:
        selected_gla.append(layer_idx)
        layer_types[layer_idx - 1] = "gla"
    
    for layer_idx, _ in sorted_rwkv7:
        selected_rwkv7.append(layer_idx)
        layer_types[layer_idx - 1] = "rwkv7"
    
    config = {
        "base_model_path": "Qwen3-1.7B",
        "weights_base_path": "linear-attention-checkpoints",
        "description": f"Top {percentile}% - GLA layers: {sorted(selected_gla)}, RWKV7 layers: {sorted(selected_rwkv7)}",
        "layer_attention_types": layer_types,
        "rwkv7_config_path": "linear_attn/rwkv7_config.json",
        "gla_config_path": "linear_attn/gla_config.json"
    }
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"top{percentile}.json")
    with open(output_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Generated {output_path}")

if __name__ == "__main__":
    for percentile in [10, 25, 50]:
        generate_config_for_percentile(percentile)
