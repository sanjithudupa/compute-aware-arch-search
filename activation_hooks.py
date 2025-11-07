from functools import wraps
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def wrap_forward(module, name, activations):
    """wraps the forward pass of some module and sets it in the activations dictionary"""
    if 'inputs' not in activations:
        activations['inputs'] = {}
    if 'outputs' not in activations:
        activations['outputs'] = {}

    orig_forward = module.forward

    @wraps(orig_forward)
    def wrapped_forward(*args, **kwargs):
        def clean_tensors(obj):
            if torch.is_tensor(obj):
                return obj.detach().cpu()
            if isinstance(obj, (list, tuple)):
                tensors = [clean_tensors(x) for x in obj if torch.is_tensor(x) or isinstance(x, (list, tuple, dict))]
                tensors = [x for x in tensors if x is not None]
                return tensors if tensors else None
            if isinstance(obj, dict):
                tensors = {k: clean_tensors(v) for k, v in obj.items() if torch.is_tensor(v)}
                return tensors if tensors else None
            return None

        # extract only hidden_states-like activations
        cleaned_inputs = []
        if "hidden_states" in kwargs:
            cleaned_inputs.append(clean_tensors(kwargs["hidden_states"]))
        elif len(args) > 0:
            cleaned_inputs.append(clean_tensors(args[0]))

        out = orig_forward(*args, **kwargs)

        # extract only tensor outputs
        cleaned_outputs = []
        if torch.is_tensor(out):
            cleaned_outputs.append(out.detach().cpu())
        elif isinstance(out, (list, tuple)):
            for x in out:
                t = clean_tensors(x)
                if t is not None:
                    cleaned_outputs.append(t)
        elif isinstance(out, dict):
            for v in out.values():
                t = clean_tensors(v)
                if t is not None:
                    cleaned_outputs.append(t)

        activations["inputs"][name] = cleaned_inputs
        activations["outputs"][name] = cleaned_outputs
        return out

    module.forward = wrapped_forward
    return module

if __name__ == "__main__":
    model_path = "models/qwen3-1.7b"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cpu")
    model = model.to(torch.float32)
    print('loaded model')

    activations = {}
    for i, layer in enumerate(model.model.layers):
        wrap_forward(layer.self_attn, f"attn_{i}", activations)
    
    inputs = tokenizer("Hello, how are you?", return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=100)
    
    # print(activations)
    
    # for a single self attn block
    print(activations['inputs']['attn_0'], ' ==> ', activations['outputs']['attn_0'])
    