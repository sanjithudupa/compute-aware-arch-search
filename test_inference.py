from transformers import AutoModelForCausalLM, AutoTokenizer
from benchmark import benchmark_model

model_path = "models/qwen3-1.7b"

print('loading model')
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cpu")
print('loaded model')

inputs = tokenizer("Hello, how are you?", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

# results = benchmark_model(model, tokenizer)
# print(results)