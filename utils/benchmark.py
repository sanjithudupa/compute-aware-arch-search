import torch
import time

torch.manual_seed(67) # xD

def benchmark_model(model, tokenizer, seq_len=256, batch_size=2):
    """ purely measures throughput, not accuracy or latency (latency shouldn't be changed by our modifiications I think??) """
    
    inputs = torch.randint(
        low=0, high=tokenizer.vocab_size, 
        size=(batch_size, seq_len),
        dtype=torch.long,
        device=model.device
    )
    attention_mask = torch.ones_like(inputs)

    # warm up
    with torch.no_grad():
        _ = model(input_ids=inputs, attention_mask=attention_mask)

    # benchmark forward pass
    if next(model.parameters()).device.type == 'cpu':
        torch.cpu.synchronize()
    else:
        torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        _ = model(input_ids=inputs, attention_mask=attention_mask)
    if next(model.parameters()).device.type == 'cpu':
        torch.cpu.synchronize()
    else:
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    tokens_processed = batch_size * seq_len
    
    throughput = tokens_processed / elapsed
    
    return throughput