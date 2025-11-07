import torch
import time

torch.manual_seed(67) # xD

def benchmark_model(model, tokenizer, seq_len=512, batch_size=10):
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
    torch.cpu.synchronize()
    start = time.perf_counter()
    with torch.no_grad():
        _ = model(input_ids=inputs, attention_mask=attention_mask)
    torch.cpu.synchronize()
    elapsed = time.perf_counter() - start
    tokens_processed = batch_size * seq_len
    
    throughput = tokens_processed / elapsed
    
    return throughput