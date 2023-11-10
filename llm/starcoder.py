from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

checkpoint = "/mnt/d/llm-models/starcoder"
device = "cuda" # for GPU usage or "cpu" for CPU usage

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map="auto", load_in_8bit=True)

inputs = tokenizer.encode("def print_hello_world_with_json():", return_tensors="pt").to(device)
attention_mask = torch.ones(inputs.shape, dtype=torch.long, device=device)
outputs = model.generate(inputs, max_length=128, do_sample=True, num_return_sequences=1, pad_token_id=model.config.eos_token_id, attention_mask=attention_mask)

print("-------------------------------------")
print(tokenizer.decode(outputs[0], clean_up_tokenization_spaces=False))