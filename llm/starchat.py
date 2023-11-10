import torch
from transformers import pipeline

pipe = pipeline("text-generation", model="/mnt/d/llm-models/starchat-beta", torch_dtype=torch.bfloat16, device_map="auto")

# We use a variant of ChatML to format each message
prompt_template = "<|system|>\n<|end|>\n<|user|>\n{query}<|end|>\n<|assistant|>"
prompt = prompt_template.format(query="How do I sort a list in Python?")
# We use a special <|end|> token with ID 49155 to denote ends of a turn
print(prompt)
print("--------------[result]-----------------------")
outputs = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.2, top_k=50, top_p=0.95, eos_token_id=49155)
# You can sort a list in Python by using the sort() method. Here's an example:\n\n```\nnumbers = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]\nnumbers.sort()\nprint(numbers)\n```\n\nThis will sort the list in place and print the sorted list.
print(outputs)
