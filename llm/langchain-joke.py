import os
from langchain.llm import OpenAI

llm = OpenAI(model_name="gpt-3.5-turbo", n=2, temperature=0.5, max_token=1024)
output = llm("Tell me a joke. Less than 100 words.")
print(output)
