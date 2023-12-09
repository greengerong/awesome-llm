from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch
torch.manual_seed(1234)

model_name= '/mnt/d/llm-models/Qwen-VL-Chat'

# Note: The default behavior now has injection attack prevention off.
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# use bf16
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="auto", trust_remote_code=True, bf16=True).eval()
# use fp16
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="auto", trust_remote_code=True, fp16=True).eval()
# use cpu only
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="cpu", trust_remote_code=True).eval()
# use cuda device
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cuda", trust_remote_code=True).eval()


# Specify hyperparameters for generation
model.generation_config = GenerationConfig.from_pretrained(model_name, trust_remote_code=True)

# demo 1
query = tokenizer.from_list_format([
    {'image': 'https://raw.githubusercontent.com/greengerong/awesome-llm/main/assets/VL-冰箱.png'},
    {'text': '识别图片中商品名称以及对应商品价格，按照JSON格式(商品名称、商品价格2个属性)输出。'},
])
response, history = model.chat(tokenizer, query=query, history=None)
print(response)

print('---------------')

# demo 2
query = tokenizer.from_list_format([
     {'image': 'https://github.com/greengerong/awesome-llm/blob/main/assets/VL-%E5%B9%B3%E6%9D%BF.png?raw=true'},
    {'text': '识别图片中商品名称以及对应商品价格，按照JSON格式(商品名称、商品价格2个属性)输出。'},
])
response, history = model.chat(tokenizer, query=query, history=None)
print(response)


