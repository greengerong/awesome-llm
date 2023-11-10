from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

model_name = '/mnt/d/llm-models/Qwen-7B'

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", trust_remote_code=True).eval()
model.generation_config = GenerationConfig.from_pretrained(model_name, trust_remote_code=True) # 可指定不同的生成长度、top_p等相关超参

def askAi(prompt):
    inputs = tokenizer(prompt, return_tensors='pt')
    inputs = inputs.to('cuda:0')
    pred = model.generate(**inputs)
    print(tokenizer.decode(pred.cpu()[0], skip_special_tokens=True))
    # 蒙古国的首都是乌兰巴托（Ulaanbaatar）\n冰岛的首都是雷克雅未克（Reykjavik）\n埃塞俄比亚的首都是亚的斯亚贝巴（Addis Ababa）...

# askAi("蒙古国的首都是乌兰巴托（Ulaanbaatar）\n冰岛的首都是雷克雅未克（Reykjavik）\n埃塞俄比亚的首都是")
print("---------------------------")
askAi("实现java 8皇后算法，请给出java代码")