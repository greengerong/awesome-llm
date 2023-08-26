from transformers import  AutoModelForCausalLM, AutoTokenizer, pipeline
import transformers
import torch

# model = "codellama/CodeLlama-34b-hf"
model_name = "/mnt/d/llm-models/CodeLlama-34b-hf"

def call_by_automodel(prompt):
    device = "cuda" # for GPU usage or "cpu" for CPU usage
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        load_in_8bit=True
        )

    inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
    attention_mask = torch.ones(inputs.shape, dtype=torch.long, device=device)
    outputs = model.generate(inputs, max_length=200, do_sample=True, num_return_sequences=1, pad_token_id=model.config.eos_token_id, attention_mask=attention_mask)
    
    result = tokenizer.decode(outputs[0], clean_up_tokenization_spaces=False)
    print("-------------------------------------")
    print(result)


def call_by_pipeline(prompt):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    sequences = pipeline(
        prompt,
        do_sample=True,
        top_k=10,
        temperature=0.1,
        top_p=0.95,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_length=200,
    )

    for seq in sequences:
        print(f"Result: {seq['generated_text']}")


def main():
   # prompt = 'import socket\n\ndef ping_exponential_backoff(host: str):'
    prompt = 'public class QuickSort {'
    print("Ask: " + prompt)
    call_by_automodel(prompt)
    # call_by_pipeline(prompt)

if __name__ == "__main__":
    main()