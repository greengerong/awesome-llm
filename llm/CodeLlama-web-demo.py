from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
import gradio as gr
import mdtex2html
import torch

# model_name = "codellama/CodeLlama-7b-Instruct-hf"
model_name = "/mnt/d/llm-models/CodeLlama-7b-hf"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True,)
model = AutoModelForCausalLM.from_pretrained(model_name,  trust_remote_code=True,load_in_8bit=True, device_map="auto")

device = "cuda" # for GPU usage or "cpu" for CPU usage

"""Override Chatbot.postprocess"""
def postprocess(self, y):
    if y is None:
        return []
    for i, (message, response) in enumerate(y):
        y[i] = (
            None if message is None else mdtex2html.convert((message)),
            None if response is None else mdtex2html.convert(response),
        )
    return y


gr.Chatbot.postprocess = postprocess


def parse_text(text):     
    lines = text.replace("<s>", "").split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split('`')
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = f'<br></code></pre>'
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("`", "\`")
                    line = line.replace("<", "&lt;")
                    line = line.replace(">", "&gt;")
                    line = line.replace(" ", "&nbsp;")
                    line = line.replace("*", "&ast;")
                    line = line.replace("_", "&lowbar;")
                    line = line.replace("-", "&#45;")
                    line = line.replace(".", "&#46;")
                    line = line.replace("!", "&#33;")
                    line = line.replace("(", "&#40;")
                    line = line.replace(")", "&#41;")
                    line = line.replace("$", "&#36;")
                lines[i] = "<br>"+line
    text = "".join(lines)
    return text


def predict(input, chatbot, max_length, top_p, temperature, history, past_key_values):
    print("-------------input:\n" + input)
    chatbot.append((parse_text(input), ""))

    inputs = tokenizer.encode(input, return_tensors="pt").to(device)
    attention_mask = torch.ones(inputs.shape, dtype=torch.long, device=device)
    outputs = model.generate(inputs, max_length=max_length, do_sample=True, top_p=top_p, temperature=temperature, num_return_sequences=1, pad_token_id=model.config.eos_token_id, attention_mask=attention_mask)
    result = tokenizer.decode(outputs[0], clean_up_tokenization_spaces=True)
    print("-------------generate:\n" + result)

    # for response, history, past_key_values in model.stream_chat(tokenizer, input, history, past_key_values=past_key_values,
    #                                                                 return_past_key_values=True,
    #                                                                 max_length=max_length, top_p=top_p,
    #                                                                 temperature=temperature):
    chatbot[-1] = (parse_text(input),parse_text( '```\n' + result))
    yield chatbot, history, past_key_values


def reset_user_input():
    return gr.update(value='')


def reset_state():
    return [], [], None


with gr.Blocks() as demo:
    gr.HTML("""<h1 align="center">codellama/CodeLlama-7b-Instruct-hf</h1>""")

    chatbot = gr.Chatbot()
    with gr.Row():
        with gr.Column(scale=4):
            with gr.Column(scale=12):
                user_input = gr.Textbox(show_label=False, placeholder="Input...", value='tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True,)', lines=10).style(container=False)
            with gr.Column(min_width=32, scale=1):
                submitBtn = gr.Button("Submit", variant="primary")
        with gr.Column(scale=1):
            emptyBtn = gr.Button("Clear History")
            max_length = gr.Slider(0, 32768, value=200, step=1.0, label="Maximum length", interactive=True)
            top_p = gr.Slider(0, 1, value=0.8, step=0.01, label="Top P", interactive=True)
            temperature = gr.Slider(0, 1, value=0.1, step=0.01, label="Temperature", interactive=True)

    history = gr.State([])
    past_key_values = gr.State(None)

    submitBtn.click(predict, [user_input, chatbot, max_length, top_p, temperature, history, past_key_values],
                    [chatbot, history, past_key_values], show_progress=True)
    submitBtn.click(reset_user_input, [], [user_input])

    emptyBtn.click(reset_state, outputs=[chatbot, history, past_key_values], show_progress=True)

demo.queue().launch(share=False, inbrowser=True, debug=True)