from fastapi import FastAPI, Query
from claude_api import Client
import os

app = FastAPI()
# uvicorn llm.claude-api:app --reload
# http://127.0.0.1:8000/chat/claude?prompt=中国首都在哪

cookie = os.environ.get('CLAUDE_COOKIE')
claude_api = Client(cookie)

conversation_id = claude_api.create_new_chat()['uuid']
print(conversation_id)

def chat(prompt):
    response = claude_api.send_message(prompt, conversation_id)
    print(response)
    return response

@app.get("/chat/claude")
def ask(prompt: str = Query(..., max_length=40)):    
    res = chat(prompt)
    print(res)
    return {'res': res}

