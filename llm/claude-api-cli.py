from claude_api import Client
import os

cookie = os.environ.get('CLAUDE_COOKIE')
claude_api = Client(cookie)

conversation_id = claude_api.create_new_chat()['uuid']
print(conversation_id)


def chat(prompt):
    response = claude_api.send_message(prompt, conversation_id)
    print(response)
    return response
    
if __name__ == '__main__':
    chat('帮我根据颜色  # 389e0d 生成一组颜色，只返回 json数据格式')