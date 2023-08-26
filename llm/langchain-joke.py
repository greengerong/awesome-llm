import os
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

chat = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.5)
prompt = "Tell me a joke. Less than 100 words. Response in Chinese."
print(prompt)

result = chat.predict_messages([HumanMessage(content=prompt)])
print(result)

