from pydantic import Field, root_validator

from langchain.schema.language_model import BaseLanguageModel
from langchain.utils import (
    get_from_dict_or_env,
)

from typing import Any, AsyncIterator, Dict, Iterator, List, Optional

from langchain.callbacks.manager import (
    CallbackManagerForLLMRun,
)
from langchain.chat_models.base import BaseChatModel
from langchain.schema import (
    ChatGeneration,
    ChatResult,
)
from langchain.schema.messages import (
    AIMessage,
    BaseMessage,
    ChatMessage,
    HumanMessage,
    SystemMessage,
)

from fastapi import FastAPI, Query

# uvicorn llm.claude-api:app --reload
# http://127.0.0.1:8000/chat/claude?prompt=中国首都在哪

app = FastAPI()

class _ClaudeCommon(BaseLanguageModel):
    HUMAN_PROMPT: Optional[str] = ''
    AI_PROMPT: Optional[str] = '简单回答'

    @property
    def lc_secrets(self) -> Dict[str, str]:
        return {"claude_cookie": "CLAUDE_COOKIE"}

    @property
    def lc_serializable(self) -> bool:
        return True

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        values["claude_cookie"] = get_from_dict_or_env(values, "claude_cookie", "CLAUDE_COOKIE")
        values["conversation_id"] = get_from_dict_or_env(values, "conversation_id", "CONVERSATION_ID")

        try:
            from claude_api import Client
            claude_api = Client(values["claude_cookie"])
            values["client"] = claude_api
        except ImportError:
            raise ImportError(
                "Could not import ChatClaude python package. "
            )
        return values
    

class ChatClaude(BaseChatModel, _ClaudeCommon):
    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "claude-chat"

    def _convert_messages_to_text(self, messages: List[BaseMessage]) -> str:
        return "".join(
            self._convert_one_message_to_text(message) for message in messages
        )

    def _convert_messages_to_prompt(self, messages: List[BaseMessage]) -> str:
        messages = messages.copy()  # don't mutate the original list
        if not isinstance(messages[-1], AIMessage):
            messages.append(AIMessage(content=""))
        text = self._convert_messages_to_text(messages)
        return (
            text.rstrip()
        )  # trim off the trailing ' ' that might come from the "Assistant: "

    def _generate(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> ChatResult:
        prompt = self._convert_messages_to_prompt(messages)
        completion = self.client.send_message(prompt, self.conversation_id)
        print(completion)
        message = AIMessage(content=completion)
        return ChatResult(generations=[ChatGeneration(message=message)])
    

@app.get("/chat/claude")
def ask(prompt: str = Query(..., max_length=40)):
    chat = ChatClaude()
    messages = [
        SystemMessage(content="简单回答，使用中文"),
        HumanMessage(content=prompt)
    ]
    res = chat(messages)
    print(res)
    return {'res': res}

