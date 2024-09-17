import os
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()
# Replace the API key with your actual key

os.environ["NVIDIA_API_KEY"] = os.getenv("NVIDIA_API_KEY")
model = ChatNVIDIA(model="meta/llama-3.1-405b-instruct")

from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)
from langchain_core.runnables.history import RunnableWithMessageHistory

store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

def run_agent():
    with_message_history = RunnableWithMessageHistory(model, get_session_history)

    config = {"configurable": {"session_id": "abc5"}}

    while (True):

        response = with_message_history.invoke(
            [HumanMessage(content=input("Me - "))],
            config=config,
        )

        print("AI -"+response.content)