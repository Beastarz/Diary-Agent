"""M1: a plain LangChain (LCEL) chat chain.

prompt | model is a Runnable. The model is stateless, so *we* keep the history
and pass it in on every call via the MessagesPlaceholder.
"""

from datetime import date

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable

from diary_agent.prompts import COMPANION_SYSTEM


def build_chain(model: BaseChatModel) -> Runnable:
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", COMPANION_SYSTEM),
            MessagesPlaceholder("history"),
            ("human", "{input}"),
        ]
    )
    return prompt | model | StrOutputParser()


class ChatSession:
    """Holds in-memory history for one conversation. Lost when the process exits (fixed in M3)."""

    def __init__(self, model: BaseChatModel):
        self.chain = build_chain(model)
        self.history: list[BaseMessage] = []

    def stream(self, user_input: str):
        """Yield reply chunks, then record the full turn in history."""
        inputs = {"today": date.today().isoformat(), "history": self.history, "input": user_input}
        reply = ""
        for chunk in self.chain.stream(inputs):
            reply += chunk
            yield chunk
        self.history += [HumanMessage(user_input), AIMessage(reply)]
