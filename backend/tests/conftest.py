"""Shared test fakes."""

import re

import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel

VOCAB = ("work", "dinner", "sister", "run", "tired", "deadline", "beach", "rain", "exam", "family")


class KeywordEmbeddings(Embeddings):
    """Bag-of-words over a tiny vocabulary. Unlike DeterministicFakeEmbedding (random vectors),
    texts that share words end up close, so search *ranking* can be tested without Ollama."""

    def _embed(self, text: str) -> list[float]:
        words = re.findall(r"[a-z]+", text.lower())
        vec = [float(words.count(w)) for w in VOCAB]
        return vec + [1.0]  # never all-zero: cosine distance is undefined for a zero vector

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


@pytest.fixture
def embeddings() -> Embeddings:
    return KeywordEmbeddings()


class ToolFakeChatModel(FakeMessagesListChatModel):
    """Replies with scripted messages (cycling), which may carry tool_calls. The stock fakes
    raise on bind_tools(); here binding is a no-op, so the script decides what gets "called"."""

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, *args, **kwargs):
        # Copy: add_messages stamps an id on the message in place, and the same object
        # returned twice would then *replace* the earlier reply instead of appending.
        result = super()._generate(*args, **kwargs)
        for g in result.generations:
            g.message = g.message.model_copy()
        return result


@pytest.fixture
def chat_model_cls() -> type[ToolFakeChatModel]:
    return ToolFakeChatModel
