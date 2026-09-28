"""Model factories. The rest of the code only talks to LangChain interfaces
(BaseChatModel / Embeddings), so swapping Ollama for another provider is a one-file change."""

from langchain_ollama import ChatOllama, OllamaEmbeddings

from diary_agent.core.config import settings


def get_chat_model(temperature: float | None = None, max_tokens: int | None = None) -> ChatOllama:
    return ChatOllama(
        model=settings.chat_model,
        base_url=settings.ollama_base_url,
        temperature=settings.temperature if temperature is None else temperature,
        num_predict=max_tokens,
        # Non-reasoning models (e.g. llama3.1) reject the flag, so only send it when it's on.
        reasoning=settings.reasoning or None,
    )


def get_embeddings() -> OllamaEmbeddings:
    return OllamaEmbeddings(model=settings.embed_model, base_url=settings.ollama_base_url)
