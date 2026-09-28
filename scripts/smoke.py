"""M0 smoke test: can we talk to Ollama? Run: uv run python scripts/smoke.py"""

from langchain_core.messages import HumanMessage, SystemMessage

from diary_agent.core.llm import get_chat_model, get_embeddings

model = get_chat_model()
messages = [SystemMessage("Reply in one short sentence."), HumanMessage("Say hi to a new journaler.")]
for chunk in model.stream(messages):
    print(chunk.content, end="", flush=True)
print()

vec = get_embeddings().embed_query("I went hiking today.")
print(f"embedding dim: {len(vec)}")
