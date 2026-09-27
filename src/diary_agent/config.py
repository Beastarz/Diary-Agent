"""Central configuration. Everything is overridable via environment variables or a .env file."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    # llama3.1 by default: qwen3:8b leaked Chinese tokens into ~6-9% of English replies (see NOTES.md M1).
    chat_model: str = os.getenv("DIARY_CHAT_MODEL", "llama3.1:8b")
    embed_model: str = os.getenv("DIARY_EMBED_MODEL", "nomic-embed-text")
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    temperature: float = float(os.getenv("DIARY_TEMPERATURE", "0.7"))
    # Only used by reasoning models like qwen3. Thinking makes replies slower; off by default for chat.
    reasoning: bool = os.getenv("DIARY_REASONING", "false").lower() == "true"
    data_dir: Path = Path(os.getenv("DIARY_DATA_DIR", PROJECT_ROOT / "data"))


settings = Settings()
