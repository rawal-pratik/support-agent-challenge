from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class Settings:
    """Application configuration."""

    openrouter_api_key: str | None
    openrouter_model: str

    embedding_model: str
    embedding_batch_size: int

    @classmethod
    def from_environment(cls) -> "Settings":
        return cls(
            openrouter_api_key=os.getenv("OPENROUTER_API_KEY"),
            openrouter_model=os.getenv(
                "OPENROUTER_MODEL",
                "qwen/qwen3-8b:free",
            ),
            embedding_model=os.getenv(
                "EMBEDDING_MODEL",
                "BAAI/bge-small-en-v1.5",
            ),
            embedding_batch_size=int(
                os.getenv("EMBEDDING_BATCH_SIZE", "32")
            ),
        )


settings = Settings.from_environment()