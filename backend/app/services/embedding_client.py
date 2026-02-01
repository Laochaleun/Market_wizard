"""Embedding client supporting local sentence-transformers and OpenAI embeddings."""

from typing import Protocol
import logging

import numpy as np

from app.config import get_settings


class EmbeddingClient(Protocol):
    """Protocol for embedding clients."""

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed a list of texts into vectors."""
        ...


class LocalEmbeddingClient:
    """Local embedding client using sentence-transformers (MiniLM by default)."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed texts using local model."""
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        return np.array(embeddings)


class OpenAIEmbeddingClient:
    """OpenAI embedding client using text-embedding-3-small."""

    def __init__(self, api_key: str, model: str = "text-embedding-3-small"):
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key)
        self.model = model

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed texts using OpenAI API."""
        response = self.client.embeddings.create(input=texts, model=self.model)
        embeddings = [item.embedding for item in response.data]
        return np.array(embeddings)


def get_embedding_client() -> EmbeddingClient:
    """Factory function to get the configured embedding client."""
    settings = get_settings()

    if settings.embedding_provider == "local":
        return LocalEmbeddingClient(model_name=settings.embedding_model)
    elif settings.embedding_provider == "openai":
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY required for OpenAI embeddings")
        return OpenAIEmbeddingClient(
            api_key=settings.openai_api_key, model=settings.embedding_model
        )
    else:
        raise ValueError(f"Unknown embedding provider: {settings.embedding_provider}")


def warmup_embeddings() -> None:
    """Ensure the local embedding model is downloaded and ready."""
    settings = get_settings()
    if settings.embedding_provider != "local":
        return

    logger = logging.getLogger(__name__)
    warmup_enabled = str(getattr(settings, "embedding_warmup", True)).lower() not in {
        "0",
        "false",
        "no",
        "off",
    }
    if not warmup_enabled:
        logger.info("Embedding warmup disabled")
        return

    try:
        client = LocalEmbeddingClient(model_name=settings.embedding_model)
        # Trigger model download/caching with a tiny encode.
        client.embed(["warmup"])
        logger.info("Embedding warmup complete | model=%s", settings.embedding_model)
    except Exception as exc:
        logger.warning("Embedding warmup failed | model=%s | error=%s", settings.embedding_model, exc)
