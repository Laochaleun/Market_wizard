"""Embedding client supporting local BGE-M3 and OpenAI embeddings."""

import numpy as np
from typing import Protocol

from app.config import get_settings


class EmbeddingClient(Protocol):
    """Protocol for embedding clients."""

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed a list of texts into vectors."""
        ...


class LocalEmbeddingClient:
    """Local embedding client using sentence-transformers (BGE-M3)."""

    def __init__(self, model_name: str = "BAAI/bge-m3"):
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
