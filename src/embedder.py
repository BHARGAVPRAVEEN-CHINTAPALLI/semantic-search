"""
Pluggable embedding providers.

Every embedder implements the same tiny interface:
    embed(texts: list[str]) -> list[list[float]]

Swap providers by changing one line in config.py / main.py — nothing
else in the system needs to know which one is active.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import List


class BaseEmbedder(ABC):
    """Common interface for all embedding providers."""

    #: Dimensionality of vectors this embedder produces. Set by subclasses.
    dimension: int

    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of texts. Must return one vector per input text, in order."""
        raise NotImplementedError

    def embed_one(self, text: str) -> List[float]:
        return self.embed([text])[0]


class OpenAIEmbedder(BaseEmbedder):
    """
    Uses OpenAI's embeddings API (text-embedding-3-small by default).

    Requires: pip install openai
    Requires: OPENAI_API_KEY environment variable.
    """

    DIMENSIONS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(self, model: str = "text-embedding-3-small", api_key: str | None = None):
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ImportError(
                "OpenAIEmbedder requires the 'openai' package. Install with: pip install openai"
            ) from e

        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "No OpenAI API key found. Set OPENAI_API_KEY env var or pass api_key=..."
            )

        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.dimension = self.DIMENSIONS.get(model, 1536)

    def embed(self, texts: List[str]) -> List[List[float]]:
        # OpenAI's API accepts batches directly; chunk defensively for very large inputs.
        out: List[List[float]] = []
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]
            resp = self.client.embeddings.create(model=self.model, input=chunk)
            out.extend([item.embedding for item in resp.data])
        return out


class VoyageEmbedder(BaseEmbedder):
    """
    Uses Voyage AI's embeddings API (voyage-3.5 by default).
    Anthropic recommends Voyage AI for embeddings since Claude models
    themselves don't expose an embeddings endpoint.

    Requires: pip install voyageai
    Requires: VOYAGE_API_KEY environment variable.
    """

    DIMENSIONS = {
        "voyage-3.5": 1024,
        "voyage-3.5-lite": 1024,
        "voyage-3-large": 1024,
        "voyage-code-3": 1024,
    }

    def __init__(self, model: str = "voyage-3.5", api_key: str | None = None):
        try:
            import voyageai
        except ImportError as e:
            raise ImportError(
                "VoyageEmbedder requires the 'voyageai' package. Install with: pip install voyageai"
            ) from e

        api_key = api_key or os.environ.get("VOYAGE_API_KEY")
        if not api_key:
            raise ValueError(
                "No Voyage API key found. Set VOYAGE_API_KEY env var or pass api_key=..."
            )

        self.client = voyageai.Client(api_key=api_key)
        self.model = model
        self.dimension = self.DIMENSIONS.get(model, 1024)

    def embed(self, texts: List[str]) -> List[List[float]]:
        out: List[List[float]] = []
        batch_size = 128
        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]
            resp = self.client.embed(chunk, model=self.model, input_type="document")
            out.extend(resp.embeddings)
        return out


class LocalEmbedder(BaseEmbedder):
    """
    Runs a local sentence-transformers model — no API key, no network
    calls at query time (model weights are downloaded once on first use).

    Requires: pip install sentence-transformers
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "LocalEmbedder requires 'sentence-transformers'. "
                "Install with: pip install sentence-transformers"
            ) from e

        self.model = SentenceTransformer(model_name)
        # Newer sentence-transformers renamed this method; support both.
        if hasattr(self.model, "get_embedding_dimension"):
            self.dimension = self.model.get_embedding_dimension()
        else:
            self.dimension = self.model.get_sentence_embedding_dimension()

    def embed(self, texts: List[str]) -> List[List[float]]:
        vectors = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return vectors.tolist()


def get_embedder(provider: str, **kwargs) -> BaseEmbedder:
    """
    Factory function. provider is one of: "openai", "voyage", "local".
    Extra kwargs are forwarded to the chosen embedder's constructor.
    """
    provider = provider.lower()
    if provider == "openai":
        return OpenAIEmbedder(**kwargs)
    if provider == "voyage":
        return VoyageEmbedder(**kwargs)
    if provider == "local":
        return LocalEmbedder(**kwargs)
    raise ValueError(f"Unknown provider '{provider}'. Choose: openai, voyage, local.")
