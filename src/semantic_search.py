"""
Top-level SemanticSearch API: the class most users will actually import.

    from semantic_search import SemanticSearch

    search = SemanticSearch(provider="local")          # or "openai" / "voyage"
    search.add_documents(["The cat sat on the mat.", "Dogs are loyal pets."])
    results = search.search("a feline on a rug", top_k=3)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from embedder import get_embedder
from vector_store import VectorStore


class SemanticSearch:
    def __init__(
        self,
        provider: str = "local",
        persist_dir: str = "./chroma_db",
        collection_name: str = "documents",
        **embedder_kwargs,
    ):
        """
        provider: "local" (sentence-transformers, no API key),
                  "openai" (needs OPENAI_API_KEY),
                  "voyage" (needs VOYAGE_API_KEY)
        embedder_kwargs: forwarded to the embedder constructor, e.g. model="voyage-3.5"
        """
        self.embedder = get_embedder(provider, **embedder_kwargs)
        self.store = VectorStore(persist_dir=persist_dir, collection_name=collection_name)

    def add_documents(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """Embed and store a batch of documents. Returns their IDs."""
        if not texts:
            return []
        vectors = self.embedder.embed(texts)
        return self.store.add(texts=texts, embeddings=vectors, metadatas=metadatas, ids=ids)

    def search(
        self, query: str, top_k: int = 5, where: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Return the top_k documents most semantically similar to the query."""
        query_vector = self.embedder.embed_one(query)
        return self.store.query(query_vector, top_k=top_k, where=where)

    def count(self) -> int:
        return self.store.count()

    def reset(self) -> None:
        self.store.reset()
