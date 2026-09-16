"""
Vector storage layer, backed by ChromaDB (embedded, on-disk, no server
process to run). All embedding is done ourselves via the embedder module
and passed in explicitly, so Chroma never tries to download its own
default embedding model.
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

import chromadb


class VectorStore:
    def __init__(self, persist_dir: str = "./chroma_db", collection_name: str = "documents"):
        self.client = chromadb.PersistentClient(path=persist_dir)
        # embedding_function=None: we always supply our own precomputed vectors.
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},  # cosine similarity for search
        )

    def add(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """Add documents + their precomputed embeddings to the store."""
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]
        if metadatas is None:
            metadatas = [None for _ in texts]
        # Newer Chroma versions reject empty {} metadata dicts outright, so
        # give every document at least one real key instead of leaving it blank.
        metadatas = [m if m else {"source": "unknown"} for m in metadatas]

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )
        return ids

    def query(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Return top_k most similar documents to the query embedding."""
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
        )

        hits = []
        ids = results["ids"][0]
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        dists = results["distances"][0]

        for i in range(len(ids)):
            # Chroma returns cosine *distance*; convert to a 0-1 similarity score.
            similarity = 1 - dists[i]
            hits.append(
                {
                    "id": ids[i],
                    "text": docs[i],
                    "metadata": metas[i],
                    "score": round(similarity, 4),
                }
            )
        return hits

    def count(self) -> int:
        return self.collection.count()

    def delete(self, ids: List[str]) -> None:
        self.collection.delete(ids=ids)

    def reset(self) -> None:
        """Delete all documents in the collection."""
        name = self.collection.name
        self.client.delete_collection(name)
        self.collection = self.client.get_or_create_collection(
            name=name, metadata={"hnsw:space": "cosine"}
        )
