# Semantic Search System

A small, self-contained semantic search system:

- **Vector database:** [ChromaDB](https://www.trychroma.com/) — embedded, on-disk, no server to run
- **Embeddings:** pluggable — choose **local** (no API key), **OpenAI**, or **Voyage AI**

## How it fits together

```
your text  --->  embedder.py   --->  vector_store.py (Chroma)  --->  semantic_search.py
              (text -> vector)      (stores + searches vectors)      (the API you use)
```

`SemanticSearch` is the only class most code needs to touch. It embeds your
text with whichever provider you choose and stores/queries the resulting
vectors in a local Chroma collection on disk.

## Setup

```bash
cd src
pip install chromadb

# Then install whichever embedding provider(s) you want:
pip install sentence-transformers   # local, no API key (default)
pip install openai                  # if using OpenAI embeddings
pip install voyageai                # if using Voyage AI embeddings
```

If using OpenAI or Voyage, set the API key as an environment variable:

```bash
export OPENAI_API_KEY="sk-..."
# or
export VOYAGE_API_KEY="pa-..."
```

## Quick start (Python)

```python
from semantic_search import SemanticSearch

# provider: "local" (default, no key needed), "openai", or "voyage"
search = SemanticSearch(provider="local")

search.add_documents([
    "The Eiffel Tower is in Paris, France.",
    "Golden retrievers are friendly family dogs.",
    "Quantum computers use qubits instead of bits.",
])

results = search.search("landmarks in the French capital", top_k=2)
for r in results:
    print(r["score"], r["text"])
```

Run the built-in demo:

```bash
python demo.py                    # local embeddings
python demo.py --provider openai  # requires OPENAI_API_KEY
python demo.py --provider voyage  # requires VOYAGE_API_KEY
```

## Command-line interface

Ingest text files (splits on blank lines into paragraph-sized chunks) and search them:

```bash
# Ingest a folder of .txt files
python cli.py ingest ./my_docs --provider local

# One-off search
python cli.py search "refund policy for late deliveries" --top-k 5

# Interactive search shell
python cli.py shell
```

All commands share `--db` (storage directory, default `./chroma_db`) and
`--collection` (default `documents`) flags, so you can maintain multiple
separate indexes.

## Switching embedding providers

Everything routes through `embedder.get_embedder(provider, **kwargs)`. To add
a new provider (e.g. Cohere, Google), implement `BaseEmbedder.embed()` in
`embedder.py` and register it in the `get_embedder` factory — nothing else in
the system needs to change.

| Provider | Package | API key | Notes |
|---|---|---|---|
| `local` | `sentence-transformers` | none | Runs `all-MiniLM-L6-v2` on your machine; downloads once (~90MB) |
| `openai` | `openai` | `OPENAI_API_KEY` | Default model: `text-embedding-3-small` |
| `voyage` | `voyageai` | `VOYAGE_API_KEY` | Default model: `voyage-3.5` |

You can pick a different model per provider, e.g.:

```python
SemanticSearch(provider="openai", model="text-embedding-3-large")
SemanticSearch(provider="voyage", model="voyage-3-large")
SemanticSearch(provider="local", model_name="all-mpnet-base-v2")
```

## Notes on scale

ChromaDB's embedded/persistent mode is great up to roughly low millions of
vectors on a single machine. If you outgrow that, the same `embedder.py`
layer drops into a hosted vector DB (Pinecone, Weaviate, Qdrant, pgvector)
with only `vector_store.py` needing a rewrite — the embedding and search API
stay identical.

## Files

```
src/
  embedder.py         # embedding provider interface + implementations
  vector_store.py      # ChromaDB wrapper (add / query / delete / reset)
  semantic_search.py   # top-level SemanticSearch class
  cli.py               # command-line ingest/search/shell
  demo.py              # runnable end-to-end example
requirements.txt
```
