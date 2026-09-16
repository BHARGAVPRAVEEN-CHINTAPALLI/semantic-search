"""
Quick end-to-end demo. Run with:

    python demo.py                 # uses local embeddings (no API key)
    python demo.py --provider openai
    python demo.py --provider voyage
"""

import argparse

from semantic_search import SemanticSearch

DOCS = [
    "The Eiffel Tower is a wrought-iron lattice tower in Paris, France.",
    "Photosynthesis is the process plants use to convert sunlight into energy.",
    "The stock market experienced significant volatility during the recession.",
    "Golden retrievers are known for being friendly and great with children.",
    "Quantum computers use qubits instead of classical bits to perform calculations.",
    "Paris is famous for its art museums, cafes, and historic architecture.",
    "A balanced diet and regular exercise are key to maintaining good health.",
    "Machine learning models improve their accuracy by training on large datasets.",
]

QUERIES = [
    "landmarks in the French capital",
    "how do plants make food",
    "friendly family dog breeds",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", default="local", choices=["local", "openai", "voyage"])
    args = parser.parse_args()

    print(f"Building semantic search index with provider='{args.provider}' ...")
    search = SemanticSearch(provider=args.provider, persist_dir="./demo_chroma_db")
    search.reset()
    search.add_documents(DOCS)
    print(f"Indexed {search.count()} documents.\n")

    for q in QUERIES:
        print(f"Query: {q!r}")
        for r in search.search(q, top_k=2):
            print(f"    score={r['score']:.4f}  {r['text']}")
        print()


if __name__ == "__main__":
    main()
