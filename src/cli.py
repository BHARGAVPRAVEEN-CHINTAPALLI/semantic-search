"""
Command-line interface.

Usage:
    # Ingest one or more .txt files (splits each into paragraphs)
    python cli.py ingest my_notes.txt other_doc.txt --provider local

    # Ingest a whole folder of .txt files
    python cli.py ingest ./docs --provider local

    # Search
    python cli.py search "how does the refund process work" --provider local --top-k 5

    # Interactive search shell
    python cli.py shell --provider local
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

from semantic_search import SemanticSearch


def load_text_chunks(path: str) -> list[str]:
    """Read a .txt file and split it into non-trivial paragraphs."""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
    chunks = [p.strip() for p in content.split("\n\n") if p.strip() and len(p.strip()) > 20]
    return chunks if chunks else [content.strip()]


def collect_files(paths: list[str]) -> list[str]:
    files = []
    for p in paths:
        if os.path.isdir(p):
            files.extend(sorted(glob.glob(os.path.join(p, "**/*.txt"), recursive=True)))
        elif os.path.isfile(p):
            files.append(p)
        else:
            print(f"Warning: '{p}' not found, skipping.", file=sys.stderr)
    return files


def cmd_ingest(args):
    search = SemanticSearch(
        provider=args.provider, persist_dir=args.db, collection_name=args.collection
    )
    files = collect_files(args.paths)
    if not files:
        print("No files found to ingest.")
        return

    total = 0
    for path in files:
        chunks = load_text_chunks(path)
        metadatas = [{"source": path, "chunk": i} for i in range(len(chunks))]
        search.add_documents(chunks, metadatas=metadatas)
        total += len(chunks)
        print(f"  ingested {len(chunks):>4} chunk(s) from {path}")

    print(f"\nDone. {total} chunks added. Collection now has {search.count()} documents total.")


def cmd_search(args):
    search = SemanticSearch(
        provider=args.provider, persist_dir=args.db, collection_name=args.collection
    )
    results = search.search(args.query, top_k=args.top_k)
    _print_results(results)


def cmd_shell(args):
    search = SemanticSearch(
        provider=args.provider, persist_dir=args.db, collection_name=args.collection
    )
    print(f"Semantic search shell ({search.count()} documents loaded). Ctrl+C to quit.\n")
    try:
        while True:
            query = input("query> ").strip()
            if not query:
                continue
            results = search.search(query, top_k=args.top_k)
            _print_results(results)
    except (KeyboardInterrupt, EOFError):
        print("\nBye.")


def _print_results(results):
    if not results:
        print("No results.")
        return
    for rank, r in enumerate(results, 1):
        source = r["metadata"].get("source", "")
        print(f"\n[{rank}] score={r['score']}  {source}")
        text = r["text"]
        print(f"    {text[:300]}{'...' if len(text) > 300 else ''}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Semantic search over your documents.")
    parser.add_argument(
        "--provider", default="local", choices=["local", "openai", "voyage"],
        help="Embedding provider (default: local, no API key needed)",
    )
    parser.add_argument("--db", default="./chroma_db", help="Vector DB storage directory")
    parser.add_argument("--collection", default="documents", help="Collection name")

    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="Ingest text file(s) or a folder of .txt files")
    p_ingest.add_argument("paths", nargs="+", help="File(s) or folder(s) to ingest")
    p_ingest.set_defaults(func=cmd_ingest)

    p_search = sub.add_parser("search", help="Run a single search query")
    p_search.add_argument("query", help="Search query text")
    p_search.add_argument("--top-k", type=int, default=5)
    p_search.set_defaults(func=cmd_search)

    p_shell = sub.add_parser("shell", help="Interactive search shell")
    p_shell.add_argument("--top-k", type=int, default=5)
    p_shell.set_defaults(func=cmd_shell)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
