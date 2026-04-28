#!/usr/bin/env python3
"""Semantic search CLI — returns the top-K most relevant chunks for a query."""

import argparse

from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector

from config import (
    COLLECTION_NAME,
    DATABASE_URL,
    EMBEDDING_MODEL,
    OPENAI_API_KEY,
    SEARCH_TOP_K,
)


def get_vector_store() -> PGVector:
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, openai_api_key=OPENAI_API_KEY)
    return PGVector(
        embeddings=embeddings,
        collection_name=COLLECTION_NAME,
        connection=DATABASE_URL,
    )


def semantic_search(query: str, top_k: int = SEARCH_TOP_K) -> list:
    vector_store = get_vector_store()
    results = vector_store.similarity_search_with_relevance_scores(query, k=top_k)
    return results


def display_results(results: list) -> None:
    if not results:
        print("No results found.")
        return

    for rank, (doc, score) in enumerate(results, start=1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        print(f"\n{'─' * 60}")
        print(f"  Rank {rank}  |  Score: {score:.4f}  |  {source}  (page {page})")
        print(f"{'─' * 60}")
        print(doc.page_content.strip())

    print(f"\n{'─' * 60}")
    print(f"  {len(results)} result(s) returned.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic search over ingested documents.")
    parser.add_argument("query", type=str, help="Search query.")
    parser.add_argument(
        "--top-k",
        type=int,
        default=SEARCH_TOP_K,
        help=f"Number of results to return (default: {SEARCH_TOP_K}).",
    )
    args = parser.parse_args()

    print(f"\n[search] Query: \"{args.query}\"")
    results = semantic_search(args.query, top_k=args.top_k)
    display_results(results)


if __name__ == "__main__":
    main()
