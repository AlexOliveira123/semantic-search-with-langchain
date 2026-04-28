#!/usr/bin/env python3
"""Ingest a PDF file: load → split → embed → store in pgVector."""

import argparse
import sys
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    DATABASE_URL,
    EMBEDDING_MODEL,
    OPENAI_API_KEY,
)


def get_vector_store(*, create_extension: bool = False) -> PGVector:
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, openai_api_key=OPENAI_API_KEY)
    return PGVector(
        embeddings=embeddings,
        collection_name=COLLECTION_NAME,
        connection=DATABASE_URL,
        create_extension=create_extension,
    )


def load_pdf(path: Path) -> list:
    print(f"[ingest] Loading '{path}' …")
    loader = PyPDFLoader(str(path))
    docs = loader.load()
    print(f"[ingest] {len(docs)} page(s) loaded.")
    return docs


def split_documents(docs: list) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        add_start_index=True,
    )
    chunks = splitter.split_documents(docs)
    print(f"[ingest] {len(chunks)} chunk(s) after splitting.")
    return chunks


def ingest(pdf_path: Path) -> None:
    if not pdf_path.exists():
        print(f"[ingest] ERROR: file not found: {pdf_path}", file=sys.stderr)
        sys.exit(1)

    docs = load_pdf(pdf_path)
    chunks = split_documents(docs)

    print("[ingest] Connecting to vector store and creating embeddings …")
    vector_store = get_vector_store(create_extension=True)
    vector_store.add_documents(chunks)
    print(f"[ingest] Done. {len(chunks)} chunk(s) stored in '{COLLECTION_NAME}'.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest a PDF into pgVector.")
    parser.add_argument("pdf", type=Path, help="Path to the PDF file to ingest.")
    args = parser.parse_args()
    ingest(args.pdf)


if __name__ == "__main__":
    main()
