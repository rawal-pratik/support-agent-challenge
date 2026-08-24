from __future__ import annotations

from pathlib import Path

from support_agent.ingestion.chunker import chunk_articles
from support_agent.ingestion.parser import load_articles
from support_agent.retrieval.bm25 import BM25Retriever
from support_agent.retrieval.storage import save_chunks
from support_agent.retrieval.vector import VectorRetriever


class RetrievalIndexer:
    """Build all local retrieval indexes."""

    def __init__(
        self,
        knowledge_base: Path,
        index_directory: Path,
    ) -> None:
        self.knowledge_base = knowledge_base
        self.index_directory = index_directory

    def build(self) -> None:
        print("Loading knowledge base...")

        articles = load_articles(
            self.knowledge_base
        )

        chunks = chunk_articles(articles)

        print(
            f"Loaded {len(articles)} articles "
            f"and {len(chunks)} chunks."
        )

        self.index_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        chunks_path = (
            self.index_directory / "chunks.json"
        )

        bm25_path = (
            self.index_directory / "bm25.pkl"
        )

        faiss_path = (
            self.index_directory / "faiss.index"
        )

        print("Saving chunk metadata...")

        save_chunks(
            chunks,
            chunks_path,
        )

        print("Building BM25 index...")

        bm25 = BM25Retriever.build(chunks)

        bm25.save(bm25_path)

        print("Building embedding index...")

        vector = VectorRetriever.build(chunks)

        vector.save(faiss_path)

        print("\nIndex built successfully.")

        print(f"Articles: {len(articles)}")
        print(f"Chunks: {len(chunks)}")
        print(f"BM25: {bm25_path}")
        print(f"FAISS: {faiss_path}")
        print(f"Chunks: {chunks_path}")