from __future__ import annotations

from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from support_agent.config import settings
from support_agent.ingestion.chunker import KnowledgeChunk


class VectorRetriever:
    """Dense vector retrieval using Sentence Transformers + FAISS."""

    def __init__(
        self,
        chunks: list[KnowledgeChunk],
        index: faiss.Index,
        model: SentenceTransformer,
    ) -> None:
        self.chunks = chunks
        self.index = index
        self.model = model

    @staticmethod
    def _document_text(chunk: KnowledgeChunk) -> str:
        """Build the representation embedded for a KB chunk."""
        return "\n".join(
            [
                chunk.title,
                " > ".join(chunk.breadcrumbs),
                " > ".join(chunk.section_path),
                chunk.text,
            ]
        )

    @classmethod
    def build(
        cls,
        chunks: list[KnowledgeChunk],
    ) -> "VectorRetriever":
        model = SentenceTransformer(
            settings.embedding_model
        )

        documents = [
            cls._document_text(chunk)
            for chunk in chunks
        ]

        embeddings = model.encode(
            documents,
            batch_size=settings.embedding_batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        embeddings = np.asarray(
            embeddings,
            dtype=np.float32,
        )

        dimension = embeddings.shape[1]

        index = faiss.IndexFlatIP(dimension)

        index.add(embeddings)

        return cls(
            chunks=chunks,
            index=index,
            model=model,
        )

    def search(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[tuple[int, float]]:
        """Return chunk indexes and cosine similarity scores."""
        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        query_embedding = np.asarray(
            query_embedding,
            dtype=np.float32,
        )

        scores, indexes = self.index.search(
            query_embedding,
            top_k,
        )

        results: list[tuple[int, float]] = []

        for index, score in zip(
            indexes[0],
            scores[0],
        ):
            if index < 0:
                continue

            results.append(
                (
                    int(index),
                    float(score),
                )
            )

        return results

    def save(self, path: Path) -> None:
        """Persist FAISS index."""
        path.parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(
            self.index,
            str(path),
        )

    @classmethod
    def load(
        cls,
        chunks: list[KnowledgeChunk],
        path: Path,
    ) -> "VectorRetriever":
        """Load a persisted FAISS index."""
        index = faiss.read_index(str(path))

        model = SentenceTransformer(
            settings.embedding_model
        )

        return cls(
            chunks=chunks,
            index=index,
            model=model,
        )