from __future__ import annotations

from dataclasses import dataclass

from support_agent.ingestion.chunker import KnowledgeChunk
from support_agent.retrieval.bm25 import BM25Retriever
from support_agent.retrieval.vector import VectorRetriever


@dataclass(frozen=True)
class RetrievalResult:
    """A retrieved knowledge-base chunk."""

    chunk: KnowledgeChunk

    hybrid_score: float

    bm25_score: float | None
    vector_score: float | None

    bm25_rank: int | None
    vector_rank: int | None


class HybridRetriever:
    """
    Combine BM25 and dense retrieval using Reciprocal Rank Fusion.
    """

    def __init__(
        self,
        chunks: list[KnowledgeChunk],
        bm25: BM25Retriever,
        vector: VectorRetriever,
    ) -> None:
        self.chunks = chunks
        self.bm25 = bm25
        self.vector = vector

    def search(
        self,
        query: str,
        top_k: int = 5,
        candidate_k: int = 20,
        rrf_k: int = 60,
    ) -> list[RetrievalResult]:
        bm25_results = self.bm25.search(
            query,
            top_k=candidate_k,
        )

        vector_results = self.vector.search(
            query,
            top_k=candidate_k,
        )

        bm25_by_index = {
            index: (rank, score)
            for rank, (index, score)
            in enumerate(bm25_results, start=1)
        }

        vector_by_index = {
            index: (rank, score)
            for rank, (index, score)
            in enumerate(vector_results, start=1)
        }

        candidate_indexes = (
            set(bm25_by_index)
            | set(vector_by_index)
        )

        results: list[RetrievalResult] = []

        for index in candidate_indexes:
            bm25_entry = bm25_by_index.get(index)
            vector_entry = vector_by_index.get(index)

            bm25_rank = (
                bm25_entry[0]
                if bm25_entry
                else None
            )

            vector_rank = (
                vector_entry[0]
                if vector_entry
                else None
            )

            bm25_score = (
                bm25_entry[1]
                if bm25_entry
                else None
            )

            vector_score = (
                vector_entry[1]
                if vector_entry
                else None
            )

            hybrid_score = 0.0

            if bm25_rank is not None:
                hybrid_score += 1.0 / (
                    rrf_k + bm25_rank
                )

            if vector_rank is not None:
                hybrid_score += 1.0 / (
                    rrf_k + vector_rank
                )

            results.append(
                RetrievalResult(
                    chunk=self.chunks[index],
                    hybrid_score=hybrid_score,
                    bm25_score=bm25_score,
                    vector_score=vector_score,
                    bm25_rank=bm25_rank,
                    vector_rank=vector_rank,
                )
            )

        results.sort(
            key=lambda result: result.hybrid_score,
            reverse=True,
        )

        return results[:top_k]