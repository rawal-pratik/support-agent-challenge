from __future__ import annotations

from dataclasses import dataclass

from support_agent.llm.openrouter import OpenRouterClient
from support_agent.llm.understanding import understand_ticket
from support_agent.models.schemas import (
    RetrievedEvidence,
    TicketUnderstanding,
)
from support_agent.retrieval.evidence import EvidenceSelector
from support_agent.retrieval.hybrid import (
    HybridRetriever,
    RetrievalResult,
)


@dataclass(frozen=True)
class QueryAwareRetrieval:
    """Results from query-aware retrieval."""

    understanding: TicketUnderstanding
    evidence: list[RetrievedEvidence]


class QueryAwareRetriever:
    """
    Retrieve knowledge-base evidence using multiple
    LLM-generated search queries.

    Each query runs independently through the existing
    hybrid retriever. Results are then fused using the
    ranks produced by those independent searches.

    Clearly unrelated tickets do not trigger knowledge-base
    retrieval.
    """

    RRF_K = 60

    def __init__(
        self,
        retriever: HybridRetriever,
        llm_client: OpenRouterClient,
        evidence_selector: EvidenceSelector | None = None,
    ) -> None:
        self.retriever = retriever
        self.llm_client = llm_client
        self.evidence_selector = (
            evidence_selector
            or EvidenceSelector()
        )

    @classmethod
    def _fuse_results(
        cls,
        query_results: dict[
            str,
            list[RetrievalResult],
        ],
    ) -> list[RetrievedEvidence]:
        """
        Fuse results from multiple queries.

        A document appearing highly across several queries
        receives a stronger score than a document appearing
        for only one query.
        """

        documents: dict[
            str,
            dict,
        ] = {}

        for query, results in query_results.items():
            for rank, result in enumerate(
                results,
                start=1,
            ):
                chunk = result.chunk
                chunk_id = chunk.chunk_id

                if chunk_id not in documents:
                    documents[chunk_id] = {
                        "result": result,
                        "queries": [],
                        "score": 0.0,
                    }

                entry = documents[chunk_id]

                entry["score"] += 1.0 / (
                    cls.RRF_K + rank
                )

                entry["queries"].append(query)

                if (
                    result.hybrid_score
                    > entry["result"].hybrid_score
                ):
                    entry["result"] = result

        ranked = sorted(
            documents.values(),
            key=lambda entry: entry["score"],
            reverse=True,
        )

        evidence: list[RetrievedEvidence] = []

        for entry in ranked:
            result = entry["result"]
            chunk = result.chunk

            evidence.append(
                RetrievedEvidence(
                    chunk_id=chunk.chunk_id,
                    article_id=chunk.article_id,
                    title=chunk.title,
                    breadcrumbs=list(chunk.breadcrumbs),
                    source_path=chunk.source_path,
                    source_url=chunk.source_url,
                    section=chunk.section,
                    section_path=list(chunk.section_path),
                    text=chunk.text,
                    hybrid_score=entry["score"],
                    bm25_score=result.bm25_score,
                    vector_score=result.vector_score,
                    bm25_rank=result.bm25_rank,
                    vector_rank=result.vector_rank,
                    matched_queries=entry["queries"],
                )
            )

        return evidence

    def search(
        self,
        ticket: str,
        *,
        top_k: int = 5,
        candidate_k: int = 20,
    ) -> QueryAwareRetrieval:
        """
        Understand the ticket and search the KB using the
        original ticket plus every generated search query.

        Clearly unrelated tickets return no retrieval evidence.
        """

        understanding = understand_ticket(
            ticket=ticket,
            client=self.llm_client,
        )

        if not understanding.is_hackerrank_related:
            return QueryAwareRetrieval(
                understanding=understanding,
                evidence=[],
            )

        queries: list[str] = []

        for query in [
            ticket,
            *understanding.search_queries,
        ]:
            normalized = query.strip()

            if (
                normalized
                and normalized not in queries
            ):
                queries.append(normalized)

        query_results: dict[
            str,
            list[RetrievalResult],
        ] = {}

        for query in queries:
            query_results[query] = self.retriever.search(
                query=query,
                top_k=candidate_k,
                candidate_k=candidate_k,
            )

        fused_evidence = self._fuse_results(
            query_results
        )

        selected_evidence = self.evidence_selector.select(
            fused_evidence[:top_k]
        )

        return QueryAwareRetrieval(
            understanding=understanding,
            evidence=selected_evidence,
        )