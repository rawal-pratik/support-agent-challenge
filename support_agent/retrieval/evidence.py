from __future__ import annotations

from support_agent.models.schemas import RetrievedEvidence


class EvidenceSelector:
    """
    Select and bound retrieved evidence before it is passed
    to downstream components.

    This layer intentionally does not attempt to determine
    semantic relevance independently. Retrieval has already
    performed that work.

    Its responsibilities are limited to:
    - removing exact duplicate chunks
    - preserving retrieval order
    - bounding the amount of evidence passed downstream
    """

    def __init__(
        self,
        *,
        max_evidence: int = 8,
    ) -> None:
        if max_evidence < 1:
            raise ValueError(
                "max_evidence must be at least 1."
            )

        self.max_evidence = max_evidence

    def select(
        self,
        evidence: list[RetrievedEvidence],
    ) -> list[RetrievedEvidence]:
        """
        Select evidence while preserving the ranking produced
        by query-aware retrieval.
        """

        selected: list[RetrievedEvidence] = []
        seen_chunk_ids: set[str] = set()

        for item in evidence:
            if item.chunk_id in seen_chunk_ids:
                continue

            seen_chunk_ids.add(item.chunk_id)
            selected.append(item)

            if len(selected) >= self.max_evidence:
                break

        return selected