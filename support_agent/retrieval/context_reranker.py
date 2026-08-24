from __future__ import annotations

import re
from dataclasses import dataclass

from support_agent.models.schemas import (
    RetrievedEvidence,
    TicketUnderstanding,
)


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set[str]:
    """Return normalized lexical tokens."""
    return set(
        _TOKEN_RE.findall(
            text.lower()
        )
    )


def _text(
    evidence: RetrievedEvidence,
) -> str:
    """Build the searchable metadata/text representation."""
    return " ".join(
        [
            evidence.title,
            evidence.section,
            " ".join(evidence.breadcrumbs),
            " ".join(evidence.section_path),
            evidence.text,
        ]
    ).lower()


def _overlap_score(
    query_tokens: set[str],
    document_tokens: set[str],
) -> float:
    """
    Calculate lexical overlap.

    The score is intentionally normalized by the query size so
    that a document matching most of the important ticket terms
    receives a useful boost.
    """
    if not query_tokens:
        return 0.0

    return len(
        query_tokens & document_tokens
    ) / len(query_tokens)


@dataclass(frozen=True)
class ContextScore:
    """Breakdown of contextual reranking."""

    relevance: float
    requester: float
    product_area: float
    audience_penalty: float

    @property
    def total(self) -> float:
        return (
            self.relevance
            + self.requester
            + self.product_area
            - self.audience_penalty
        )


class ContextReranker:
    """
    Deterministic second-stage reranker.

    The first-stage retriever determines semantic/lexical
    relevance. This component adds ticket-specific context,
    particularly requester/audience compatibility.
    """

    # These are deliberately conservative. The reranker should
    # adjust ranking, not completely override retrieval.
    RELEVANCE_WEIGHT = 4.0
    REQUESTER_WEIGHT = 2.5
    PRODUCT_AREA_WEIGHT = 1.5

    CANDIDATE_WRONG_AUDIENCE_PENALTY = 3.0
    ADMIN_WRONG_AUDIENCE_PENALTY = 2.5
    COMMUNITY_WRONG_AUDIENCE_PENALTY = 2.5

    def _requester_score(
        self,
        requester_type: str,
        document: str,
    ) -> tuple[float, float]:
        """
        Return (positive compatibility, wrong-audience penalty).
        """

        if requester_type == "candidate":
            candidate_markers = {
                "candidate",
                "candidates",
                "candidate experience",
                "test taker",
                "test takers",
                "applicant",
                "assessment",
            }

            admin_markers = {
                "recruiter",
                "recruiters",
                "administrator",
                "administrators",
                "admin",
                "workspace",
                "test owner",
                "team member",
                "editor access",
            }

            integration_markers = {
                "integration",
                "integrations",
                "api",
                "api token",
                "webhook",
                "ats",
                "applicant tracking system",
            }

            positive = 0.0
            penalty = 0.0

            if any(
                marker in document
                for marker in candidate_markers
            ):
                positive += 1.0

            if any(
                marker in document
                for marker in admin_markers
            ):
                penalty += 1.5

            if any(
                marker in document
                for marker in integration_markers
            ):
                penalty += 2.0

            return positive, penalty

        if requester_type == "customer_admin":
            admin_markers = {
                "recruiter",
                "recruiters",
                "administrator",
                "administrators",
                "admin",
                "workspace",
                "test owner",
                "team member",
                "editor",
                "editor access",
            }

            candidate_only_markers = {
                "candidate experience",
                "candidate support",
                "test taker",
                "test takers",
            }

            positive = 0.0
            penalty = 0.0

            if any(
                marker in document
                for marker in admin_markers
            ):
                positive += 1.0

            if any(
                marker in document
                for marker in candidate_only_markers
            ):
                penalty += 1.0

            return positive, penalty

        if requester_type == "community_user":
            community_markers = {
                "community",
                "community user",
                "discussion",
                "forum",
            }

            positive = 0.0

            if any(
                marker in document
                for marker in community_markers
            ):
                positive += 1.0

            return positive, 0.0

        # Unknown requester: don't make assumptions.
        return 0.0, 0.0

    def _product_area_score(
        self,
        product_area_hint: str,
        document: str,
    ) -> float:
        """
        Measure lexical compatibility between the product-area
        hypothesis and retrieved document metadata/content.
        """

        hint_tokens = _tokens(
            product_area_hint
        )

        document_tokens = _tokens(
            document
        )

        overlap = _overlap_score(
            hint_tokens,
            document_tokens,
        )

        return overlap

    def _relevance_score(
        self,
        understanding: TicketUnderstanding,
        evidence: RetrievedEvidence,
    ) -> float:
        """
        Measure lexical compatibility between the actual user
        intent/entities and the retrieved document.
        """

        document = _text(evidence)

        intent_tokens = _tokens(
            understanding.intent
        )

        entity_tokens = _tokens(
            " ".join(
                understanding.entities
            )
        )

        intent_overlap = _overlap_score(
            intent_tokens,
            _tokens(document),
        )

        entity_overlap = _overlap_score(
            entity_tokens,
            _tokens(document),
        )

        # Intent is more important than individual entities.
        return (
            0.65 * intent_overlap
            + 0.35 * entity_overlap
        )

    def score(
        self,
        understanding: TicketUnderstanding,
        evidence: RetrievedEvidence,
    ) -> ContextScore:
        document = _text(evidence)

        relevance = (
            self._relevance_score(
                understanding,
                evidence,
            )
            * self.RELEVANCE_WEIGHT
        )

        requester_positive, requester_penalty = (
            self._requester_score(
                understanding.requester_type,
                document,
            )
        )

        requester = (
            requester_positive
            * self.REQUESTER_WEIGHT
        )

        product_area = (
            self._product_area_score(
                understanding.product_area_hint,
                document,
            )
            * self.PRODUCT_AREA_WEIGHT
        )

        if (
            understanding.requester_type
            == "candidate"
        ):
            audience_penalty = (
                requester_penalty
                * self.CANDIDATE_WRONG_AUDIENCE_PENALTY
            )

        elif (
            understanding.requester_type
            == "customer_admin"
        ):
            audience_penalty = (
                requester_penalty
                * self.ADMIN_WRONG_AUDIENCE_PENALTY
            )

        elif (
            understanding.requester_type
            == "community_user"
        ):
            audience_penalty = (
                requester_penalty
                * self.COMMUNITY_WRONG_AUDIENCE_PENALTY
            )

        else:
            audience_penalty = 0.0

        return ContextScore(
            relevance=relevance,
            requester=requester,
            product_area=product_area,
            audience_penalty=audience_penalty,
        )

    def rerank(
        self,
        understanding: TicketUnderstanding,
        evidence: list[RetrievedEvidence],
    ) -> list[RetrievedEvidence]:
        """
        Apply contextual reranking to already-retrieved evidence.
        """

        reranked: list[RetrievedEvidence] = []

        for item in evidence:
            score = self.score(
                understanding=understanding,
                evidence=item,
            )

            item.context_score = score.total
            item.final_score = (
                item.hybrid_score
                + score.total
            )

            reranked.append(item)

        reranked.sort(
            key=lambda item: item.final_score,
            reverse=True,
        )

        return reranked