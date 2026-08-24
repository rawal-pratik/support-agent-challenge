from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


Status = Literal["replied", "escalated"]

RequestType = Literal[
    "product_issue",
    "feature_request",
    "bug",
    "invalid",
]

RequesterType = Literal[
    "customer_admin",
    "candidate",
    "community_user",
    "unknown",
]


class SupportResponse(BaseModel):
    """Final output for a single support ticket."""

    model_config = ConfigDict(extra="forbid")

    status: Status
    product_area: str = Field(min_length=1)
    response: str = Field(min_length=1)
    justification: str = Field(min_length=1)
    request_type: RequestType

    @field_validator(
        "product_area",
        "response",
        "justification",
    )
    @classmethod
    def reject_whitespace_only(
        cls,
        value: str,
    ) -> str:
        if not value.strip():
            raise ValueError(
                "Value must not be empty or whitespace only."
            )
        return value.strip()


class TicketUnderstanding(BaseModel):
    """Structured interpretation of an incoming support ticket."""

    model_config = ConfigDict(extra="forbid")

    requester_type: RequesterType

    is_hackerrank_related: bool = Field(
        description=(
            "Whether the ticket is related to HackerRank "
            "products, services, platform usage, or support."
        ),
    )

    intent: str = Field(
        min_length=1,
        description="The user's underlying goal or task.",
    )

    product_area_hint: str = Field(
        min_length=1,
        description=(
            "A preliminary product-area hypothesis "
            "based only on the ticket."
        ),
    )

    issue_summary: str = Field(
        min_length=1,
        description=(
            "A concise description of the actual "
            "problem or request."
        ),
    )

    entities: list[str] = Field(
        default_factory=list,
        description=(
            "Important product terms, objects, "
            "features, or concepts from the ticket."
        ),
    )

    search_queries: list[str] = Field(
        min_length=1,
        max_length=5,
        description=(
            "Search queries optimized for the "
            "knowledge-base retriever."
        ),
    )

    @field_validator(
        "intent",
        "product_area_hint",
        "issue_summary",
    )
    @classmethod
    def reject_whitespace_only(
        cls,
        value: str,
    ) -> str:
        if not value.strip():
            raise ValueError(
                "Value must not be empty or whitespace only."
            )
        return value.strip()

    @field_validator(
        "entities",
        "search_queries",
    )
    @classmethod
    def remove_empty_values(
        cls,
        values: list[str],
    ) -> list[str]:
        return [
            value.strip()
            for value in values
            if value.strip()
        ]


class RetrievedEvidence(BaseModel):
    """A single piece of retrieved knowledge-base evidence."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    article_id: str
    title: str
    breadcrumbs: list[str]
    source_path: str
    section: str
    section_path: list[str]
    text: str

    # Score from query-aware RRF fusion before contextual reranking.
    hybrid_score: float

    # Score produced by the contextual reranker.
    context_score: float = 0.0

    # Final score used to rank the evidence.
    final_score: float = 0.0

    bm25_score: float | None
    vector_score: float | None

    bm25_rank: int | None
    vector_rank: int | None

    matched_queries: list[str] = Field(
        default_factory=list,
    )