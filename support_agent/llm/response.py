from __future__ import annotations

import json

from support_agent.llm.openrouter import OpenRouterClient
from support_agent.models.schemas import (
    RetrievedEvidence,
    SupportResponse,
    TicketUnderstanding,
)


SYSTEM_PROMPT = """
You are the answer-generation component of a HackerRank technical
support agent.

Your job is to generate the final support response for a ticket
using the ticket understanding and retrieved knowledge-base evidence.

The knowledge base is authoritative.

IMPORTANT RULES:

1. Answer only using information supported by the retrieved evidence.
2. Do not invent product behavior, UI labels, workflows, limitations,
   URLs, or troubleshooting steps.
3. Do not use general world knowledge to answer the ticket.
4. If the ticket is clearly unrelated to HackerRank, politely decline
   the request without answering the unrelated question.
5. If the evidence does not provide enough information to answer a
   HackerRank-related request reliably, escalate the ticket.
6. Do not claim that an issue is a bug unless the evidence supports
   that conclusion.
7. Do not escalate ordinary product-usage questions when the evidence
   provides sufficient instructions.
8. Keep the response concise and similar to how a HackerRank support
   agent would respond to a customer or candidate.
9. Do not mention internal retrieval, embeddings, RRF, chunks, models,
   prompts, or the knowledge base.
10. Do not expose the reasoning process.

REQUEST TYPES:

- product_issue:
  A HackerRank product question or issue that can be answered using
  the available evidence.

- feature_request:
  The user is requesting a new capability or behavior that is not
  simply asking how to use an existing feature.

- bug:
  The user reports behavior that appears to be a product defect and
  the evidence supports treating it as a bug.

- invalid:
  The request is unrelated to HackerRank support or otherwise outside
  the supported scope.

STATUS:

- replied:
  Use when the request can be answered confidently using the evidence.

- escalated:
  Use when the issue requires developer/product-engineering attention
  or the available evidence is insufficient to provide a reliable
  resolution.

PRODUCT AREA:

Use the product area supported by the retrieved evidence.

Do not blindly copy product_area_hint from the ticket understanding.
The knowledge base is authoritative for product-area classification.

RESPONSE:

Write the response as a real support-agent reply.

For procedural questions, provide clear numbered steps when appropriate.

For candidates, use candidate-facing language.

For customer administrators, use customer-admin-facing language.

Do not add unnecessary disclaimers.

JUSTIFICATION:

Provide a concise internal justification explaining why the selected
status and request type are appropriate and which evidence supports
the decision.

Return ONLY valid JSON matching the requested schema.
""".strip()


def _tokens(text: str) -> set[str]:
    """
    Convert text into a simple normalized token set.

    This is intentionally lightweight. The purpose is only to identify
    the most relevant chunks inside an already-selected article.
    """

    return {
        token
        for token in (
            text.lower()
            .replace("-", " ")
            .replace("/", " ")
            .replace("_", " ")
            .split()
        )
        if token
    }


def _evidence_relevance_score(
    evidence: RetrievedEvidence,
    ticket: str,
    understanding: TicketUnderstanding,
) -> float:
    """
    Estimate how relevant an evidence item is to the ticket.

    Retrieval has already performed the primary ranking. This score is
    used only when deciding which article should supply evidence to the
    response generator.
    """

    query_tokens = _tokens(
        " ".join(
            [
                ticket,
                understanding.intent,
                understanding.issue_summary,
                *understanding.entities,
                *understanding.search_queries,
            ]
        )
    )

    title_tokens = _tokens(evidence.title)

    section_tokens = _tokens(
        " ".join(
            [
                evidence.section,
                *evidence.section_path,
            ]
        )
    )

    content_tokens = _tokens(evidence.text)

    title_overlap = len(
        query_tokens & title_tokens
    )

    section_overlap = len(
        query_tokens & section_tokens
    )

    content_overlap = len(
        query_tokens & content_tokens
    )

    return (
        title_overlap * 10.0
        + section_overlap * 5.0
        + content_overlap
        + evidence.hybrid_score
    )


def _select_response_evidence(
    ticket: str,
    understanding: TicketUnderstanding,
    evidence: list[RetrievedEvidence],
    max_chunks: int = 3,
) -> list[RetrievedEvidence]:
    """
    Select the most relevant chunks from the most relevant article.

    Retrieval may return several related articles. First select the
    article that is most directly relevant to the ticket. Then select
    only the strongest chunks from that article so the response
    generator does not mix different workflows or unrelated sections.
    """

    if not evidence:
        return []

    article_scores: dict[str, float] = {}

    for item in evidence:
        score = _evidence_relevance_score(
            evidence=item,
            ticket=ticket,
            understanding=understanding,
        )

        article_scores[item.article_id] = max(
            article_scores.get(
                item.article_id,
                0.0,
            ),
            score,
        )

    best_article_id = evidence[0].article_id
    best_article_score = article_scores[
        best_article_id
    ]

    for item in evidence:
        score = article_scores[item.article_id]

        if score > best_article_score:
            best_article_id = item.article_id
            best_article_score = score

    article_evidence = [
        item
        for item in evidence
        if item.article_id == best_article_id
    ]

    if len(article_evidence) <= max_chunks:
        return article_evidence

    query_tokens = _tokens(
        " ".join(
            [
                ticket,
                understanding.intent,
                understanding.issue_summary,
                *understanding.entities,
                *understanding.search_queries,
            ]
        )
    )

    def chunk_score(
        item: RetrievedEvidence,
    ) -> float:
        title_tokens = _tokens(item.title)

        section_tokens = _tokens(
            " ".join(
                [
                    item.section,
                    *item.section_path,
                ]
            )
        )

        content_tokens = _tokens(item.text)

        title_overlap = len(
            query_tokens & title_tokens
        )

        section_overlap = len(
            query_tokens & section_tokens
        )

        content_overlap = len(
            query_tokens & content_tokens
        )

        return (
            title_overlap * 10.0
            + section_overlap * 5.0
            + content_overlap
            + item.hybrid_score
        )

    ranked_chunks = sorted(
        article_evidence,
        key=chunk_score,
        reverse=True,
    )

    return ranked_chunks[:max_chunks]


def _build_evidence_prompt(
    evidence: list[RetrievedEvidence],
) -> str:
    """
    Format selected evidence for the response-generation model.
    """

    if not evidence:
        return "No knowledge-base evidence was retrieved."

    sections: list[str] = []

    for index, item in enumerate(
        evidence,
        start=1,
    ):
        sections.append(
            f"""
EVIDENCE {index}

Article ID: {item.article_id}
Title: {item.title}
Breadcrumbs: {" > ".join(item.breadcrumbs)}
Section: {" > ".join(item.section_path)}

Content:
{item.text}
""".strip()
        )

    return "\n\n".join(sections)


def _build_user_prompt(
    ticket: str,
    understanding: TicketUnderstanding,
    evidence: list[RetrievedEvidence],
) -> str:
    """
    Build the final answer-generation prompt.
    """

    evidence_text = _build_evidence_prompt(
        evidence
    )

    return f"""
Generate the final support response for the following ticket.

TICKET:
{ticket}

TICKET UNDERSTANDING:
{json.dumps(
    understanding.model_dump(),
    indent=2,
    ensure_ascii=False,
)}

The evidence below was selected from the most relevant article
and contains only the sections most relevant to the user's request.

Use these sections as the authoritative source for the answer.

Do not introduce information from other sections of the article
unless it is explicitly present in the supplied evidence.

RETRIEVED EVIDENCE:
{evidence_text}

Return JSON with exactly these fields:

{{
  "status": "replied" or "escalated",
  "product_area": "...",
  "response": "...",
  "justification": "...",
  "request_type": "product_issue" or "feature_request" or "bug" or "invalid"
}}
""".strip()


def generate_response(
    ticket: str,
    understanding: TicketUnderstanding,
    evidence: list[RetrievedEvidence],
    client: OpenRouterClient,
) -> SupportResponse:
    """
    Generate the final support response from the ticket understanding
    and retrieved evidence.
    """

    if not ticket.strip():
        raise ValueError(
            "Ticket must not be empty."
        )

    if not understanding.is_hackerrank_related:
        return SupportResponse(
            status="replied",
            product_area=(
                understanding.product_area_hint
                or "not_applicable"
            ),
            response=(
                "This question is outside the scope of "
                "HackerRank technical support. We can only "
                "assist with HackerRank platform-related "
                "issues such as assessments, interviews, "
                "certifications, and account management."
            ),
            justification=(
                "The ticket is unrelated to HackerRank "
                "support. The ticket understanding classifies "
                "it as outside HackerRank scope, so the request "
                "is declined without answering the unrelated "
                "question."
            ),
            request_type="invalid",
        )

    selected_evidence = _select_response_evidence(
        ticket=ticket,
        understanding=understanding,
        evidence=evidence,
    )

    if not selected_evidence:
        return SupportResponse(
            status="escalated",
            product_area=(
                understanding.product_area_hint
                or "unknown"
            ),
            response=(
                "We’re unable to provide a reliable "
                "resolution from the available information. "
                "We’ll escalate this issue for further review."
            ),
            justification=(
                "The request is related to HackerRank, but "
                "there is insufficient knowledge-base evidence "
                "to provide a reliable answer."
            ),
            request_type="product_issue",
        )

    raw_response = client.chat(
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": _build_user_prompt(
                    ticket=ticket,
                    understanding=understanding,
                    evidence=selected_evidence,
                ),
            },
        ],
        temperature=0.0,
        response_format={
            "type": "json_object",
        },
    )

    try:
        payload = json.loads(
            raw_response
        )
    except json.JSONDecodeError as exc:
        raise ValueError(
            "LLM returned invalid JSON."
        ) from exc

    try:
        return SupportResponse.model_validate(
            payload
        )
    except Exception as exc:
        raise ValueError(
            "LLM returned invalid support response: "
            f"{exc}"
        ) from exc