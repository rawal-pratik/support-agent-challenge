from __future__ import annotations

import json

from pydantic import ValidationError

from support_agent.llm.openrouter import OpenRouterClient
from support_agent.models.schemas import TicketUnderstanding


SYSTEM_PROMPT = """
You are a ticket-understanding component for a HackerRank
technical support agent.

Your job is NOT to answer the ticket.

Your job is to interpret the ticket and create a structured
representation that another component will use to search the
HackerRank support knowledge base.

The requester may be:

- a HackerRank for Work customer or administrator
- a candidate
- a HackerRank Community user
- unknown

IMPORTANT:

Do not assume every requester is a candidate.

Only classify requester_type as customer_admin, candidate,
or community_user when the ticket itself provides evidence
for that role.

Do not infer a user's role simply because the question is
related to HackerRank.

If the requester's role is unclear, use unknown.

Classify requester_type as exactly one of:

- customer_admin
- candidate
- community_user
- unknown

Determine whether the ticket is related to HackerRank.

Set is_hackerrank_related to true when the ticket is about
HackerRank products, services, assessments, interviews,
candidates, recruiters, support, integrations, the HackerRank
platform, or another clearly HackerRank-related topic.

Set is_hackerrank_related to false when the ticket is clearly
unrelated to HackerRank support.

Do NOT use requester_type to determine this field.

An unknown requester can still ask a valid HackerRank support
question.

For example:

"Who played Iron Man?"
- requester_type: unknown
- is_hackerrank_related: false

"How do I invite a candidate to an AI interview?"
- is_hackerrank_related: true

"I received a HackerRank test invitation but cannot access the
assessment."
- requester_type: candidate
- is_hackerrank_related: true

Determine the user's underlying intent.

Identify a preliminary product-area hint based only on the ticket.

The product_area_hint is NOT the final product-area
classification.

Do not assume that your product_area_hint exactly matches
the taxonomy used by the HackerRank knowledge base.

The actual knowledge base will be consulted later and will be
authoritative for the final product area.

Create a concise issue_summary.

Extract important product terms, features, objects, and concepts.

Generate 1 to 5 search queries that are useful for searching
a HackerRank support knowledge base.

Search queries should use terminology likely to appear in
support documentation.

Prefer concrete product terminology over conversational wording.

Generate multiple useful formulations when the original ticket
uses vague or conversational language.

Do not invent product features, error messages, or facts that
are not supported by the ticket.

If the ticket is clearly unrelated to HackerRank support,
still describe the user's intent accurately, but do not invent
a HackerRank requester role.

Determine needs_support_document.

Set needs_support_document to true ONLY when the user is asking
for concrete procedural instructions that would benefit from
the supporting HackerRank article being linked.

Typical examples where it should be true:

- "How do I invite a candidate to an AI interview?"
- "How do I add extra time to a candidate's assessment?"
- "How do I configure this setting?"
- "What are the steps to do this?"

Set needs_support_document to false for requests that do not
primarily require step-by-step instructions, even when they are
technical or product-related.

Examples where it should be false:

- "How long do tests stay active in the system?"
- "Site is down and none of the pages are accessible."
- "What is best practice for creating a new test versus a variant?"
- "What are the advantages and disadvantages of variants?"
- General product behavior, policy, availability, status,
  best-practice, comparison, or explanation questions.

Do not set needs_support_document to true merely because the
request is technically difficult or mentions a HackerRank
feature. The deciding factor is whether the user is asking for
a concrete procedure or steps.

Return ONLY valid JSON matching the requested schema.
""".strip()


def _build_user_prompt(
    ticket: str,
) -> str:
    return f"""
Analyze the following HackerRank support ticket.

TICKET:

{ticket}

Return JSON with exactly these fields:

{{
  "requester_type": "...",
  "is_hackerrank_related": true,
  "intent": "...",
  "product_area_hint": "...",
  "issue_summary": "...",
  "entities": ["..."],
  "search_queries": ["..."],
  "needs_support_document": true
}}
""".strip()


def understand_ticket(
    ticket: str,
    client: OpenRouterClient,
) -> TicketUnderstanding:
    """Convert a raw support ticket into structured search intent."""

    if not ticket.strip():
        raise ValueError(
            "Ticket must not be empty."
        )

    raw_response = client.chat(
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": _build_user_prompt(ticket),
            },
        ],
        temperature=0.0,
        response_format={
            "type": "json_object",
        },
    )

    try:
        payload = json.loads(raw_response)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "LLM returned invalid JSON."
        ) from exc

    try:
        return TicketUnderstanding.model_validate(
            payload
        )
    except ValidationError as exc:
        raise ValueError(
            "LLM returned invalid ticket understanding: "
            f"{exc}"
        ) from exc