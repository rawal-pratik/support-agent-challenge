from __future__ import annotations

import re
from dataclasses import dataclass

from support_agent.ingestion.parser import KnowledgeArticle


@dataclass(frozen=True)
class KnowledgeChunk:
    """A semantically meaningful chunk from a KB article."""

    chunk_id: str
    article_id: str
    title: str
    breadcrumbs: tuple[str, ...]
    source_path: str
    section: str
    section_path: tuple[str, ...]
    text: str

    @property
    def product_area(self) -> str:
        """Return the top-level knowledge-base area."""
        return self.breadcrumbs[0] if self.breadcrumbs else "Unknown"

    @property
    def category(self) -> str:
        """Return the second-level knowledge-base category."""
        return self.breadcrumbs[1] if len(self.breadcrumbs) > 1 else "Unknown"


_HEADING_PATTERN = re.compile(
    r"^(#{1,6})\s+(.+?)\s*$"
)


def chunk_article(
    article: KnowledgeArticle,
) -> list[KnowledgeChunk]:
    """
    Split an article into Markdown-section-based chunks.

    Each chunk retains:
    - article metadata
    - KB breadcrumbs
    - section name
    - heading hierarchy
    - section text
    """
    lines = article.text.splitlines()

    chunks: list[KnowledgeChunk] = []

    # Markdown heading hierarchy.
    heading_stack: list[tuple[int, str]] = []

    current_section = article.title
    current_section_path: tuple[str, ...] = (article.title,)
    current_lines: list[str] = []

    section_number = 0

    def flush() -> None:
        nonlocal section_number

        text = "\n".join(current_lines).strip()

        if not text:
            return

        section_number += 1

        chunks.append(
            KnowledgeChunk(
                chunk_id=f"{article.article_id}:{section_number}",
                article_id=article.article_id,
                title=article.title,
                breadcrumbs=article.breadcrumbs,
                source_path=article.source_path,
                section=current_section,
                section_path=current_section_path,
                text=text,
            )
        )

    for line in lines:
        match = _HEADING_PATTERN.match(line)

        if match:
            flush()

            level = len(match.group(1))
            heading = match.group(2).strip()

            # Remove headings at the same or deeper level.
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()

            heading_stack.append((level, heading))

            current_section = heading
            current_section_path = tuple(
                heading_text
                for _, heading_text in heading_stack
            )

            current_lines = [line]

        else:
            current_lines.append(line)

    flush()

    # Preserve an article even if it has no headings.
    if not chunks and article.text.strip():
        chunks.append(
            KnowledgeChunk(
                chunk_id=f"{article.article_id}:1",
                article_id=article.article_id,
                title=article.title,
                breadcrumbs=article.breadcrumbs,
                source_path=article.source_path,
                section=article.title,
                section_path=(article.title,),
                text=article.text.strip(),
            )
        )

    return chunks


def chunk_articles(
    articles: list[KnowledgeArticle],
) -> list[KnowledgeChunk]:
    """Chunk all knowledge-base articles."""
    chunks: list[KnowledgeChunk] = []

    for article in articles:
        chunks.extend(chunk_article(article))

    return chunks