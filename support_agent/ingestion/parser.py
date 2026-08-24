from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class KnowledgeArticle:
    """A parsed HackerRank knowledge-base article."""

    article_id: str
    title: str
    title_slug: str
    source_url: str
    article_slug: str
    last_updated_exact: str
    last_updated_relative: str
    breadcrumbs: tuple[str, ...]
    source_path: str
    text: str

    @property
    def product_area(self) -> str:
        """Return the top-level knowledge-base area."""
        return self.breadcrumbs[0] if self.breadcrumbs else "Unknown"

    @property
    def category(self) -> str:
        """Return the second-level knowledge-base category."""
        return self.breadcrumbs[1] if len(self.breadcrumbs) > 1 else "Unknown"


def _unquote(value: str) -> str:
    """Remove surrounding single or double quotes."""
    value = value.strip()

    if len(value) >= 2:
        if value[0] == value[-1] and value[0] in {"'", '"'}:
            return value[1:-1]

    return value


def _extract_frontmatter(text: str) -> tuple[dict[str, str], tuple[str, ...]]:
    """
    Extract the simple frontmatter used by the HackerRank KB.

    Supported scalar fields include:
        title
        title_slug
        source_url
        article_slug
        last_updated_exact
        last_updated_relative

    The breadcrumbs field is represented as a YAML-style list.
    """
    if not text.startswith("---"):
        return {}, ()

    parts = text.split("---", 2)

    if len(parts) != 3:
        return {}, ()

    frontmatter = parts[1]

    metadata: dict[str, str] = {}
    breadcrumbs: list[str] = []

    current_list: str | None = None

    for raw_line in frontmatter.splitlines():
        line = raw_line.rstrip()

        if not line.strip():
            continue

        # Handle list entries such as:
        #
        # breadcrumbs:
        #   - "Screen"
        #   - "Test Integrity"
        if current_list == "breadcrumbs":
            match = re.match(r"^\s*-\s*(.+?)\s*$", line)

            if match:
                breadcrumbs.append(_unquote(match.group(1)))
                continue

            current_list = None

        match = re.match(
            r"^\s*([A-Za-z_]+):\s*(.*?)\s*$",
            line,
        )

        if not match:
            continue

        key, value = match.groups()

        if key == "breadcrumbs":
            current_list = "breadcrumbs"

            if value:
                # This is not currently used by the supplied KB,
                # but supports a simple inline value if encountered.
                breadcrumbs.append(_unquote(value))

            continue

        metadata[key] = _unquote(value)

    return metadata, tuple(breadcrumbs)


def _extract_article_id(
    article_slug: str,
    source_path: Path,
) -> str:
    """Extract the numeric article ID from article_slug."""
    match = re.match(r"(\d+)", article_slug)

    if match:
        return match.group(1)

    # Fallback to the filename if article_slug is unavailable.
    match = re.match(r"(\d+)", source_path.stem)

    if match:
        return match.group(1)

    return source_path.stem


def parse_article(
    path: Path,
    knowledge_base_root: Path,
) -> KnowledgeArticle:
    """Parse one Markdown knowledge-base article."""
    raw_text = path.read_text(encoding="utf-8")

    metadata, breadcrumbs = _extract_frontmatter(raw_text)

    article_slug = metadata.get("article_slug", "")

    article_id = _extract_article_id(
        article_slug=article_slug,
        source_path=path,
    )

    title = metadata.get("title", "")

    if not title:
        match = re.search(
            r"^#\s+(.+?)\s*$",
            raw_text,
            re.MULTILINE,
        )

        title = match.group(1).strip() if match else path.stem

    source_path = path.relative_to(knowledge_base_root).as_posix()

    # Remove frontmatter before passing the article body downstream.
    text = raw_text

    if raw_text.startswith("---"):
        parts = raw_text.split("---", 2)

        if len(parts) == 3:
            text = parts[2].strip()

    return KnowledgeArticle(
        article_id=article_id,
        title=title,
        title_slug=metadata.get("title_slug", ""),
        source_url=metadata.get("source_url", ""),
        article_slug=article_slug,
        last_updated_exact=metadata.get("last_updated_exact", ""),
        last_updated_relative=metadata.get("last_updated_relative", ""),
        breadcrumbs=breadcrumbs,
        source_path=source_path,
        text=text.strip(),
    )


def load_articles(
    knowledge_base_root: Path,
) -> list[KnowledgeArticle]:
    """Load all Markdown knowledge-base articles."""
    paths = sorted(
        path
        for path in knowledge_base_root.rglob("*.md")
        if path.name.lower() != "index.md"
    )

    return [
        parse_article(
            path=path,
            knowledge_base_root=knowledge_base_root,
        )
        for path in paths
    ]