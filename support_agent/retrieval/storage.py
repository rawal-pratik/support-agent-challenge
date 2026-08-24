from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from support_agent.ingestion.chunker import KnowledgeChunk


def save_chunks(
    chunks: list[KnowledgeChunk],
    path: Path,
) -> None:
    """Save knowledge chunks as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = [
        asdict(chunk)
        for chunk in chunks
    ]

    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def load_chunks(path: Path) -> list[KnowledgeChunk]:
    """Load knowledge chunks from JSON."""
    payload = json.loads(
        path.read_text(encoding="utf-8")
    )

    return [
        KnowledgeChunk(
            chunk_id=item["chunk_id"],
            article_id=item["article_id"],
            title=item["title"],
            breadcrumbs=tuple(item["breadcrumbs"]),
            source_path=item["source_path"],
            section=item["section"],
            section_path=tuple(item["section_path"]),
            text=item["text"],
        )
        for item in payload
    ]