from __future__ import annotations

import pickle
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from support_agent.ingestion.chunker import KnowledgeChunk


_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+")


def tokenize(text: str) -> list[str]:
    """Tokenize text for BM25."""
    return [
        token.lower()
        for token in _TOKEN_PATTERN.findall(text)
    ]


class BM25Retriever:
    """Lexical retrieval over knowledge-base chunks."""

    def __init__(
        self,
        chunks: list[KnowledgeChunk],
        index: BM25Okapi,
    ) -> None:
        self.chunks = chunks
        self.index = index

    @classmethod
    def build(
        cls,
        chunks: list[KnowledgeChunk],
    ) -> "BM25Retriever":
        corpus = [
            tokenize(
                " ".join(
                    [
                        chunk.title,
                        " ".join(chunk.breadcrumbs),
                        " ".join(chunk.section_path),
                        chunk.text,
                    ]
                )
            )
            for chunk in chunks
        ]

        index = BM25Okapi(corpus)

        return cls(
            chunks=chunks,
            index=index,
        )

    def search(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[tuple[int, float]]:
        """Return chunk indexes and BM25 scores."""
        query_tokens = tokenize(query)

        scores = self.index.get_scores(query_tokens)

        ranked_indexes = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True,
        )

        return [
            (index, float(scores[index]))
            for index in ranked_indexes[:top_k]
        ]

    def save(self, path: Path) -> None:
        """Persist BM25 index."""
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("wb") as file:
            pickle.dump(
                self.index,
                file,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

    @classmethod
    def load(
        cls,
        chunks: list[KnowledgeChunk],
        path: Path,
    ) -> "BM25Retriever":
        """Load a persisted BM25 index."""
        with path.open("rb") as file:
            index = pickle.load(file)

        return cls(
            chunks=chunks,
            index=index,
        )