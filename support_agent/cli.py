from __future__ import annotations

import argparse
import json
from pathlib import Path

from support_agent import __version__
from support_agent.ingestion.chunker import chunk_articles
from support_agent.ingestion.parser import load_articles
from support_agent.llm.openrouter import OpenRouterClient
from support_agent.llm.response import generate_response
from support_agent.llm.understanding import understand_ticket
from support_agent.retrieval.bm25 import BM25Retriever
from support_agent.retrieval.hybrid import HybridRetriever
from support_agent.retrieval.indexer import RetrievalIndexer
from support_agent.retrieval.query_aware import QueryAwareRetriever
from support_agent.retrieval.storage import load_chunks
from support_agent.retrieval.vector import VectorRetriever


PROJECT_ROOT = Path(__file__).resolve().parent.parent

KNOWLEDGE_BASE_DIR = (
    PROJECT_ROOT / "data" / "knowledge_base"
)

INDEX_DIR = (
    PROJECT_ROOT / "data" / "index"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="support-agent",
        description="Terminal-based HackerRank support triage agent.",
    )

    parser.add_argument(
        "--version",
        action="version",
        version=__version__,
    )

    subparsers = parser.add_subparsers(
        dest="command"
    )

    index_parser = subparsers.add_parser(
        "index",
        help="Inspect the knowledge base.",
    )

    index_parser.add_argument(
        "--knowledge-base",
        type=Path,
        default=KNOWLEDGE_BASE_DIR,
    )

    build_parser = subparsers.add_parser(
        "build-index",
        help="Build the local retrieval indexes.",
    )

    build_parser.add_argument(
        "--knowledge-base",
        type=Path,
        default=KNOWLEDGE_BASE_DIR,
    )

    build_parser.add_argument(
        "--index-dir",
        type=Path,
        default=INDEX_DIR,
    )

    search_parser = subparsers.add_parser(
        "search",
        help="Search the knowledge base.",
    )

    search_parser.add_argument(
        "query",
        type=str,
    )

    search_parser.add_argument(
        "--top-k",
        type=int,
        default=5,
    )

    search_parser.add_argument(
        "--candidate-k",
        type=int,
        default=20,
    )

    search_parser.add_argument(
        "--index-dir",
        type=Path,
        default=INDEX_DIR,
    )

    understand_parser = subparsers.add_parser(
        "understand",
        help="Understand a support ticket.",
    )

    understand_parser.add_argument(
        "ticket",
        type=str,
    )

    retrieve_parser = subparsers.add_parser(
        "retrieve",
        help="Understand a ticket and retrieve evidence.",
    )

    retrieve_parser.add_argument(
        "ticket",
        type=str,
    )

    retrieve_parser.add_argument(
        "--top-k",
        type=int,
        default=5,
    )

    retrieve_parser.add_argument(
        "--candidate-k",
        type=int,
        default=20,
    )

    retrieve_parser.add_argument(
        "--index-dir",
        type=Path,
        default=INDEX_DIR,
    )

    answer_parser = subparsers.add_parser(
        "answer",
        help="Generate a grounded support response.",
    )

    answer_parser.add_argument(
        "ticket",
        type=str,
    )

    answer_parser.add_argument(
        "--top-k",
        type=int,
        default=5,
    )

    answer_parser.add_argument(
        "--candidate-k",
        type=int,
        default=20,
    )

    answer_parser.add_argument(
        "--index-dir",
        type=Path,
        default=INDEX_DIR,
    )

    return parser


def run_index(
    knowledge_base: Path,
) -> None:
    if not knowledge_base.exists():
        raise SystemExit(
            f"Knowledge base directory does not exist: "
            f"{knowledge_base}"
        )

    articles = load_articles(
        knowledge_base
    )

    chunks = chunk_articles(articles)

    print(
        f"Knowledge base: {knowledge_base}"
    )

    print(f"Articles: {len(articles)}")
    print(f"Chunks: {len(chunks)}")

    print("\nSample articles:")

    for article in articles[:5]:
        breadcrumb_path = " > ".join(
            article.breadcrumbs
        )

        print(
            f"- [{article.article_id}] "
            f"{article.title}"
        )

        print(
            f"  Path: {article.source_path}"
        )

        print(
            f"  URL: {article.source_url}"
        )

        print(
            f"  Breadcrumbs: {breadcrumb_path}"
        )

    print("\nSample chunks:")

    for chunk in chunks[:5]:
        section_path = " > ".join(
            chunk.section_path
        )

        preview = (
            chunk.text
            .replace("\n", " ")
            .strip()
        )

        if len(preview) > 160:
            preview = f"{preview[:160]}..."

        print(
            f"- [{chunk.chunk_id}] "
            f"{section_path}"
        )

        print(f"  {preview}")


def run_build_index(
    knowledge_base: Path,
    index_dir: Path,
) -> None:
    indexer = RetrievalIndexer(
        knowledge_base=knowledge_base,
        index_directory=index_dir,
    )

    indexer.build()


def _load_hybrid_retriever(
    index_dir: Path,
) -> HybridRetriever:
    chunks_path = index_dir / "chunks.json"
    bm25_path = index_dir / "bm25.pkl"
    faiss_path = index_dir / "faiss.index"

    required_files = [
        chunks_path,
        bm25_path,
        faiss_path,
    ]

    missing = [
        path
        for path in required_files
        if not path.exists()
    ]

    if missing:
        print(
            "Retrieval index is incomplete."
        )

        print(
            "Run "
            "`python -m support_agent.cli "
            "build-index` first."
        )

        raise SystemExit(1)

    chunks = load_chunks(chunks_path)

    bm25 = BM25Retriever.load(
        chunks=chunks,
        path=bm25_path,
    )

    vector = VectorRetriever.load(
        chunks=chunks,
        path=faiss_path,
    )

    return HybridRetriever(
        chunks=chunks,
        bm25=bm25,
        vector=vector,
    )


def run_search(
    query: str,
    top_k: int,
    candidate_k: int,
    index_dir: Path,
) -> None:
    retriever = _load_hybrid_retriever(
        index_dir
    )

    results = retriever.search(
        query=query,
        top_k=top_k,
        candidate_k=candidate_k,
    )

    print(f"\nQuery: {query}")
    print(
        f"Showing top {len(results)} results:\n"
    )

    for rank, result in enumerate(
        results,
        start=1,
    ):
        chunk = result.chunk

        print(
            f"{rank}. "
            f"[{chunk.chunk_id}] "
            f"{chunk.title}"
        )

        print(
            f"   Section: "
            f"{' > '.join(chunk.section_path)}"
        )

        print(
            f"   Breadcrumbs: "
            f"{' > '.join(chunk.breadcrumbs)}"
        )

        print(
            f"   Source URL: "
            f"{chunk.source_url}"
        )

        print(
            f"   Hybrid: "
            f"{result.hybrid_score:.6f}"
        )

        print(
            f"   BM25: "
            f"{result.bm25_score}"
            f" (rank {result.bm25_rank})"
        )

        print(
            f"   Vector: "
            f"{result.vector_score}"
            f" (rank {result.vector_rank})"
        )

        preview = (
            chunk.text
            .replace("\n", " ")
            .strip()
        )

        if len(preview) > 300:
            preview = f"{preview[:300]}..."

        print(f"   {preview}")
        print()


def run_understand(
    ticket: str,
) -> None:
    client = OpenRouterClient()

    understanding = understand_ticket(
        ticket=ticket,
        client=client,
    )

    print(
        json.dumps(
            understanding.model_dump(),
            indent=2,
            ensure_ascii=False,
        )
    )


def _retrieve_ticket(
    ticket: str,
    top_k: int,
    candidate_k: int,
    index_dir: Path,
):
    retriever = _load_hybrid_retriever(
        index_dir
    )

    client = OpenRouterClient()

    query_aware = QueryAwareRetriever(
        retriever=retriever,
        llm_client=client,
    )

    return query_aware.search(
        ticket=ticket,
        top_k=top_k,
        candidate_k=candidate_k,
    )


def run_retrieve(
    ticket: str,
    top_k: int,
    candidate_k: int,
    index_dir: Path,
) -> None:
    result = _retrieve_ticket(
        ticket=ticket,
        top_k=top_k,
        candidate_k=candidate_k,
        index_dir=index_dir,
    )

    print("\nTicket:")
    print(ticket)

    print("\nUnderstanding:")
    print(
        json.dumps(
            result.understanding.model_dump(),
            indent=2,
            ensure_ascii=False,
        )
    )

    print("\nRetrieved evidence:")
    print(
        f"Showing top {len(result.evidence)} results:\n"
    )

    for rank, evidence in enumerate(
        result.evidence,
        start=1,
    ):
        print(
            f"{rank}. "
            f"[{evidence.chunk_id}] "
            f"{evidence.title}"
        )

        print(
            f"   Section: "
            f"{' > '.join(evidence.section_path)}"
        )

        print(
            f"   Breadcrumbs: "
            f"{' > '.join(evidence.breadcrumbs)}"
        )

        print(
            f"   Source URL: "
            f"{evidence.source_url}"
        )

        print(
            f"   Query matches: "
            f"{'; '.join(evidence.matched_queries)}"
        )

        print(
            f"   Fused score: "
            f"{evidence.hybrid_score:.6f}"
        )

        print(
            f"   Best BM25 score: "
            f"{evidence.bm25_score}"
        )

        print(
            f"   Best vector score: "
            f"{evidence.vector_score}"
        )

        preview = (
            evidence.text
            .replace("\n", " ")
            .strip()
        )

        if len(preview) > 350:
            preview = f"{preview[:350]}..."

        print(
            f"   {preview}"
        )

        print()


def run_answer(
    ticket: str,
    top_k: int,
    candidate_k: int,
    index_dir: Path,
) -> None:
    """
    Retrieve evidence and generate the final grounded
    support response.
    """

    result = _retrieve_ticket(
        ticket=ticket,
        top_k=top_k,
        candidate_k=candidate_k,
        index_dir=index_dir,
    )

    client = OpenRouterClient()

    response = generate_response(
        ticket=ticket,
        understanding=result.understanding,
        evidence=result.evidence,
        client=client,
    )

    print("\nTicket:")
    print(ticket)

    print("\nUnderstanding:")
    print(
        json.dumps(
            result.understanding.model_dump(),
            indent=2,
            ensure_ascii=False,
        )
    )

    print("\nRetrieved evidence:")
    print(
        f"Showing top {len(result.evidence)} results."
    )

    print("\nFinal response:")
    print(
        json.dumps(
            response.model_dump(),
            indent=2,
            ensure_ascii=False,
        )
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "index":
        run_index(args.knowledge_base)

    elif args.command == "build-index":
        run_build_index(
            knowledge_base=args.knowledge_base,
            index_dir=args.index_dir,
        )

    elif args.command == "search":
        run_search(
            query=args.query,
            top_k=args.top_k,
            candidate_k=args.candidate_k,
            index_dir=args.index_dir,
        )

    elif args.command == "understand":
        run_understand(args.ticket)

    elif args.command == "retrieve":
        run_retrieve(
            ticket=args.ticket,
            top_k=args.top_k,
            candidate_k=args.candidate_k,
            index_dir=args.index_dir,
        )

    elif args.command == "answer":
        run_answer(
            ticket=args.ticket,
            top_k=args.top_k,
            candidate_k=args.candidate_k,
            index_dir=args.index_dir,
        )


if __name__ == "__main__":
    main()