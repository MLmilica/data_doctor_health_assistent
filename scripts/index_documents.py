"""Index clinical markdown documents into the Chroma vector store."""

from __future__ import annotations

import argparse
import json
import sys

from data.vectorstore import DocumentVectorStore, IndexSummary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Index data/documents/*.md sections into Chroma (data/chroma/).",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete and recreate the collection before indexing.",
    )
    args = parser.parse_args(argv)

    try:
        summary = DocumentVectorStore().index_documents(reset=args.reset)
    except Exception as exc:
        print(f"Indexing failed: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(_summary_payload(summary), indent=2))
    return 0


def _summary_payload(summary: IndexSummary) -> dict[str, object]:
    return {
        "document_count": summary.document_count,
        "chunk_count": summary.chunk_count,
        "collection_name": summary.collection_name,
        "chroma_dir": str(summary.chroma_dir.resolve()),
    }


if __name__ == "__main__":
    raise SystemExit(main())
