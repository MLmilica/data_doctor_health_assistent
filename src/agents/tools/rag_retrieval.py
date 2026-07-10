"""Read-only retrieval tool over the indexed clinical document vector store."""

from __future__ import annotations

from data.vectorstore import DocumentVectorStore, get_vectorstore, reset_vectorstore
from schemas.citation import RetrievedChunk


def retrieve_chunks(
    query: str,
    *,
    k: int | None = None,
    store: DocumentVectorStore | None = None,
) -> list[RetrievedChunk]:
    """Return top-k document sections similar to the user query."""
    vectorstore = store or get_vectorstore()
    return vectorstore.search(query, k=k)


__all__ = ["retrieve_chunks", "get_vectorstore", "reset_vectorstore"]
