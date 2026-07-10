"""Chroma vector store for clinical document sections."""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import Any, Literal, overload

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.utils import embedding_functions
from pydantic import BaseModel, Field

from config import settings
from data.document_parser import DocumentSection, ParsedDocument, parse_documents_directory
from schemas.citation import RetrievedChunk

_SECTION_SLUG_PATTERN = re.compile(r"[^a-z0-9]+")


class IndexSummary(BaseModel):
    """Result of indexing markdown documents into Chroma."""

    document_count: int
    chunk_count: int
    collection_name: str
    chroma_dir: Path


class DocumentVectorStore:
    """Index and search parsed markdown sections in a persistent Chroma collection."""

    def __init__(
        self,
        *,
        chroma_dir: Path | None = None,
        documents_dir: Path | None = None,
        collection_name: str | None = None,
        embedding_function: embedding_functions.EmbeddingFunction[Any] | None = None,
    ) -> None:
        self._chroma_dir = chroma_dir or settings.chroma_dir
        self._documents_dir = documents_dir or settings.documents_dir
        self._collection_name = collection_name or settings.rag_collection_name
        self._embedding_function = embedding_function
        self._client = chromadb.PersistentClient(path=str(self._chroma_dir))
        self._collection: Collection | None = None

    @property
    def collection_name(self) -> str:
        return self._collection_name

    @property
    def chroma_dir(self) -> Path:
        return self._chroma_dir

    def chunk_count(self) -> int:
        collection = self._get_collection(create_if_missing=False)
        if collection is None:
            return 0
        return collection.count()

    def is_indexed(self) -> bool:
        return self.chunk_count() > 0

    def index_documents(self, *, reset: bool = False) -> IndexSummary:
        """Parse markdown files and upsert one chunk per document section."""
        documents = parse_documents_directory(self._documents_dir)
        records = build_chunk_records(documents)
        if not records:
            raise ValueError(f"No document sections found in {self._documents_dir}")

        if reset and self._collection_exists():
            self._client.delete_collection(self._collection_name)
            self._collection = None

        collection = self._get_collection(create_if_missing=True)
        batch_size = settings.rag_index_batch_size
        for start in range(0, len(records), batch_size):
            batch = records[start : start + batch_size]
            collection.upsert(
                ids=[record["id"] for record in batch],
                documents=[record["document"] for record in batch],
                metadatas=[record["metadata"] for record in batch],
            )

        return IndexSummary(
            document_count=len(documents),
            chunk_count=len(records),
            collection_name=self._collection_name,
            chroma_dir=self._chroma_dir,
        )

    def search(self, query: str, *, k: int | None = None) -> list[RetrievedChunk]:
        """Return the top-k most similar document sections for a query."""
        if not query.strip():
            return []

        limit = k or settings.rag_top_k
        collection = self._get_collection(create_if_missing=False)
        if collection is None or collection.count() == 0:
            return []

        result = collection.query(
            query_texts=[query.strip()],
            n_results=min(limit, collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        ids = (result.get("ids") or [[]])[0]
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]

        chunks: list[RetrievedChunk] = []
        for chunk_id, content, metadata, distance in zip(
            ids,
            documents,
            metadatas,
            distances,
            strict=True,
        ):
            if content is None or metadata is None or distance is None:
                continue
            chunks.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    source_file=str(metadata.get("source_file", "")),
                    section_name=str(metadata.get("section_name", "")),
                    content=content,
                    score=_distance_to_score(float(distance)),
                )
            )
        return chunks

    def _collection_exists(self) -> bool:
        existing = {collection.name for collection in self._client.list_collections()}
        return self._collection_name in existing

    @overload
    def _get_collection(self, *, create_if_missing: Literal[True]) -> Collection: ...

    @overload
    def _get_collection(self, *, create_if_missing: Literal[False]) -> Collection | None: ...

    def _get_collection(self, *, create_if_missing: bool) -> Collection | None:
        if self._collection is not None:
            return self._collection

        if not create_if_missing and not self._collection_exists():
            return None

        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            embedding_function=self._embedding_function or build_embedding_function(),
            metadata={"hnsw:space": "cosine"},
        )
        return self._collection


def build_chunk_id(source_file: str, section_name: str) -> str:
    """Stable chunk id used as the Chroma primary key."""
    slug = _SECTION_SLUG_PATTERN.sub("-", section_name.lower()).strip("-") or "section"
    return f"{source_file}::{slug}"


def build_chunk_document(section: DocumentSection, *, title: str | None = None) -> str:
    """Text stored in Chroma and later shown to the LLM."""
    header = f"# {title}\n" if title else ""
    return (
        f"{header}"
        f"Source: {section.source_file}\n"
        f"Section: {section.section_name}\n\n"
        f"{section.content.strip()}"
    ).strip()


def build_chunk_records(documents: list[ParsedDocument]) -> list[dict[str, Any]]:
    """Flatten parsed documents into Chroma upsert payloads."""
    records: list[dict[str, Any]] = []
    for document in documents:
        for section in document.sections:
            content = section.content.strip()
            if not content:
                continue
            metadata: dict[str, str] = {
                "source_file": section.source_file,
                "section_name": section.section_name,
            }
            if section.diagnosis_codes:
                metadata["diagnosis_codes"] = ",".join(section.diagnosis_codes)
            records.append(
                {
                    "id": build_chunk_id(section.source_file, section.section_name),
                    "document": build_chunk_document(section, title=document.title),
                    "metadata": metadata,
                }
            )
    return records


def build_embedding_function() -> embedding_functions.EmbeddingFunction[Any]:
    """OpenAI embeddings for production indexing and retrieval."""
    api_key = settings.openai_api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY is required for document embeddings. "
            "Set it in .env before running index_documents."
        )
    return embedding_functions.OpenAIEmbeddingFunction(
        api_key=api_key,
        model_name=settings.rag_embedding_model,
    )


def build_test_embedding_function() -> embedding_functions.EmbeddingFunction[Any]:
    """Deterministic local embeddings for unit tests (no network)."""

    class _DeterministicEmbeddingFunction(embedding_functions.EmbeddingFunction[list[str]]):
        def name(self) -> str:
            return "deterministic-test"

        def __call__(self, input: list[str]) -> list[list[float]]:
            return [_deterministic_embedding(text) for text in input]

    return _DeterministicEmbeddingFunction()


def _deterministic_embedding(text: str, *, dimensions: int = 384) -> list[float]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    values: list[float] = []
    while len(values) < dimensions:
        digest = hashlib.sha256(digest).digest()
        for byte in digest:
            values.append((byte / 255.0) * 2.0 - 1.0)
            if len(values) >= dimensions:
                break
    return values[:dimensions]


def _distance_to_score(distance: float) -> float:
    """Convert Chroma cosine distance to a similarity score in [0, 1]."""
    return max(0.0, min(1.0, 1.0 - distance))


_vectorstore: DocumentVectorStore | None = None


def get_vectorstore() -> DocumentVectorStore:
    """Return a process-wide vector store singleton."""
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = DocumentVectorStore()
    return _vectorstore


def reset_vectorstore() -> None:
    """Clear the singleton (mainly for tests)."""
    global _vectorstore
    _vectorstore = None
