"""Tests for the Chroma document vector store."""

from __future__ import annotations

from pathlib import Path

import pytest

from data.vectorstore import (
    DocumentVectorStore,
    build_chunk_id,
    build_chunk_records,
    build_test_embedding_function,
    reset_vectorstore,
)
from data.document_parser import parse_document


@pytest.fixture
def sample_documents_dir(tmp_path: Path) -> Path:
    docs_dir = tmp_path / "documents"
    docs_dir.mkdir()
    (docs_dir / "copd_guideline.md").write_text(
        """# COPD Guideline

## Exercise
Patients with COPD should perform regular low-intensity exercise.

## Smoking
Smoking cessation is strongly recommended for all COPD patients.
""",
        encoding="utf-8",
    )
    (docs_dir / "alt_monitoring.md").write_text(
        """# ALT Monitoring

## Recheck intervals
Recheck ALT levels 4-6 weeks after a significant dose change.
""",
        encoding="utf-8",
    )
    return docs_dir


@pytest.fixture
def vectorstore(tmp_path: Path, sample_documents_dir: Path) -> DocumentVectorStore:
    reset_vectorstore()
    store = DocumentVectorStore(
        chroma_dir=tmp_path / "chroma",
        documents_dir=sample_documents_dir,
        collection_name="test_clinical_documents",
        embedding_function=build_test_embedding_function(),
    )
    yield store
    reset_vectorstore()


def test_build_chunk_id_slugifies_section_name() -> None:
    assert build_chunk_id("copd_guideline.md", "Exercise Frequency") == (
        "copd_guideline.md::exercise-frequency"
    )


def test_build_chunk_records_from_parsed_document(sample_documents_dir: Path) -> None:
    document = parse_document(sample_documents_dir / "copd_guideline.md")
    records = build_chunk_records([document])
    assert len(records) == 2
    assert records[0]["id"] == "copd_guideline.md::exercise"
    assert "regular low-intensity exercise" in records[0]["document"]


def test_index_and_search_documents(vectorstore: DocumentVectorStore) -> None:
    summary = vectorstore.index_documents()
    assert summary.document_count == 2
    assert summary.chunk_count == 3
    assert vectorstore.is_indexed()

    results = vectorstore.search("What does the COPD guideline say about exercise?", k=3)
    assert len(results) == 3
    assert any(result.section_name == "Exercise" for result in results)
    assert all(0.0 <= result.score <= 1.0 for result in results)


def test_index_documents_reset_replaces_collection(
    vectorstore: DocumentVectorStore,
    sample_documents_dir: Path,
) -> None:
    vectorstore.index_documents()
    assert vectorstore.chunk_count() == 3

    (sample_documents_dir / "copd_guideline.md").unlink()
    summary = vectorstore.index_documents(reset=True)
    assert summary.chunk_count == 1
    assert vectorstore.chunk_count() == 1
