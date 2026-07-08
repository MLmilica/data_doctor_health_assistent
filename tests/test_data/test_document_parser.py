"""Tests for clinical document parser."""

from data.document_parser import parse_documents_directory, summarize_document_corpus
from config import settings


def test_parse_all_documents() -> None:
    documents = parse_documents_directory()
    assert len(documents) == 1050


def test_documents_have_core_sections() -> None:
    documents = parse_documents_directory()
    summary = summarize_document_corpus(documents)

    assert summary.section_count > 0
    assert "Diagnosis" in summary.section_names
    assert "Medications" in summary.section_names
    assert "Treatment Plan" in summary.section_names


def test_first_document_sections_are_non_empty() -> None:
    documents = parse_documents_directory(settings.documents_dir)
    first_document = documents[0]

    assert first_document.source_file.endswith(".md")
    assert len(first_document.sections) > 0
    assert all(section.content.strip() for section in first_document.sections)
