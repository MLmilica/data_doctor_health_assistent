"""Parse clinical markdown documents into structured sections."""

import re
from pathlib import Path

from pydantic import BaseModel, Field

from config import settings

SECTION_HEADER_PATTERN = re.compile(r"^##\s+(.+)$", re.MULTILINE)
DIAGNOSIS_CODE_PATTERN = re.compile(r"\*\*([A-Z]\d{2}(?:\.\d+)?)\*\*")


class DocumentSection(BaseModel):
    source_file: str
    section_name: str
    content: str
    diagnosis_codes: list[str] = Field(default_factory=list)


class ParsedDocument(BaseModel):
    source_file: str
    title: str | None = None
    sections: list[DocumentSection] = Field(default_factory=list)


class DocumentCorpusSummary(BaseModel):
    document_count: int
    section_count: int
    section_names: dict[str, int]
    average_sections_per_document: float


def _extract_title(header_text: str) -> str | None:
    for line in header_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip()
    return None


def _extract_diagnosis_codes(content: str) -> list[str]:
    return sorted(set(DIAGNOSIS_CODE_PATTERN.findall(content)))


def parse_document(path: Path) -> ParsedDocument:
    text = path.read_text(encoding="utf-8")
    matches = list(SECTION_HEADER_PATTERN.finditer(text))

    title = _extract_title(text[: matches[0].start()] if matches else text)
    sections: list[DocumentSection] = []

    for index, match in enumerate(matches):
        section_name = match.group(1).strip()
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        content = text[start:end].strip()

        sections.append(
            DocumentSection(
                source_file=path.name,
                section_name=section_name,
                content=content,
                diagnosis_codes=_extract_diagnosis_codes(content),
            )
        )

    return ParsedDocument(
        source_file=path.name,
        title=title,
        sections=sections,
    )


def parse_documents_directory(directory: Path | None = None) -> list[ParsedDocument]:
    documents_dir = directory or settings.documents_dir
    if not documents_dir.exists():
        raise FileNotFoundError(f"Documents directory not found: {documents_dir}")

    documents = [
        parse_document(path)
        for path in sorted(documents_dir.glob("*.md"))
    ]
    return documents


def summarize_document_corpus(documents: list[ParsedDocument]) -> DocumentCorpusSummary:
    section_names: dict[str, int] = {}
    section_count = 0

    for document in documents:
        section_count += len(document.sections)
        for section in document.sections:
            section_names[section.section_name] = (
                section_names.get(section.section_name, 0) + 1
            )

    document_count = len(documents)
    average_sections = section_count / document_count if document_count else 0.0

    return DocumentCorpusSummary(
        document_count=document_count,
        section_count=section_count,
        section_names=dict(sorted(section_names.items())),
        average_sections_per_document=round(average_sections, 2),
    )
