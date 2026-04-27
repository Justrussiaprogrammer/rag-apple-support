import re
from app.rag.work_classes import Document, Chunk


HEADING_RE = re.compile(r"^(#{1,3})\s+(.+)$")


def _normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _split_long_text(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    current = ""

    for paragraph in paragraphs:
        if len(current) + len(paragraph) + 2 <= max_chars:
            current = f"{current}\n\n{paragraph}".strip()
            continue

        if current:
            chunks.append(current)

        if len(paragraph) <= max_chars:
            current = paragraph
        else:
            start = 0
            while start < len(paragraph):
                end = start + max_chars
                chunks.append(paragraph[start:end].strip())
                start = max(0, end - overlap_chars)
            current = ""

    if current:
        chunks.append(current)

    if overlap_chars > 0 and len(chunks) > 1:
        with_overlap = list()
        for idx, chunk in enumerate(chunks):
            if idx == 0:
                with_overlap.append(chunk)
            else:
                prev_tail = chunks[idx - 1][-overlap_chars:].strip()
                with_overlap.append(f"{prev_tail}\n\n{chunk}".strip())
        return with_overlap

    return chunks


def _split_markdown_sections(text: str, default_section: str) -> list[tuple[str, str]]:
    sections: list[tuple[str, list[str]]] = []
    current_title = default_section
    current_lines: list[str] = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        match = HEADING_RE.match(line)

        if match:
            if current_lines:
                sections.append((current_title, current_lines))
                current_lines = []
            current_title = match.group(2).strip()
            continue

        if line:
            current_lines.append(line)
        else:
            current_lines.append("")

    if current_lines:
        sections.append((current_title, current_lines))

    result: list[tuple[str, str]] = []
    for title, lines in sections:
        section_text = _normalize_text("\n".join(lines))
        if section_text:
            result.append((title, section_text))

    return result


def chunk_documents(
    documents: list[Document],
    max_chars: int = 1200,
    min_chars: int = 120,
    overlap_chars: int = 150,
) -> list[Chunk]:
    chunks: list[Chunk] = []

    for doc in documents:
        sections = _split_markdown_sections(doc.text, default_section=doc.title)
        local_idx = 0

        for section_title, section_text in sections:
            parts = (
                [section_text]
                if len(section_text) <= max_chars
                else _split_long_text(section_text, max_chars=max_chars, overlap_chars=overlap_chars)
            )

            for part in parts:
                part = _normalize_text(part)
                if len(part) < min_chars:
                    continue

                local_idx += 1
                chunks.append(
                    Chunk(
                        chunk_id=f"{doc.doc_id}_{local_idx:04d}",
                        doc_id=doc.doc_id,
                        doc_title=doc.title,
                        section_title=section_title,
                        source_url=doc.source_url,
                        text=part,
                    )
                )

    if not chunks:
        raise ValueError("После чанкинга не получилось ни одного чанка.")

    return chunks
