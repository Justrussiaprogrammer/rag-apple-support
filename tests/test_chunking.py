from app.rag.chunking import chunk_documents
from app.rag.work_classes import Document


def test_chunk_documents_basic():
    doc = Document(
        doc_id="doc1",
        title="Тестовый документ",
        source_url="https://example.com",
        text="# Тестовый документ\n\n## Раздел 1\n\n" + "Это тестовый текст. " * 50,
    )

    chunks = chunk_documents([doc], max_chars=500, min_chars=50, overlap_chars=50)

    assert len(chunks) >= 1
    assert chunks[0].doc_id == "doc1"
    assert chunks[0].doc_title == "Тестовый документ"
    assert chunks[0].source_url == "https://example.com"
