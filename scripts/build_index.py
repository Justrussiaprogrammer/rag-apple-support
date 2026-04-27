import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from app.config import settings
from app.rag.chunking import chunk_documents
from app.rag.work_classes import EmbeddingModel, Document
from app.rag.vector_store import build_faiss_index, save_chunks, save_faiss_index


def load_documents(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Файл с документами не найден: {path}. "
            "Сначала запустите: python scripts/fetch_docs.py"
        )

    docs: list[Document] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            docs.append(
                Document(
                    doc_id=row["doc_id"],
                    title=row["title"],
                    source_url=row["source_url"],
                    text=row["text"],
                )
            )

    if not docs:
        raise ValueError(f"В файле {path} нет документов.")

    return docs


def main():
    documents = load_documents(settings.raw_docs_path)
    print(f"Всего документов: {len(documents)}")

    chunks = chunk_documents(documents)
    print(f"Всего чанков: {len(chunks)}")

    model = EmbeddingModel(settings.embedding_model_name)
    vectors = model.encode_passages(
        [chunk.text for chunk in chunks],
        batch_size=settings.embedding_batch_size,
    )
    print(f"Размерность эмбеддинга: {vectors.shape}")

    index = build_faiss_index(vectors)
    save_faiss_index(index, settings.faiss_index_path)
    save_chunks(chunks, settings.chunks_path)

    print(f"Saved FAISS index: {settings.faiss_index_path}")
    print(f"Saved chunks metadata: {settings.chunks_path}")


if __name__ == "__main__":
    main()
