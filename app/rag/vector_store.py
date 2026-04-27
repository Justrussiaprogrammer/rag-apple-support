import json
import faiss

from app.rag.work_classes import Chunk, dataclass_to_dict


def build_faiss_index(vectors):
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index


def save_faiss_index(index, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(path))


def load_faiss_index(path):
    if not path.exists():
        raise ValueError(
            f"Нет сохраненного индекса FAISS: {path}. "
            "Сначала запустите: python scripts/build_index.py"
        )
    return faiss.read_index(str(path))


def save_chunks(chunks, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(dataclass_to_dict(chunk), ensure_ascii=False) + "\n")


def load_chunks(path):
    if not path.exists():
        raise ValueError(
            f"Чанки отсутствуют на месте: {path}. "
            "Сначала запустите: python scripts/build_index.py"
        )

    chunks = list()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            chunks.append(Chunk(**row))

    return chunks
