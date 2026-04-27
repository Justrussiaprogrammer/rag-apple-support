from dataclasses import dataclass, asdict
import numpy as np
from sentence_transformers import SentenceTransformer

@dataclass
class Document:
    doc_id: str
    title: str
    source_url: str
    text: str


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    doc_title: str
    section_title: str | None
    source_url: str
    text: str

@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float


def dataclass_to_dict(object):
    return asdict(object)


class EmbeddingModel:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def _encode(self, texts: list[str], batch_size: int) -> np.ndarray:
        vectors = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        return vectors.astype("float32")

    def encode_passages(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        prepared = [f"passage: {text}" for text in texts]
        return self._encode(prepared, batch_size=batch_size)

    def encode_queries(self, queries: list[str], batch_size: int = 32) -> np.ndarray:
        prepared = [f"query: {query}" for query in queries]
        return self._encode(prepared, batch_size=batch_size)
