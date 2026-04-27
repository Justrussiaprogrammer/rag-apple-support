from app.rag.vector_store import load_chunks, load_faiss_index
from app.rag.work_classes import EmbeddingModel, RetrievedChunk
from app.config import settings

class Retriever:
    def __init__(self):
        self.chunks = load_chunks(settings.chunks_path)
        self.index = load_faiss_index(settings.faiss_index_path)
        self.embedding_model = EmbeddingModel(settings.embedding_model_name)

        if self.index.ntotal != len(self.chunks):
            raise ValueError(
                f"FAISS index size ({self.index.ntotal}) != chunks count ({len(self.chunks)})"
            )

    def retrieve(self, query, top_k=3):
        query_vector = self.embedding_model.encode_queries([query], batch_size=1)
        scores, indices = self.index.search(query_vector, top_k)

        results = list()
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            results.append(RetrievedChunk(chunk=self.chunks[int(idx)], score=float(score)))

        return results