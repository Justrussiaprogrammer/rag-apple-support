from functools import lru_cache
from fastapi import FastAPI

from app.config import settings
from app.rag.pipeline import RAGService
from app.rag.work_classes import RetrievedChunk
from app.schemas import (
    AskRequest,
    AskResponse,
    HealthResponse,
    RetrieveRequest,
    RetrieveResponse,
    Source,
)


app = FastAPI()


@lru_cache(maxsize=1)
def get_rag_service() -> RAGService:
    return RAGService()


def to_source(item: RetrievedChunk) -> Source:
    chunk = item.chunk
    return Source(
        chunk_id=chunk.chunk_id,
        doc_title=chunk.doc_title,
        section_title=chunk.section_title,
        source_url=chunk.source_url,
        score=round(item.score, 4),
        text=chunk.text,
    )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    index_exists = settings.faiss_index_path.exists()
    chunks_exists = settings.chunks_path.exists()
    status = "ok" if index_exists and chunks_exists else "index_not_built"
    return HealthResponse(
        status=status,
        index_exists=index_exists,
        chunks_exists=chunks_exists,
    )


@app.post("/retrieve", response_model=RetrieveResponse)
def retrieve(request: RetrieveRequest) -> RetrieveResponse:
    service = get_rag_service()
    results = service.retrieve(request.query, top_k=request.top_k)
    return RetrieveResponse(sources=[to_source(item) for item in results])


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    service = get_rag_service()
    answer, sources = service.ask(request.question)
    return AskResponse(
        answer=answer,
        sources=[to_source(item) for item in sources],
    )
