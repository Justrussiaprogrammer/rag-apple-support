from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, examples=["Как обновить iPhone?"])


class RetrieveRequest(BaseModel):
    query: str = Field(..., min_length=3, examples=["Как создать резервную копию iPhone?"])
    top_k: int = Field(default=3, ge=1, le=10)


class Source(BaseModel):
    chunk_id: str
    doc_title: str
    section_title: str | None = None
    source_url: str
    score: float
    text: str


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]


class RetrieveResponse(BaseModel):
    sources: list[Source]


class HealthResponse(BaseModel):
    status: str
    index_exists: bool
    chunks_exists: bool
