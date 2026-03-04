from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class DocumentMetadata(BaseModel):
    """Standardized metadata for a document chunk (Chroma/Pinecone)."""

    source: str = Field(description="Original file name or path.")
    page_number: Optional[int] = Field(
        default=None, description="Page in the source PDF.")
    chunk_hash: str = Field(
        description="SHA-256 content hash for deduplication.")
    parser_used: str = Field(
        description="Text extraction engine (e.g. 'pymupdf', 'docling').")
    processed_at: str = Field(
        default_factory=lambda: datetime.now().isoformat())


class SearchResult(BaseModel):
    """Single result returned from HybridSearcher."""

    content: str = Field(description="Text content of the retrieved chunk.")
    metadata: DocumentMetadata
    score: float = Field(
        description="Relevance score (cosine similarity or BM25 rank).")


class IngestionTask(BaseModel):
    """Tracks an ingestion job in the pipeline."""

    task_id: str
    files_processed: List[str]
    total_chunks: int
    status: str = Field(default="pending",
                        pattern="^(pending|completed|failed)$")
    error_message: Optional[str] = None


class RetrievalContext(BaseModel):
    """Final payload passed from the retrieval layer to the agent layer."""

    query: str
    top_k: int
    results: List[SearchResult]

    def to_context_string(self) -> str:
        """Flatten results into a single string for LLM prompt injection."""
        return "\n\n".join(
            f"Source: {res.metadata.source}\nContent: {res.content}"
            for res in self.results
        )
