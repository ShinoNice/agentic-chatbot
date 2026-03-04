from typing import Annotated, List, Optional, TypedDict

from langchain_core.documents import Document

from src.schemas.agent_schemas import RelevanceStatus, VerificationReport


def merge_documents(old_docs: List[Document], new_docs: List[Document]) -> List[Document]:
    """Merge new documents into existing state, deduplicating by chunk hash."""
    existing_hashes = {d.metadata.get("chunk_hash") for d in old_docs}
    return list(old_docs) + [
        doc for doc in new_docs
        if doc.metadata.get("chunk_hash") not in existing_hashes
    ]


class AgentState(TypedDict):
    """Shared state maintained throughout the LangGraph workflow."""

    question: str
    documents: Annotated[List[Document], merge_documents]
    relevance_status: RelevanceStatus
    draft_answer: Optional[str]
    verification: Optional[VerificationReport]
    iterations: int
    error: Optional[str]
