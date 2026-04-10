from typing import Annotated, List, Optional, TypedDict

from langchain_core.documents import Document

from src.schemas.agent_schemas import RelevanceStatus, VerificationReport


def merge_documents(
    old_docs: List[Document], new_docs: List[Document]
) -> List[Document]:
    """Merge new documents into existing state, deduplicating by chunk hash."""
    existing_hashes = {d.metadata.get("chunk_hash") for d in old_docs}
    return list(old_docs) + [
        doc for doc in new_docs if doc.metadata.get("chunk_hash") not in existing_hashes
    ]


class AgentState(TypedDict):
    """Shared state maintained throughout the LangGraph workflow."""

    question: str
    # candidate_documents is written exactly once per query by node_retrieve
    # and never re-written, so the default last-write-wins merge is correct.
    # No reducer annotation needed.
    candidate_documents: List[Document]
    # documents is annotated with the merge_documents reducer for historical
    # reasons (the original design contemplated multiple writers across the
    # self-correction loop). Today only node_rerank writes to it, once per
    # query — merge_documents([], reranked) == reranked, so the reducer is
    # harmless. Left in place to avoid touching unrelated behavior.
    documents: Annotated[List[Document], merge_documents]
    relevance_status: RelevanceStatus
    draft_answer: Optional[str]
    verification: Optional[VerificationReport]
    iterations: int
    error: Optional[str]
