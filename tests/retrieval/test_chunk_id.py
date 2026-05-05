from langchain_core.documents import Document

from src.retrieval.hybrid_search import ensure_chunk_ids


def test_ensure_chunk_ids_copies_chunk_hash_when_missing():
    doc = Document(page_content="x", metadata={"chunk_hash": "abc"})
    out = ensure_chunk_ids([doc])
    assert out[0].metadata["chunk_id"] == "abc"


def test_ensure_chunk_ids_synthesizes_when_no_hash():
    doc = Document(page_content="x", metadata={"source": "a.pdf", "page_number": 1})
    out = ensure_chunk_ids([doc])
    assert out[0].metadata["chunk_id"]
    out2 = ensure_chunk_ids([Document(page_content="x", metadata={"source": "a.pdf", "page_number": 1})])
    assert out[0].metadata["chunk_id"] == out2[0].metadata["chunk_id"]


def test_ensure_chunk_ids_preserves_existing_chunk_id():
    doc = Document(page_content="x", metadata={"chunk_id": "preset"})
    out = ensure_chunk_ids([doc])
    assert out[0].metadata["chunk_id"] == "preset"
