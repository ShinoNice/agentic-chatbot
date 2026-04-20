"""Tests for HybridSearcher caching + vector-only fallback.

The BM25 index build is mocked so tests are fast. The key invariants:
- Second ``get_retriever()`` call hits the cache (no rebuild).
- Missing cache AND missing documents → fall back to vector-only.
- Async variant offloads to a worker thread and matches sync behaviour.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

from src.retrieval.hybrid_search import HybridSearcher


@pytest.fixture
def tiny_docs():
    return [
        Document(page_content="alpha", metadata={"source": "a.pdf"}),
        Document(page_content="beta", metadata={"source": "b.pdf"}),
    ]


@pytest.fixture
def fake_vector_store():
    """Vector store whose as_retriever() returns a real Runnable.

    EnsembleRetriever validates its children via Pydantic and rejects plain
    MagicMocks, so we wrap a no-op callable in RunnableLambda.
    """
    vs = MagicMock()
    vs.as_retriever.return_value = RunnableLambda(lambda _q: [])
    return vs


def test_get_retriever_caches_result(tmp_path, monkeypatch, fake_vector_store, tiny_docs):
    """Successive calls return the same retriever without rebuilding the BM25 index."""
    from src.core.config_loader import settings as live_settings

    monkeypatch.setattr(live_settings.rag, "cache_dir", str(tmp_path))

    searcher = HybridSearcher(fake_vector_store, documents=tiny_docs, k=3)
    r1 = searcher.get_retriever()
    r2 = searcher.get_retriever()

    assert r1 is r2, "HybridSearcher must cache the built retriever"


def test_vector_only_fallback_when_no_documents_and_no_cache(
    tmp_path, monkeypatch, fake_vector_store
):
    """No seed docs AND no cache file on disk → degrade to vector-only retrieval."""
    from src.core.config_loader import settings as live_settings

    monkeypatch.setattr(live_settings.rag, "cache_dir", str(tmp_path))

    searcher = HybridSearcher(fake_vector_store, documents=None, k=3)
    retriever = searcher.get_retriever()

    # Fallback should hand back the vector retriever directly, not an
    # EnsembleRetriever built on top of it.
    assert retriever is fake_vector_store.as_retriever.return_value


def test_bm25_cache_is_persisted_and_reloadable(
    tmp_path, monkeypatch, fake_vector_store, tiny_docs
):
    """Building the BM25 index writes a cache file that a fresh searcher can load.

    Smoke-tests the round-trip through JSON (content, metadata preservation).
    """
    from src.core.config_loader import settings as live_settings

    monkeypatch.setattr(live_settings.rag, "cache_dir", str(tmp_path))

    # First searcher builds + persists.
    first = HybridSearcher(fake_vector_store, documents=tiny_docs, k=2)
    first.get_retriever()

    cache_file = Path(tmp_path) / "bm25_documents.json"
    assert cache_file.exists(), "BM25 build must write the cache JSON"

    with open(cache_file, encoding="utf-8") as f:
        payload = json.load(f)
    assert len(payload) == 2
    assert {p["page_content"] for p in payload} == {"alpha", "beta"}

    # Second searcher (no docs) should load from cache, not fall back.
    second = HybridSearcher(fake_vector_store, documents=None, k=2)
    retriever = second.get_retriever()

    # Not the raw vector retriever: we got the hybrid ensemble back.
    assert retriever is not fake_vector_store.as_retriever.return_value


async def test_aget_retriever_matches_sync_result(
    tmp_path, monkeypatch, fake_vector_store, tiny_docs
):
    """Async wrapper returns the same retriever the sync call would."""
    from src.core.config_loader import settings as live_settings

    monkeypatch.setattr(live_settings.rag, "cache_dir", str(tmp_path))

    searcher = HybridSearcher(fake_vector_store, documents=tiny_docs, k=3)
    r_async = await searcher.aget_retriever()
    r_sync = searcher.get_retriever()

    assert r_async is r_sync
