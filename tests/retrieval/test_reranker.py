"""Unit tests for src/retrieval/reranker.py.

These tests monkeypatch sentence_transformers.CrossEncoder so no real
model is ever loaded. They verify the reranker contract (sort order,
truncation, edge cases, error propagation), not reranker quality.
Quality is what evaluation/eval_pipeline.py measures.
"""

from __future__ import annotations

import pytest
from langchain_core.documents import Document

from src.core.exceptions import RerankerError
from src.retrieval.reranker import BGEReranker


def _patch_cross_encoder(monkeypatch, fake_cls, **fake_kwargs):
    """Patch sentence_transformers.CrossEncoder with a configured fake instance.

    Returns the constructed fake instance so the test can introspect its state.
    """
    instance_holder = {}

    def _factory(model_name):
        inst = fake_cls(model_name, **fake_kwargs)
        instance_holder["instance"] = inst
        return inst

    monkeypatch.setattr("sentence_transformers.CrossEncoder", _factory)
    return instance_holder


def test_rerank_orders_by_score(monkeypatch, fake_cross_encoder, sample_documents):
    # Score doc[2] highest, doc[0] second, doc[3] third, doc[1] lowest.
    holder = _patch_cross_encoder(monkeypatch, fake_cross_encoder, scores=[5.0, 1.0, 9.0, 3.0])

    reranker = BGEReranker("BAAI/bge-reranker-base")
    result = reranker.rerank("any query", sample_documents, top_k=4)

    assert len(result) == 4
    assert result[0] is sample_documents[2]
    assert result[1] is sample_documents[0]
    assert result[2] is sample_documents[3]
    assert result[3] is sample_documents[1]
    assert holder["instance"].predict_calls == 1


def test_rerank_truncates_to_top_k(monkeypatch, fake_cross_encoder, sample_documents):
    _patch_cross_encoder(monkeypatch, fake_cross_encoder, scores=[1.0, 2.0, 3.0, 4.0])

    reranker = BGEReranker("BAAI/bge-reranker-base")
    result = reranker.rerank("any query", sample_documents, top_k=2)

    assert len(result) == 2
    # Highest scores are doc[3]=4.0 and doc[2]=3.0
    assert result[0] is sample_documents[3]
    assert result[1] is sample_documents[2]


def test_rerank_top_k_larger_than_input_returns_all(
    monkeypatch, fake_cross_encoder, sample_documents
):
    _patch_cross_encoder(monkeypatch, fake_cross_encoder, scores=[1.0, 4.0, 2.0, 3.0])

    reranker = BGEReranker("BAAI/bge-reranker-base")
    result = reranker.rerank("any query", sample_documents, top_k=10)

    assert len(result) == 4
    # Order: doc[1]=4.0, doc[3]=3.0, doc[2]=2.0, doc[0]=1.0
    assert result[0] is sample_documents[1]
    assert result[1] is sample_documents[3]
    assert result[2] is sample_documents[2]
    assert result[3] is sample_documents[0]


def test_rerank_empty_input_returns_empty(monkeypatch, fake_cross_encoder):
    holder = _patch_cross_encoder(monkeypatch, fake_cross_encoder)

    reranker = BGEReranker("BAAI/bge-reranker-base")
    result = reranker.rerank("any query", [], top_k=5)

    assert result == []
    # No model load should have been triggered.
    assert "instance" not in holder


def test_rerank_top_k_zero_returns_empty(monkeypatch, fake_cross_encoder, sample_documents):
    holder = _patch_cross_encoder(monkeypatch, fake_cross_encoder)

    reranker = BGEReranker("BAAI/bge-reranker-base")
    result = reranker.rerank("any query", sample_documents, top_k=0)

    assert result == []
    # No model load should have been triggered.
    assert "instance" not in holder


def test_rerank_stable_tie_breaking(monkeypatch, fake_cross_encoder, sample_documents):
    # All four documents tie at score 1.0.
    _patch_cross_encoder(monkeypatch, fake_cross_encoder, scores=[1.0, 1.0, 1.0, 1.0])

    reranker = BGEReranker("BAAI/bge-reranker-base")
    result = reranker.rerank("any query", sample_documents, top_k=4)

    # With Python's stable sort, identical scores preserve input order.
    assert result[0] is sample_documents[0]
    assert result[1] is sample_documents[1]
    assert result[2] is sample_documents[2]
    assert result[3] is sample_documents[3]


def test_load_failure_raises_reranker_error(monkeypatch, fake_cross_encoder):
    _patch_cross_encoder(monkeypatch, fake_cross_encoder, raise_on_init=True)

    reranker = BGEReranker("BAAI/bge-reranker-base")

    with pytest.raises(RerankerError, match="Failed to load cross-encoder"):
        reranker.rerank(
            "any query",
            [Document(page_content="foo", metadata={})],
            top_k=1,
        )


def test_load_failure_is_sticky(monkeypatch, fake_cross_encoder):
    init_calls = {"count": 0}

    def _counting_factory(model_name):
        init_calls["count"] += 1
        raise RuntimeError(f"simulated load failure for {model_name}")

    monkeypatch.setattr("sentence_transformers.CrossEncoder", _counting_factory)

    reranker = BGEReranker("BAAI/bge-reranker-base")
    docs = [Document(page_content="foo", metadata={})]

    # First call: triggers the (failing) load.
    with pytest.raises(RerankerError, match="Failed to load"):
        reranker.rerank("query", docs, top_k=1)
    assert init_calls["count"] == 1

    # Second call: must NOT re-attempt the load. Different error message
    # ("previously failed to load") confirms the sticky branch fired.
    with pytest.raises(RerankerError, match="previously failed to load"):
        reranker.rerank("query", docs, top_k=1)
    assert init_calls["count"] == 1


def test_predict_failure_raises_reranker_error(monkeypatch, fake_cross_encoder, sample_documents):
    holder = _patch_cross_encoder(monkeypatch, fake_cross_encoder)

    reranker = BGEReranker("BAAI/bge-reranker-base")

    # First call: triggers the load (fake), succeeds.
    reranker.rerank("query", sample_documents, top_k=2)
    assert holder["instance"].predict_calls == 1

    # Now flip the predict to raise.
    holder["instance"].predict_should_raise = True

    with pytest.raises(RerankerError, match="predict\\(\\) failed"):
        reranker.rerank("query", sample_documents, top_k=2)


def test_predict_failure_is_not_sticky(monkeypatch, fake_cross_encoder, sample_documents):
    holder = _patch_cross_encoder(monkeypatch, fake_cross_encoder)

    reranker = BGEReranker("BAAI/bge-reranker-base")

    # First call succeeds (load + 1 predict).
    reranker.rerank("query", sample_documents, top_k=2)
    assert holder["instance"].predict_calls == 1

    # Flip predict to raise; second call fails.
    holder["instance"].predict_should_raise = True
    with pytest.raises(RerankerError):
        reranker.rerank("query", sample_documents, top_k=2)
    assert holder["instance"].predict_calls == 2

    # Flip predict back to working; third call should succeed
    # (instance not poisoned by the prior predict failure).
    holder["instance"].predict_should_raise = False
    result = reranker.rerank("query", sample_documents, top_k=2)
    assert len(result) == 2
    assert holder["instance"].predict_calls == 3
