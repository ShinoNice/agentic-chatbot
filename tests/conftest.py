"""Shared pytest fixtures for the agentic-chatbot test suite.

Currently scoped to fixtures the reranker tests need. Will grow as
the test suite grows.
"""

from __future__ import annotations

from typing import List

import pytest
from langchain_core.documents import Document


@pytest.fixture
def sample_documents() -> List[Document]:
    """Four small Documents with distinct content for reranker tests."""
    return [
        Document(
            page_content="DeepSeek-R1 uses Group Relative Policy Optimization (GRPO).",
            metadata={
                "source": "deepseek_paper.pdf",
                "page_number": 4,
                "chunk_index": 0,
            },
        ),
        Document(
            page_content="Cloudflare's 12th gen servers deliver 84% more RPS per kilowatt.",
            metadata={
                "source": "cloudflare_news.pdf",
                "page_number": 1,
                "chunk_index": 0,
            },
        ),
        Document(
            page_content="The relevance auditor returns NO_MATCH when context is unrelated.",
            metadata={
                "source": "internal_docs.pdf",
                "page_number": 12,
                "chunk_index": 3,
            },
        ),
        Document(
            page_content="DeepSeek-R1 achieves 97.3% on MATH-500, par with OpenAI-o1-1217.",
            metadata={
                "source": "deepseek_paper.pdf",
                "page_number": 7,
                "chunk_index": 2,
            },
        ),
    ]


class _FakeCrossEncoder:
    """Test double for sentence_transformers.CrossEncoder.

    Returns scores from a configurable list (default: descending integers).
    Tracks call counts so tests can assert load-once / predict-N semantics.
    """

    def __init__(
        self,
        model_name: str,
        scores: list | None = None,
        raise_on_init: bool = False,
    ):
        if raise_on_init:
            raise RuntimeError(f"simulated load failure for {model_name}")
        self.model_name = model_name
        self._scores = scores
        self.predict_calls = 0
        self.predict_should_raise = False

    def predict(self, pairs):
        self.predict_calls += 1
        if self.predict_should_raise:
            raise RuntimeError("simulated predict failure")
        if self._scores is not None:
            return self._scores[: len(pairs)]
        return list(range(len(pairs), 0, -1))


@pytest.fixture
def fake_cross_encoder():
    """Return the _FakeCrossEncoder class itself, for tests to instantiate or patch with."""
    return _FakeCrossEncoder
