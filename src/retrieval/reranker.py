"""Cross-encoder reranker for the agentic-chatbot retrieval pipeline.

Lazy-loads BAAI/bge-reranker-base via sentence-transformers on first
use. The orchestrator catches RerankerError and falls back to the
unranked candidates so a broken reranker degrades gracefully rather
than failing the request.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from langchain_core.documents import Document

from src.core.exceptions import RerankerError
from src.core.logger import logger


class BaseReranker(ABC):
    """Abstract base class for rerankers. One method, by design."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: int,
    ) -> List[Document]:
        """Return at most top_k documents ordered by relevance to query."""


class BGEReranker(BaseReranker):
    """Cross-encoder reranker using a local sentence-transformers model.

    Lazy-loads on first call so importing this module is cheap and
    tests can monkeypatch sentence_transformers.CrossEncoder before any
    real model is touched.
    """

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None
        self._load_failed = False

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        if self._load_failed:
            raise RerankerError(
                f"Reranker model {self.model_name} previously failed to load."
            )
        try:
            # Deferred import — DO NOT HOIST to module level. Two reasons:
            # (1) keeps torch/transformers off the import path entirely when
            #     rerank.enabled=False (baseline eval reproducibility);
            # (2) tests monkeypatch sentence_transformers.CrossEncoder and
            #     rely on this being a runtime attribute lookup against the
            #     sentence_transformers module — a top-level "from ... import
            #     CrossEncoder" would bind the name in this module at import
            #     time and the monkeypatch would no longer intercept it.
            from sentence_transformers import CrossEncoder

            logger.info(f"Loading cross-encoder: {self.model_name}")
            self._model = CrossEncoder(self.model_name)
        except Exception as e:
            self._load_failed = True
            raise RerankerError(
                f"Failed to load cross-encoder {self.model_name}: {e}"
            ) from e

    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_k: int,
    ) -> List[Document]:
        if not documents or top_k <= 0:
            return []

        self._ensure_loaded()

        pairs = [(query, doc.page_content) for doc in documents]
        try:
            scores = self._model.predict(pairs)
        except Exception as e:
            raise RerankerError(f"Reranker predict() failed: {e}") from e

        # Stable sort: ties preserve the original retriever order.
        ranked = sorted(
            enumerate(documents),
            key=lambda pair: scores[pair[0]],
            reverse=True,
        )
        return [doc for _, doc in ranked[:top_k]]
