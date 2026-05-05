from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path

from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from src.core.config_loader import settings
from src.core.logger import logger


def ensure_chunk_ids(docs: list) -> list:
    """Mutate-in-place + return docs with metadata['chunk_id'] populated.

    Order of preference:
      1. existing chunk_id (no-op)
      2. existing chunk_hash → copy to chunk_id
      3. sha1(source|page_number|page_content[:64]) as a deterministic fallback
    """
    for doc in docs:
        meta = doc.metadata or {}
        if meta.get("chunk_id"):
            continue
        if meta.get("chunk_hash"):
            meta["chunk_id"] = meta["chunk_hash"]
        else:
            seed = (
                f"{meta.get('source', '')}|{meta.get('page_number', '')}|"
                f"{(doc.page_content or '')[:64]}"
            )
            meta["chunk_id"] = hashlib.sha1(seed.encode()).hexdigest()
        doc.metadata = meta
    return docs


class HybridSearcher:
    """Combines dense (vector) and sparse (BM25) retrieval into a hybrid search."""

    def __init__(
        self,
        vector_store,
        documents: list[Document] | None = None,
        k: int | None = None,
    ):
        self.vector_store = vector_store
        self.documents = documents
        self.top_k = k if k is not None else settings.rag.top_k
        self.weights = settings.rag.hybrid_weights

        self.cache_dir = Path(settings.rag.cache_dir)
        self.bm25_path = self.cache_dir / "bm25_documents.json"

        # Cache the built retriever so repeated queries reuse the in-memory BM25
        # index (which itself isn't cheap to rebuild from the cached docs JSON).
        self._retriever = None

    def get_retriever(self):
        """Return the hybrid ensemble retriever, or a vector-only fallback.

        Synchronous. Safe to call from sync contexts; from async contexts prefer
        :meth:`aget_retriever` so the first-build file I/O does not block the
        event loop.
        """
        if self._retriever is not None:
            return self._retriever

        vector_retriever = self.vector_store.as_retriever(
            search_kwargs={"k": self.top_k},
        )

        try:
            bm25_retriever = self._get_or_create_bm25()
        except ValueError:
            logger.warning(
                "BM25 index unavailable (no documents and no cache). "
                "Falling back to vector-only retrieval."
            )
            self._retriever = vector_retriever
            return self._retriever

        logger.info(f"Initializing Hybrid Search (Weights: {self.weights}, k: {self.top_k})")

        self._retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever],
            weights=self.weights,
        )
        return self._retriever

    async def aget_retriever(self):
        """Async variant — offloads the first-build sync file I/O to a worker thread.

        Callers in async code (LangGraph nodes, FastAPI routes) should use this
        so BM25 cache reads and JSON parsing do not block the event loop.
        """
        if self._retriever is not None:
            return self._retriever
        return await asyncio.to_thread(self.get_retriever)

    def _get_or_create_bm25(self) -> BM25Retriever:
        """Load BM25 documents from cache and rebuild, or build and persist a new index."""
        if self.bm25_path.exists():
            try:
                logger.info("Loading BM25 documents from cache.")
                with open(self.bm25_path, encoding="utf-8") as f:
                    data = json.load(f)
                docs = [
                    Document(page_content=d["page_content"], metadata=d["metadata"]) for d in data
                ]
                retriever = BM25Retriever.from_documents(docs)
                retriever.k = self.top_k
                return retriever
            except Exception as e:
                logger.warning(f"BM25 cache corrupted, rebuilding: {e}")

        if not self.documents:
            raise ValueError("Documents are required to build the initial BM25 index.")

        logger.info(f"Building new BM25 index for {len(self.documents)} chunks.")
        retriever = BM25Retriever.from_documents(self.documents)
        retriever.k = self.top_k

        try:
            data = [
                {"page_content": d.page_content, "metadata": d.metadata} for d in self.documents
            ]
            self.bm25_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.bm25_path, "w", encoding="utf-8") as f:
                json.dump(data, f, default=str)
        except Exception as e:
            logger.error(f"Failed to save BM25 cache: {e}")

        return retriever
