import json
from pathlib import Path
from typing import List, Optional

from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from src.core.config_loader import settings
from src.core.logger import logger


class HybridSearcher:
    """Combines dense (vector) and sparse (BM25) retrieval into a hybrid search."""

    def __init__(self, vector_store, documents: Optional[List[Document]] = None):
        self.vector_store = vector_store
        self.documents = documents
        self.top_k = settings.rag.top_k
        self.weights = settings.rag.hybrid_weights

        self.cache_dir = Path(settings.rag.cache_dir)
        self.bm25_path = self.cache_dir / "bm25_documents.json"

    def get_retriever(self):
        """Create and return the hybrid ensemble retriever, or vector-only fallback."""
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
            return vector_retriever

        logger.info(
            f"Initializing Hybrid Search (Weights: {self.weights}, k: {self.top_k})"
        )

        return EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever],
            weights=self.weights,
        )

    def _get_or_create_bm25(self) -> BM25Retriever:
        """Load BM25 documents from cache and rebuild, or build and persist a new index."""
        if self.bm25_path.exists():
            try:
                logger.info("Loading BM25 documents from cache.")
                with open(self.bm25_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                docs = [
                    Document(page_content=d["page_content"], metadata=d["metadata"])
                    for d in data
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
                {"page_content": d.page_content, "metadata": d.metadata}
                for d in self.documents
            ]
            with open(self.bm25_path, "w", encoding="utf-8") as f:
                json.dump(data, f, default=str)
        except Exception as e:
            logger.error(f"Failed to save BM25 cache: {e}")

        return retriever
