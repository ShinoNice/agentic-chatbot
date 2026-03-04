"""
Singleton lifecycle manager for the core AI system.

Provides a FastAPI-compatible dependency that wraps DocumentProcessor,
VectorStoreManager, HybridSearcher and RAGOrchestrator — exactly the
same components wired together in src.main.AIAgentSystem, but designed
to live across many HTTP requests.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.core.config_loader import settings
from src.core.logger import logger
from src.retrieval.document_processor import DocumentProcessor
from src.retrieval.hybrid_search import HybridSearcher
from src.retrieval.vector_store import VectorStoreManager
from src.workflow.orchestrator import RAGOrchestrator


class SystemManager:
    """Manages the lifetime of the RAG pipeline components.

    Instantiate once at application startup, then inject via
    ``Depends(get_system)`` in route handlers.
    """

    def __init__(self) -> None:
        logger.info("SystemManager: initialising core components …")
        self.processor = DocumentProcessor()
        self.vector_manager = VectorStoreManager()

        self.searcher: Optional[HybridSearcher] = None
        self.orchestrator: Optional[RAGOrchestrator] = None

        self._files_indexed: List[str] = []
        self._total_chunks: int = 0

    # ── Properties ────────────────────────────────────────────────────

    @property
    def is_ready(self) -> bool:
        """True when the orchestrator is wired up and can answer queries."""
        return self.orchestrator is not None

    @property
    def vector_store_type(self) -> str:
        if settings.pinecone_api_key and settings.rag.pinecone_index_name:
            return "pinecone"
        return "chromadb"

    @property
    def documents_indexed(self) -> int:
        return self._total_chunks

    # ── Startup helper ────────────────────────────────────────────────

    async def try_connect_existing(self) -> None:
        """Attempt to attach to an already-populated vector store.

        Called once during lifespan startup so that the API can serve
        queries immediately if a knowledge base already exists.
        """
        try:
            vs = self.vector_manager.get_vector_store()
            self.searcher = HybridSearcher(vs)
            self.orchestrator = RAGOrchestrator(self.searcher)
            logger.info(
                "SystemManager: connected to existing vector store.")
        except Exception as exc:
            logger.info(
                f"SystemManager: no existing vector store found ({exc}). "
                "Ingest documents via POST /ingest first."
            )

    # ── Ingestion ─────────────────────────────────────────────────────

    async def ingest(self, namespace: str = "default") -> Dict[str, Any]:
        """Process PDFs in ``data/raw/`` and upsert into the vector store.

        Returns a summary dict consumed by the ``/ingest`` route.
        """
        raw_dir = Path(settings.rag.raw_data_dir)
        if not raw_dir.exists():
            raw_dir.mkdir(parents=True, exist_ok=True)

        pdf_files = list(raw_dir.glob("*.pdf"))
        if not pdf_files:
            return {
                "files_processed": [],
                "total_chunks": 0,
                "status": "failed",
                "message": f"No PDF files found in {raw_dir}.",
            }

        logger.info(
            f"SystemManager: ingesting {len(pdf_files)} document(s) …")

        chunks = self.processor.process(pdf_files)
        vector_store = self.vector_manager.create_index(
            chunks, namespace=namespace)

        self.searcher = HybridSearcher(vector_store, documents=chunks)
        self.orchestrator = RAGOrchestrator(self.searcher)

        self._files_indexed = [p.name for p in pdf_files]
        self._total_chunks = len(chunks)

        logger.info(
            f"SystemManager: ingestion complete — "
            f"{len(pdf_files)} files, {len(chunks)} chunks."
        )

        return {
            "files_processed": self._files_indexed,
            "total_chunks": self._total_chunks,
            "status": "completed",
            "message": "Ingestion successful.",
        }

    # ── Query ─────────────────────────────────────────────────────────

    async def query(self, question: str) -> Dict[str, Any]:
        """Run the full agentic RAG pipeline and return the final state.

        Raises ``RuntimeError`` when called before the knowledge base is
        loaded.
        """
        if not self.is_ready:
            raise RuntimeError(
                "Knowledge base is not loaded. "
                "Call POST /ingest or add documents first."
            )

        result = await self.orchestrator.run(question)
        return result


# ── FastAPI dependency ────────────────────────────────────────────────

_system: Optional[SystemManager] = None


def get_system() -> SystemManager:
    """FastAPI ``Depends`` callable returning the global SystemManager."""
    if _system is None:
        raise RuntimeError("SystemManager has not been initialised yet.")
    return _system


def init_system() -> SystemManager:
    """Create (or return) the global SystemManager singleton."""
    global _system
    if _system is None:
        _system = SystemManager()
    return _system
