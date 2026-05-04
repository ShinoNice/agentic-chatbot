from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

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

        self.searcher: HybridSearcher | None = None
        self.orchestrator: RAGOrchestrator | None = None

        self._files_indexed: list[str] = []
        self._total_chunks: int = 0

        # Per-session orchestrators built from user-uploaded PDFs. A session
        # only appears here after it successfully posts to /api/upload; /chat
        # then prefers the session-scoped orchestrator over the default one.
        self._session_searchers: dict[str, HybridSearcher] = {}
        self._session_orchestrators: dict[str, RAGOrchestrator] = {}

    def _searcher_k(self) -> int:
        """Decide how many candidates the retriever returns.

        When reranking is enabled the retriever returns a wider candidate
        pool (rerank.candidate_k) so the reranker has something to filter.
        When disabled, fall back to the original rag.top_k.
        """
        if settings.rerank.enabled:
            return settings.rerank.candidate_k
        return settings.rag.top_k

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

            # Reconstruct documents from cached chunk JSONs so that
            # HybridSearcher can build (or rebuild) its BM25 index.
            cache_dir = Path(settings.rag.cache_dir)
            cached_docs: list[Document] = []
            for jf in sorted(cache_dir.glob("*.json")):
                if jf.name == "bm25_documents.json":
                    continue
                try:
                    with open(jf, encoding="utf-8") as f:
                        data = json.load(f)
                    cached_docs.extend(
                        Document(page_content=d["page_content"], metadata=d["metadata"])
                        for d in data
                    )
                except Exception as e:
                    logger.warning(f"Skipping corrupt cache file {jf.name}: {e}")

            self.searcher = HybridSearcher(
                vs,
                documents=cached_docs if cached_docs else None,
                k=self._searcher_k(),
            )
            self.orchestrator = RAGOrchestrator(self.searcher)
            logger.info(
                "SystemManager: connected to existing vector store "
                f"({len(cached_docs)} cached document chunks loaded for BM25)."
            )
        except Exception as exc:
            logger.info(
                f"SystemManager: no existing vector store found ({exc}). "
                "Ingest documents via POST /ingest first."
            )

    # ── Ingestion ─────────────────────────────────────────────────────

    async def ingest(self, namespace: str = "default") -> dict[str, Any]:
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

        logger.info(f"SystemManager: ingesting {len(pdf_files)} document(s) …")

        chunks = self.processor.process(pdf_files)
        vector_store = self.vector_manager.create_index(chunks, namespace=namespace)

        self.searcher = HybridSearcher(vector_store, documents=chunks, k=self._searcher_k())
        self.orchestrator = RAGOrchestrator(self.searcher)

        self._files_indexed = [p.name for p in pdf_files]
        self._total_chunks = len(chunks)

        logger.info(
            f"SystemManager: ingestion complete — {len(pdf_files)} files, {len(chunks)} chunks."
        )

        return {
            "files_processed": self._files_indexed,
            "total_chunks": self._total_chunks,
            "status": "completed",
            "message": "Ingestion successful.",
        }

    # ── Per-session upload ────────────────────────────────────────────

    async def upload(
        self,
        file_bytes: bytes,
        filename: str,
        session_id: str,
    ) -> dict[str, Any]:
        """Ingest a single uploaded PDF into a session-scoped Pinecone namespace.

        Subsequent /chat calls with the same ``session_id`` will retrieve only
        from this namespace (see ``query``), not the default curated corpus.
        """
        import tempfile

        namespace = f"session-{session_id}"

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = Path(tmp.name)

        try:
            chunks = self.processor.process([tmp_path])
        finally:
            try:
                tmp_path.unlink()
            except OSError:
                pass

        if not chunks:
            raise ValueError(f"No content could be extracted from {filename}.")

        vector_store = self.vector_manager.create_index(chunks, namespace=namespace)

        searcher = HybridSearcher(vector_store, documents=chunks, k=self._searcher_k())
        orchestrator = RAGOrchestrator(searcher)

        self._session_searchers[session_id] = searcher
        self._session_orchestrators[session_id] = orchestrator

        logger.info(
            f"SystemManager: uploaded {filename} into namespace {namespace} ({len(chunks)} chunks)."
        )

        return {
            "filename": filename,
            "total_chunks": len(chunks),
            "namespace": namespace,
            "session_id": session_id,
        }

    # ── Query ─────────────────────────────────────────────────────────

    async def query(self, question: str, session_id: str = "") -> dict[str, Any]:
        """Run the full agentic RAG pipeline and return the final state.

        When ``session_id`` matches a prior upload, routes through that
        session's orchestrator so the LLM only sees the uploaded document.
        Otherwise falls back to the default curated-corpus orchestrator.
        """
        orchestrator = self._session_orchestrators.get(session_id) or self.orchestrator

        if orchestrator is None:
            raise RuntimeError(
                "Knowledge base is not loaded. Call POST /ingest or POST /upload first."
            )

        result = await orchestrator.run(question, session_id=session_id)
        return result

    async def query_stream(self, question: str, session_id: str = ""):
        """Yield (event_type, payload) tuples for streaming /chat consumers."""
        orchestrator = self._session_orchestrators.get(session_id) or self.orchestrator

        if orchestrator is None:
            raise RuntimeError(
                "Knowledge base is not loaded. Call POST /ingest or POST /upload first."
            )

        async for event_type, payload in orchestrator.astream_run(
            question, session_id=session_id
        ):
            yield event_type, payload


# ── FastAPI dependency ────────────────────────────────────────────────

_system: SystemManager | None = None


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
