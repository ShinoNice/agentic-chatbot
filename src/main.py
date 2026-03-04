import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from src.workflow.orchestrator import RAGOrchestrator
from src.core.config_loader import settings
from src.core.logger import logger
from src.retrieval.document_processor import DocumentProcessor
from src.retrieval.hybrid_search import HybridSearcher
from src.retrieval.vector_store import VectorStoreManager

load_dotenv()

_TRACING_ENABLED = os.getenv("LANGSMITH_TRACING") == "true"
logger.info(f"LangSmith Tracing: {'ENABLED' if _TRACING_ENABLED else 'DISABLED'}")


class AIAgentSystem:
    """Initializes components and manages the user interaction loop."""

    def __init__(self):
        logger.info("Initializing AI Agent System...")

        self.processor = DocumentProcessor()
        self.vector_manager = VectorStoreManager()

        self.searcher = None
        self.orchestrator = None

    async def ingest_documents(self):
        """Scan data/raw for PDFs, process them, and upsert to the vector DB."""
        raw_dir = Path(settings.rag.raw_data_dir)
        if not raw_dir.exists():
            raw_dir.mkdir(parents=True)
            logger.warning(
                f"Raw data directory created at {raw_dir}. Add PDFs there.")
            return

        pdf_files = list(raw_dir.glob("*.pdf"))
        if not pdf_files:
            logger.info("No PDF files found in data/raw. Skipping ingestion.")
            return

        logger.info(
            f"Found {len(pdf_files)} document(s). Starting ingestion...")

        chunks = self.processor.process(pdf_files)
        vector_store = self.vector_manager.create_index(chunks)

        self.searcher = HybridSearcher(vector_store, documents=chunks)
        self.orchestrator = RAGOrchestrator(self.searcher)

        logger.info("Ingestion complete. Knowledge base is ready.")

    async def start_chat(self):
        """Run the interactive CLI loop."""
        if not self.orchestrator:
            try:
                vs = self.vector_manager.get_vector_store()
                self.searcher = HybridSearcher(vs)
                self.orchestrator = RAGOrchestrator(self.searcher)
            except Exception:
                print("\n[!] Knowledge base is empty. Add PDFs to data/raw first.")
                return

        print(f"\n{'=' * 50}")
        print("AI Agent RAG System (Type 'exit' or 'quit' to stop)")
        print("=" * 50)

        while True:
            try:
                user_input = input("\nUser: ").strip()

                if user_input.lower() in ("exit", "quit"):
                    print("Shutting down. Goodbye!")
                    break

                if not user_input:
                    continue

                print("\nAgent is thinking...", end="\r")

                result = await self.orchestrator.run(user_input)

                answer = result.get(
                    "draft_answer", "I couldn't generate an answer.")
                verification = result.get("verification")

                print(f"\rAgent: {answer}")

                if verification and not verification.supported:
                    print("\n[NOTE] This answer may contain unsupported claims.")

            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Error in chat loop: {e}", exc_info=True)
                print(f"\n[!] An error occurred: {e}")


async def main():
    system = AIAgentSystem()
    await system.ingest_documents()
    await system.start_chat()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(0)
