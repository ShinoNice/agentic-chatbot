import asyncio
import os

from dotenv import load_dotenv

from src.api.dependencies import SystemManager
from src.core.logger import logger

load_dotenv()

_TRACING_ENABLED = os.getenv("LANGSMITH_TRACING") == "true"
logger.info(f"LangSmith Tracing: {'ENABLED' if _TRACING_ENABLED else 'DISABLED'}")


async def main():
    """CLI entry point for the Agentic RAG system."""
    system = SystemManager()

    # Attempt ingestion of new PDFs
    result = await system.ingest()
    if result["status"] == "failed":
        logger.info(result["message"])

    # Connect to existing store if ingestion was skipped
    if not system.is_ready:
        await system.try_connect_existing()

    if not system.is_ready:
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

            result = await system.query(user_input)

            answer = result.get("draft_answer", "I couldn't generate an answer.")
            verification = result.get("verification")

            print(f"\rAgent: {answer}")

            if verification and not verification.supported:
                print("\n[NOTE] This answer may contain unsupported claims.")

        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.error(f"Error in chat loop: {e}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(main())
