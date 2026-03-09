class AIProjectError(Exception):
    """Base class for all exceptions raised by the AI Project."""

    def __init__(self, message: str, details: dict = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


# --- Retrieval & RAG Exceptions ---


class RetrievalError(AIProjectError):
    """Base class for errors occurring in the RAG pipeline."""

    pass


class VectorStoreConnectionError(RetrievalError):
    """Raised when the connection to Chroma or Pinecone fails."""

    pass


class DocumentProcessingError(AIProjectError):
    """Raised when parsing, chunking, or hashing a document fails."""

    pass


# --- Agent & LLM Exceptions ---


class AgentError(AIProjectError):
    """Base class for errors occurring within the LangGraph agentic loop."""

    pass


class LLMResponseError(AgentError):
    """Raised when the LLM returns an empty response or malformed data."""

    pass


class RelevanceAuditError(AgentError):
    """Raised when the RelevanceChecker fails to classify a document chunk."""

    pass
