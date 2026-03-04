import json
import time
from typing import List, Union

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from src.core.config_loader import settings
from src.core.exceptions import VectorStoreConnectionError
from src.core.logger import logger
from src.engines.embedding_model import get_embedding_engine


class VectorStoreManager:
    """Manages the lifecycle of the vector database (Pinecone or Chroma)."""

    def __init__(self):
        self.embedding_engine = get_embedding_engine()
        self.embeddings = self.embedding_engine.get_embeddings()

        self.pc_api_key = settings.pinecone_api_key
        self.index_name = settings.rag.pinecone_index_name

        self.persist_directory = settings.rag.vector_db_path

    def _clean_metadata(self, metadata: dict) -> dict:
        """Flatten or serialize values incompatible with Pinecone's metadata types."""
        cleaned = {}
        for key, value in metadata.items():
            if value is None:
                continue
            if isinstance(value, list) and all(isinstance(i, str) for i in value):
                cleaned[key] = value
            elif isinstance(value, (dict, list)):
                cleaned[key] = json.dumps(value)
            else:
                cleaned[key] = value
        return cleaned

    def create_index(
        self,
        documents: List[Document],
        namespace: str = "default",
    ) -> Union[PineconeVectorStore, Chroma]:
        """Clean metadata, then upsert documents into the configured store."""
        for doc in documents:
            doc.metadata = self._clean_metadata(doc.metadata)

        if self.pc_api_key and self.index_name:
            return self._setup_pinecone(documents, namespace)
        return self._setup_chroma(documents)

    def _setup_pinecone(
        self,
        documents: List[Document],
        namespace: str,
    ) -> PineconeVectorStore:
        """Create a Pinecone Serverless index (if needed) and upsert documents."""
        try:
            pc = Pinecone(api_key=self.pc_api_key)

            existing_indexes = [i.name for i in pc.list_indexes()]
            if self.index_name not in existing_indexes:
                logger.info(f"Creating Pinecone index: {self.index_name}")
                pc.create_index(
                    name=self.index_name,
                    dimension=1536,
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
                )
                while not pc.describe_index(self.index_name).status["ready"]:
                    time.sleep(1)

            ids = [doc.metadata.get("chunk_hash") for doc in documents]

            vector_store = PineconeVectorStore(
                index_name=self.index_name,
                embedding=self.embeddings,
                namespace=namespace,
                pinecone_api_key=self.pc_api_key,
            )

            vector_store.add_documents(documents=documents, ids=ids)
            logger.info(f"Upserted {len(documents)} docs to Pinecone.")
            return vector_store

        except Exception as e:
            logger.error(f"Pinecone setup failed: {e}")
            raise VectorStoreConnectionError(
                "Could not connect to Pinecone cloud.")

    def _setup_chroma(self, documents: List[Document]) -> Chroma:
        """Persist documents into a local ChromaDB instance."""
        try:
            logger.info(
                f"Initializing local ChromaDB at {self.persist_directory}")
            ids = [doc.metadata.get("chunk_hash") for doc in documents]

            return Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=self.persist_directory,
                ids=ids,
            )
        except Exception as e:
            logger.error(f"Chroma setup failed: {e}")
            raise VectorStoreConnectionError(
                f"Local ChromaDB initialization failed: {e}")

    def get_vector_store(
        self,
        namespace: str = "default",
    ) -> Union[PineconeVectorStore, Chroma]:
        """Return an existing connection to the vector store without re-indexing."""
        if self.pc_api_key and self.index_name:
            return PineconeVectorStore(
                index_name=self.index_name,
                embedding=self.embeddings,
                namespace=namespace,
                pinecone_api_key=self.pc_api_key,
            )
        return Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings,
        )
