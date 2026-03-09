import hashlib
import json
from pathlib import Path
from typing import List

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import RapidOcrOptions, PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_docling import DoclingLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from src.core.config_loader import settings
from src.core.logger import logger, suppress_stderr
from src.core.exceptions import DocumentProcessingError


class DocumentProcessor:
    """
    Handles the conversion of raw PDF files into cleaned, hashed,
    and metadata-enriched document chunks.
    """

    def __init__(self):
        self.chunk_size = settings.rag.chunk_size
        self.chunk_overlap = settings.rag.chunk_overlap
        self.cache_dir = Path(settings.rag.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.min_chunks_fallback = settings.docling.min_chunks_fallback

        pipeline_options = PdfPipelineOptions(
            do_ocr=settings.docling.do_ocr,
            images_scale=settings.docling.images_scale,
            ocr_options=RapidOcrOptions(
                force_full_page_ocr=settings.docling.force_full_page_ocr
            ),
            allow_external_plugins=True,
        )
        self.docling_converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            add_start_index=True,
        )

    def process(self, file_paths: List[Path]) -> List[Document]:
        """
        Orchestrates the processing of multiple files.
        Checks cache first to avoid redundant computation.
        """
        all_chunks = []
        for path in file_paths:
            try:
                chunks = self._process_single_file(path)
                all_chunks.extend(chunks)
            except DocumentProcessingError as e:
                logger.error(str(e))
                continue

        return all_chunks

    def _process_single_file(self, file_path: Path) -> List[Document]:
        """
        Processes one PDF: Load -> Clean -> Chunk -> Hash -> Cache.
        """
        try:
            cache_file = self.cache_dir / f"{file_path.stem}.json"
            if cache_file.exists():
                logger.info(f"Loading {file_path.name} from cache...")
                with open(cache_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return [
                    Document(page_content=d["page_content"], metadata=d["metadata"])
                    for d in data
                ]

            logger.info(f"Parsing new file: {file_path.name}")

            # 2. Extract Text (Hybrid Strategy)
            documents = self._extract_text(file_path)

            # 3. Create Chunks
            chunks = self.splitter.split_documents(documents)

            # 4. Enrich with Hashes and Metadata
            final_chunks = self._enrich_metadata(chunks, file_path)

            # 5. Save to Cache
            data = [
                {"page_content": c.page_content, "metadata": c.metadata}
                for c in final_chunks
            ]
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(data, f, default=str)

            return final_chunks

        except DocumentProcessingError:
            raise
        except Exception as e:
            raise DocumentProcessingError(
                f"Processing failed for {file_path.name}: {e}"
            ) from e

    def _extract_text(self, file_path: Path) -> List[Document]:
        """
        Attempts Docling for structure, falls back to PyMuPDF.
        Uses a pre-configured DocumentConverter to avoid std::bad_alloc on
        image-heavy pages. Also falls back if Docling returns suspiciously few
        chunks (silent partial failure from swallowed C++ errors).
        """
        try:
            # Attempt Docling (Better for tables/headers)
            loader = DoclingLoader(
                file_path=str(file_path), converter=self.docling_converter
            )
            with suppress_stderr():
                docs = loader.load()

            # Guard against silent partial failures: if Docling drops most of the
            # document (e.g. std::bad_alloc swallowed internally), fall back.
            if len(docs) < self.min_chunks_fallback:
                raise ValueError(
                    f"Docling returned only {len(docs)} document(s) for {file_path.name}, "
                    f"below threshold of {self.min_chunks_fallback}. Treating as failure."
                )

            for d in docs:
                d.metadata["parser"] = "docling"
            return docs
        except Exception as e:
            logger.warning(
                f"Docling failed for {file_path.name}, falling back to PyMuPDF. Error: {e}"
            )

            # Fallback to PyMuPDF (Fast and robust)
            loader = PyMuPDFLoader(str(file_path))
            docs = loader.load()
            for d in docs:
                d.metadata["parser"] = "pymupdf"
            return docs

    def _enrich_metadata(
        self, chunks: List[Document], file_path: Path
    ) -> List[Document]:
        """
        Adds deterministic hashes and source info to each chunk.
        This is critical for the VectorStore's deduplication logic.
        """
        for i, chunk in enumerate(chunks):
            # Create a unique hash based on content and file source
            content_hash = hashlib.sha256(
                f"{file_path.name}_{chunk.page_content}".encode()
            ).hexdigest()

            chunk.metadata.update(
                {
                    "source": file_path.name,
                    "chunk_index": i,
                    "chunk_hash": content_hash,
                    "chunk_size": len(chunk.page_content),
                }
            )
        return chunks
