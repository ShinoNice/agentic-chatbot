import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from pathlib import Path
from typing import List

import fitz
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from src.core.config_loader import settings
from src.core.logger import logger
from src.core.exceptions import DocumentProcessingError

_DOCLING_TIMEOUT_SECONDS = 300
_MAX_DOCLING_PAGES = 80


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

        self._docling_converter = None  # lazy init — see _get_docling_converter()

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            add_start_index=True,
        )

    def _get_docling_converter(self):
        """Lazily initialise the Docling converter on first use."""
        if self._docling_converter is None:
            logger.info("Initialising Docling DocumentConverter (first uncached PDF)...")
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import (
                RapidOcrOptions,
                PdfPipelineOptions,
            )
            from docling.document_converter import DocumentConverter, PdfFormatOption

            pipeline_options = PdfPipelineOptions(
                do_ocr=settings.docling.do_ocr,
                images_scale=settings.docling.images_scale,
                ocr_options=RapidOcrOptions(
                    force_full_page_ocr=settings.docling.force_full_page_ocr
                ),
                allow_external_plugins=True,
            )
            self._docling_converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
                }
            )
            logger.info("Docling DocumentConverter ready.")
        return self._docling_converter

    def process(self, file_paths: List[Path]) -> List[Document]:
        """
        Orchestrates the processing of multiple files.
        Checks cache first to avoid redundant computation.
        """
        all_chunks = []
        failed = []
        for path in file_paths:
            try:
                chunks = self._process_single_file(path)
                all_chunks.extend(chunks)
                logger.info(f"{path.name}: {len(chunks)} chunk(s)")
            except DocumentProcessingError as e:
                logger.error(str(e))
                failed.append(path.name)
                continue

        if failed:
            logger.warning(f"Failed to process {len(failed)} file(s): {failed}")

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
        image-heavy pages. Runs Docling in a thread with a timeout to prevent
        hangs on large/image-heavy PDFs. Also falls back if Docling returns
        suspiciously few chunks (silent partial failure from swallowed C++ errors).
        """
        with fitz.open(file_path) as pdf:
            page_count = len(pdf)
        if page_count > _MAX_DOCLING_PAGES:
            logger.info(
                f"Skipping Docling for {file_path.name} ({page_count} pages > "
                f"{_MAX_DOCLING_PAGES}). Using PyMuPDF directly."
            )
            return self._pymupdf_fallback(file_path)

        try:
            from langchain_docling import DoclingLoader

            logger.info(f"Attempting Docling parse for {file_path.name} (timeout: {_DOCLING_TIMEOUT_SECONDS}s)...")
            loader = DoclingLoader(
                file_path=str(file_path), converter=self._get_docling_converter()
            )

            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(loader.load)
                docs = future.result(timeout=_DOCLING_TIMEOUT_SECONDS)

            if len(docs) < self.min_chunks_fallback:
                raise ValueError(
                    f"Docling returned only {len(docs)} document(s) for {file_path.name}, "
                    f"below threshold of {self.min_chunks_fallback}. Treating as failure."
                )

            for d in docs:
                d.metadata["parser"] = "docling"
            logger.info(f"Docling parsed {file_path.name} successfully ({len(docs)} page(s)).")
            return docs
        except TimeoutError:
            logger.warning(
                f"Docling timed out after {_DOCLING_TIMEOUT_SECONDS}s for {file_path.name}, "
                f"falling back to PyMuPDF."
            )
            return self._pymupdf_fallback(file_path)
        except Exception as e:
            logger.warning(
                f"Docling failed for {file_path.name}, falling back to PyMuPDF. Error: {e}"
            )
            return self._pymupdf_fallback(file_path)

    def _pymupdf_fallback(self, file_path: Path) -> List[Document]:
        """Fast, robust fallback PDF loader."""
        loader = PyMuPDFLoader(str(file_path))
        docs = loader.load()
        for d in docs:
            d.metadata["parser"] = "pymupdf"
        logger.info(f"PyMuPDF parsed {file_path.name} ({len(docs)} page(s)).")
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
