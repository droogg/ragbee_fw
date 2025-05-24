"""ingestion_service.py

This module provides the IngestionService, which orchestrates the end-to-end
pipeline of loading raw documents, splitting them into smaller chunks, and
building a retrieval index. It adheres to the ports defined in the core layer,
allowing interchangeable implementations of data loaders, splitters, and
retrievers."""

from typing import Sequence

from src.ragbee_fw.core.models.document import Document
from src.ragbee_fw.core.ports.data_loader_port import DataLoaderPort
from src.ragbee_fw.core.ports.retriever_port import RetrieverPort
from src.ragbee_fw.core.ports.splitter_port import TextSplitterPort


class IngestionService:
    """Service for ingesting documents into a retrieval index.

    This service performs three steps:
      1) Load raw documents from a file or directory via a DataLoaderPort.
      2) Split each raw document into smaller chunks via a TextSplitterPort.
      3) Build or update the retrieval index via a RetrieverPort.

    Attributes:
        loader: An implementation of DataLoaderPort that loads raw Document objects.
        chunker: An implementation of TextSplitterPort that splits Document objects
            into smaller, indexable chunks.
        retriever: An implementation of RetrieverPort that builds and queries the index.
    """

    def __init__(
        self,
        loader: DataLoaderPort,
        chunker: TextSplitterPort,
        retriever: RetrieverPort,
    ) -> None:
        """Initialize the IngestionService.

        Args:
            loader: A DataLoaderPort responsible for loading raw documents.
            chunker: A TextSplitterPort responsible for splitting documents into chunks.
            retriever: A RetrieverPort responsible for indexing and retrieving chunks.
        """
        self.loader = loader
        self.chunker = chunker
        self.retriever = retriever

    def build_index(self, path: str) -> RetrieverPort:
        """Load documents, split into chunks, and build the retrieval index.

        This method performs a full re-indexing:
          1. Loads all documents under the given path (file or directory).
          2. Splits each loaded Document into smaller chunks.
          3. Indexes the resulting chunks in the retriever.

        Args:
            path: Path to a file or directory containing raw documents to ingest.

        Returns:
            RetrieverPort: The same retriever instance, now populated with the index.

        Example:
            >>> from src.ragbee_fw.infrastructure.data_loader.file_loader import FileSystemLoader
            >>> from src.ragbee_fw.infrastructure.chunker.recursive_splitter import RecursiveTextSplitter
            >>> from src.ragbee_fw.infrastructure.retriever.bm25_client import BM25Client
            >>> loader = FileSystemLoader()
            >>> chunker = RecursiveTextSplitter(chunk_size=500, chunk_overlap=50)
            >>> retriever = BM25Client()
            >>> service = IngestionService(loader, chunker, retriever)
            >>> retriever = service.build_index("data/legal_docs")
            >>> # Now retriever is ready to answer queries
        """
        raw_docs: Sequence[Document] = self.loader.load(path)
        chunks: Sequence[Document] = self.chunker.split_text(raw_docs)
        self.retriever.index(chunks)
        return self.retriever
