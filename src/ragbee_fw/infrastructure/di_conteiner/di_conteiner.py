"""
Dependency Injection container for RAGBee framework components.

This module provides a lightweight DI system for constructing and managing
framework services and their adapters (e.g., data loaders, chunkers, retrievers, LLMs)
based on the configuration defined in `AppConfig`.

To add a custom adapter, register it via `DIContainer.register_adapter()`.

Example:
    config = load_config_from_yaml("config.yaml")
    container = DIContainer(config)
    app = container.build_app_service()
    response = app.run("What is the capital of France?")
"""

import inspect
from typing import Literal

from pydantic import BaseModel

from ragbee_fw.core.models.app_config import AppConfig
from ragbee_fw.core.services.answer_service import AnswerService
from ragbee_fw.core.services.ingestion_service import IngestionService
from ragbee_fw.infrastructure.data_loader.file_loader import FileSystemLoader
from ragbee_fw.infrastructure.llm_clients.huggingface_client import (
    HuggingFaceInferenceAdapter,
)
from ragbee_fw.infrastructure.retriever.bm25_client import BM25Client
from ragbee_fw.infrastructure.text_splitter.recursive_text_splitter import (
    RecursiveTextSplitter,
)

ADAPTER_REGISTRY = {
    "data_loader": {
        "file_loader": FileSystemLoader,
    },
    "text_chunker": {
        "recursive_splitter": RecursiveTextSplitter,
    },
    "retriever": {
        "bm25": BM25Client,
    },
    "llm": {
        "HF": HuggingFaceInferenceAdapter,
    },
}


class DIContainer:
    """
    Dependency Injection container for building core services and adapters
    from configuration.

    This class is used to:
    - Resolve and instantiate adapters from config (data loader, chunker, retriever, LLM)
    - Construct core services (`IngestionService`, `AnswerService`)
    - Register custom adapters dynamically

    Args:
        config (AppConfig): Full configuration object parsed from YAML or other source.

    Example:
        container = DIContainer(config)
        app = container.build_app_service()
        response = app.run("Where is the nearest office?")
    """

    def __init__(self, config: AppConfig):
        """
        Initializes the DI container with the provided configuration.

        Args:
            config (AppConfig): Framework configuration object.
        """
        self.config = config
        self._cache = {}

    def build(
        self, component: Literal["data_loader", "text_chunker", "retriever", "llm"]
    ):
        """
        Builds (or returns cached) component instance by type.

        Args:
            component (Literal): One of the allowed component names.

        Returns:
            Any: Instantiated component (e.g., a retriever or LLM adapter).
        """
        return self._build_module(component)

    def build_ingestion_service(self):
        """
        Builds the ingestion service and optionally indexes documents.

        Returns:
            IngestionService: Service responsible for document indexing.

        Notes:
            If retriever is BM25, it will pre-index documents and cache the indexed retriever.
        """
        service = IngestionService(
            loader=self.build("data_loader"),
            chunker=self.build("text_chunker"),
            retriever=self.build("retriever"),
        )
        cfg = self.config
        if cfg.retriever.type == "bm25":
            retriever = service.build_index(cfg.data_loader.path)
            self._cache["retriever_with_index"] = retriever
        return service

    def build_answer_service(self) -> AnswerService:
        """
        Builds the answer service with appropriate retriever and LLM.

        Returns:
            AnswerService: Service responsible for answering user questions.
        """
        retriever = None
        if self.config.retriever.type == "bm25":
            retriever = self._cache.get("retriever_with_index")
        if retriever is None:
            retriever = self._cache.get("retriever")
        if retriever is None:
            retriever = self._build_module("retriever")

        service = AnswerService(retriever=retriever, llm=self.build("llm"))
        return service

    def build_app_service(self) -> AnswerService:
        """
        High-level entrypoint to build full application pipeline.

        Returns:
            AnswerService: Ready-to-use app service with all dependencies resolved.
        """
        self.build_ingestion_service()
        return self.build_answer_service()

    def _build_module(
        self,
        name: Literal[
            "data_loader",
            "text_chunker",
            "retriever",
            "llm",
        ],
    ):
        """
        Internal method to build and cache a specific adapter.

        Args:
            name (str): Component name.

        Returns:
            Any: Instantiated adapter for the given component.
        """
        if name not in self._cache:
            cfg = getattr(self.config, name)
            self._cache[name] = DIContainer.get_module(name, cfg)
        return self._cache[name]

    @staticmethod
    def get_module(name, cfg: BaseModel):
        """
        Instantiates adapter from registry based on config.

        Args:
            name (str): Component name (e.g., 'retriever').
            cfg (BaseModel): Parsed configuration for this component.

        Raises:
            ValueError: If adapter type not found or required parameters missing.

        Returns:
            Any: Instantiated adapter.
        """
        if cfg.type not in ADAPTER_REGISTRY[name]:
            raise ValueError(f"Unknown adapter type '{cfg.type}' for {name}")
        module = ADAPTER_REGISTRY[name][cfg.type]
        sig = inspect.signature(module)
        available_params = sig.parameters.keys()
        kwargs = {k: v for k, v in dict(cfg).items() if k in available_params}

        required = [
            name
            for name, param in sig.parameters.items()
            if param.default is param.empty
            and param.kind in (param.POSITIONAL_OR_KEYWORD, param.KEYWORD_ONLY)
        ]
        missing = [name for name in required if name not in kwargs]
        if missing:
            raise ValueError(
                f"Missing required params for {module.__name__}: {missing}"
            )
        return module(**kwargs)

    @staticmethod
    def register_adapter(component: str, adapter_type: str, cls: type):
        """
        Registers a custom adapter class into the registry.

        Args:
            component (str): One of "data_loader", "text_chunker", "retriever", or "llm".
            adapter_type (str): Unique name identifying the adapter type (e.g., 'qdrant', 'ollama').
            cls (type): Adapter class implementing the correct interface.

        Example:
            class MyCustomRetriever:
                ...

            DIContainer.register_adapter("retriever", "my_retriever", MyCustomRetriever)
        """
        if component not in ADAPTER_REGISTRY:
            ADAPTER_REGISTRY[component] = {}
        ADAPTER_REGISTRY[component][adapter_type] = cls
