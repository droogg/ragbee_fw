"""Retriever port and base class for retriever adapters"""

from abc import ABC, abstractmethod
from typing import List, Protocol


class RetrieverPort(Protocol):
    def retrieve(self, query: str, k: int) -> List[str]: ...


class BaseRetriever(RetrieverPort, ABC):
    @abstractmethod
    def retrieve(self, query: str, k: int = 5) -> List[str]:
        """Returns a list of documents relevant to the query."""
        raise NotImplementedError("Subclasses must implement this method")
