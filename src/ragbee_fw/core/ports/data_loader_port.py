from abc import ABC, abstractmethod
from typing import List, Protocol

from ragbee_fw.core.models.document import Document


class DataLoaderPort(Protocol):
    def load(self, path: str) -> List[Document]: ...


class BaseDataLoader(ABC, DataLoaderPort):
    @abstractmethod
    def load(self, path: str) -> List[Document]:
        """Reads all documents from the given path (file or folder)"""
        raise NotImplementedError("Subclasses must implement this method")
