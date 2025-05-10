from abc import ABC, abstractmethod
from typing import List

from src.ragbee_fw.core.models.document import Document


class DataLoaderPort(ABC):
    @abstractmethod
    def load(self, path: str) -> List[Document]:
        """Reads all documents from the given path (file or folder)"""
        raise NotImplementedError("Subclasses must implement this method")
