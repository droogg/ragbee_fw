"""Port definitions for text‑splitting components.

Defines the structural contract (:class:`TextSplitterPort`) and an
abstract convenience base (:class:`BaseTextSplitter`) that enforces the
`split_text` interface returning a list of
:class:`~src.ragbee_fw.core.models.document.Document` instances.
"""

from abc import ABC, abstractmethod
from typing import List, Protocol

from src.ragbee_fw.core.models.document import Document


class TextSplitterPort(Protocol):
    """Structural interface for text splitter implementations.

    Implementations must accept a list of :class:`Document` objects and
    return a (possibly longer) list of documents after splitting.
    """

    def split_text(self, text: List[Document]) -> List[Document]:
        """Split *text* into smaller documents.

        Args:
            text: Input collection of documents.

        Returns:
            A list of output documents after splitting.
        """
        ...


class BaseTextSplitter(TextSplitterPort, ABC):
    """Abstract helper class that enforces the :pyattr:`split_text` contract."""

    @abstractmethod
    def split_text(self, text: List[Document]) -> List[Document]:
        """Split *text* into smaller documents.

        Subclasses must override this method.
        """
        raise NotImplementedError("Subclasses must implement this method")
