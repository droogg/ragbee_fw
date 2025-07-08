"""Retriever port definitions for RAGBee FW.

This module specifies the **contract** that any document retriever must
follow to plug into the framework:

* :class:`RetrieverPort` — a :pytyping:`typing.Protocol` describing the
  two required operations:

  * :pymeth:`index` builds an internal structure over a list of
    :class:`~src.ragbee_fw.core.models.document.Document` objects.
  * :pymeth:`retrieve` returns the *k* most relevant documents for a
    natural‑language query.

* :class:`BaseRetriever` — an optional ABC that enforces the same API via
  classic subclassing.  Concrete retrievers can inherit from it or just
  implement the protocol.

Both abstractions let the application depend only on behaviour, making
it easy to swap a BM25 in‑memory retriever with, say, a vector‑database
backend without changing business logic.
"""

from abc import ABC, abstractmethod
from typing import List, Protocol

from ragbee_fw.core.models.document import Document


class RetrieverPort(Protocol):
    """Structural type for retrievers.

    Any class that implements these two methods is considered a valid
    *retriever* by the framework, even without explicit inheritance.
    """

    def index(self, docs: List[Document]) -> None:
        """Build an index over *docs*.

        Args:
            docs: Corpus of documents to make searchable.
        """
        ...

    def retrieve(self, query: str, k: int) -> List[Document]:
        """Return top‑*k* documents relevant to *query*.

        Args:
            query: Natural‑language search string.
            k: Number of hits to return.

        Returns:
            A list of documents ordered by decreasing relevance.
        """
        ...


class BaseRetriever(RetrieverPort, ABC):
    """Abstract helper base for classic inheritance.

    Subclass this if you prefer an ABC with enforced *index* / *retrieve*
    signatures instead of relying on structural typing via
    :class:`RetrieverPort`.
    """

    @abstractmethod
    def index(self, docs: List[Document]) -> None:
        """Construct an internal index.

        Args:
            docs: Collection of documents to ingest.
        """
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def retrieve(self, query: str, k: int = 5) -> List[Document]:
        """Fetch the *k* most relevant documents.

        Args:
            query: Search query.
            k: How many results to return. Defaults to **5**.

        Returns:
            A list of the *k* highest‑scoring documents.
        """
        raise NotImplementedError("Subclasses must implement this method")
