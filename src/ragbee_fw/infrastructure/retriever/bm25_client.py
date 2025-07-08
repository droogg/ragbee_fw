"""BM25‑based keyword retriever.

This module defines :class:`BM25Client`, a lightweight in‑memory
retriever that ranks documents with **Okapi BM25** — a bag‑of‑words
probabilistic relevance model.  BM25 scores a document *D* for a query
*Q* by summing, over each query term *t*, a term‑frequency factor and an
inverse‑document‑frequency factor:

⋅⋅*score(D, Q) = ∑_t IDF(t) · |D|ₖ / (|D|ₖ + k₁·(1 – b + b·|D|/avgDL))*

where *|D|ₖ* — term frequency of *t* in *D*, *avgDL* — average document
length, and *k₁, b* are smoothing constants.  We rely on the `rank_bm25`
implementation (`BM25Okapi`).

Workflow
~~~~~~~~
1. **index(** *docs* **)**
   Tokenises documents (`str.split`) and builds an internal
   :class:`rank_bm25.BM25Okapi` model.
2. **retrieve(** *query, k* **)**
   Tokenises the query, scores each document, and returns the *k* best
   :class:`~src.ragbee_fw.core.models.document.Document` instances.

The client is fully offline/in‑memory and suited for quick keyword
retrieval baselines or fallback strategies in RAG pipelines.
"""

from typing import List

from rank_bm25 import BM25Okapi

from ragbee_fw.core.models.document import Document
from ragbee_fw.core.ports.retriever_port import RetrieverPort


class BM25Client(RetrieverPort):
    """In‑memory BM25 keyword retriever.

    Attributes
    ----------
    bm25:
        The underlying :class:`rank_bm25.BM25Okapi` model built in
        :py:meth:`index`. ``None`` until the first call.
    docs:
        Original list of documents aligned with BM25’s internal order so
        that index *i* in *scores* corresponds to ``docs[i]``.
    """

    def __init__(self) -> None:
        super().__init__()
        self.bm25: BM25Okapi | None = None
        self.docs: List[Document] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def index(self, docs: List[Document]) -> None:
        """Build internal BM25 index.

        Parameters
        ----------
        docs
            List of :class:`~src.ragbee_fw.core.models.document.Document`
            whose ``text`` fields will be tokenised by whitespace.
        """
        self.docs = docs
        tokenized: List[List[str]] = [doc.text.split() for doc in docs]
        self.bm25 = BM25Okapi(tokenized)

    def retrieve(self, query: str, k: int) -> List[Document]:
        """Return the *k* most relevant documents for *query*.

        Parameters
        ----------
        query
            User search query.
        k
            Number of top documents to return.

        Returns
        -------
        List[Document]
            The *k* documents with the highest BM25 score, ordered from
            best to worst.

        Raises
        ------
        RuntimeError
            If :py:meth:`index` has not been called before retrieval.
        """
        if self.bm25 is None:
            raise RuntimeError(
                "BM25Client: index not built - call .index(docs) before retrieve()"
            )
        query_tokens = query.split()
        scores = self.bm25.get_scores(query_tokens)
        top_idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [self.docs[i] for i in top_idxs]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @classmethod
    def from_documents(cls, docs: List[Document]) -> "BM25Client":
        """Create and index a :class:`BM25Client` in a single call.

        Useful shorthand for quick experiments:

        >>> retriever = BM25Client.from_documents(corpus)
        >>> results = retriever.retrieve("what is bm25", k=5)
        """
        inst = cls()
        inst.index(docs)
        return inst
