"""Recursive text splitter utility for RAGBee FW.

This module defines :class:`RecursiveTextSplitter`, a configurable
component that breaks large texts (or lists of texts / documents) into
smaller :class:`~src.ragbee_fw.core.models.document.Document` chunks that
fit a target size.  The algorithm recursively picks the first separator
that appears in the text, splits, and then merges adjacent pieces so
that each resulting chunk respects *chunk_size* with optional
*chunk_overlap* between consecutive segments.

Typical usage example:

    >>> splitter = RecursiveTextSplitter(chunk_size=800, chunk_overlap=100)
    >>> docs = splitter.split_text(long_markdown_string)
"""

from typing import List, Optional, Union

from ragbee_fw.core.models.document import Document
from ragbee_fw.core.ports.splitter_port import BaseTextSplitter


class RecursiveTextSplitter(BaseTextSplitter):
    """Recursively splits text into size‑bounded chunks.

    The algorithm works as follows:

    1.  Pick the first separator from *separators* that is present in the
        text (fallback to empty‑string which splits by character).
    2.  Split the text by this separator.
    3.  If every part already fits *chunk_size*, merge adjacent pieces
        back together up to the size limit, keeping *chunk_overlap*
        characters between chunks.
    4.  Otherwise, recursively call :pymeth:`_split` on parts exceeding
        *chunk_size* using the remaining separators.

    Args:
        chunk_size: Maximum number of characters allowed in each final
            chunk.
        chunk_overlap: Number of tail characters from the previous chunk
            to prepend to the next one.  Useful for embedding models that
            benefit from overlapping context.
        separators: Ordered list of separators to try in descending
            preference (e.g. paragraph break, newline, space, empty
            string).  If *None*, defaults to ``["\n\n", "\n", " ", ""]``.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None,
    ) -> None:
        super().__init__()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", " ", ""]

    def split_text(
        self,
        text: Union[str, List[str], List[Document]],
        seps: Optional[List[str]] = None,
    ) -> List[Document]:
        """Public API for splitting text or documents.

        Accepts a single string, a list of strings, or an existing list
        of :class:`~src.ragbee_fw.core.models.document.Document` objects
        and produces a list of chunked :class:`Document` instances.

        Args:
            text: Input payload to be chunked.
            seps: Custom separator priority; defaults to the instance’s
                :pyattr:`separators`.

        Returns:
            List of chunked :class:`Document` objects.

        Raises:
            TypeError: If *text* is of an unsupported type.
        """
        seps = seps or self.separators
        if isinstance(text, str):
            return self._split(Document(id="0", text=text, meta=None), seps)
        if isinstance(text, list):
            out: List[Document] = []
            for i, part in enumerate(text):
                if isinstance(part, str):
                    out.extend(
                        self._split(Document(id=str(i), text=part, meta=None), seps)
                    )
                else:  # Document
                    out.extend(self._split(part, seps))
            return out
        raise TypeError(f"Unsupported type for split_text: {type(text)}")

    def _split(self, document: Document, seps: List[str]) -> List[Document]:
        """Recursive helper that actually performs the splitting logic."""
        text = document.text
        if len(text) <= self.chunk_size:
            return [document]

        for idx, sep in enumerate(seps):
            if sep == "" or sep in text:
                first_sep = sep
                rest_seps = seps[idx + 1 :]
                break
        else:
            first_sep = seps[-1]
            rest_seps = []

        parts = text.split(first_sep) if first_sep else list(text)
        docs = [
            Document(id=f"{document.id}-{i}", text=part, meta=document.meta)
            for i, part in enumerate(parts)
        ]

        if all(len(d.text) <= self.chunk_size for d in docs):
            return self._merge(docs, first_sep)

        new_docs: List[Document] = []
        for d in docs:
            if len(d.text) > self.chunk_size and rest_seps:
                new_docs.extend(self._split(d, rest_seps))
            else:
                new_docs.append(d)

        return self._merge(new_docs, first_sep)

    def _merge(self, parts: List[Document], sep: str) -> List[Document]:
        """Merges consecutive parts keeping *chunk_overlap* context.

        Args:
            parts: Sequence of Document pieces to merge.
            sep: Separator used to join fragments when rebuilding the
                buffer.

        Returns:
            List of merged :class:`Document` objects that satisfy the size
            constraint.
        """
        merged_texts: List[str] = []
        buffer = ""
        for part in parts:
            txt = part.text
            candidate = buffer + (sep if buffer else "") + txt
            if len(candidate) > self.chunk_size:
                merged_texts.append(buffer)
                overlap_text = buffer[-self.chunk_overlap :]
                last_space = overlap_text.rfind(" ")
                if last_space != -1:
                    overlap_text = overlap_text[last_space:]
                buffer = overlap_text + txt
            else:
                buffer = candidate
        if buffer:
            merged_texts.append(buffer)

        out: List[Document] = []
        for i, txt in enumerate(merged_texts):
            out.append(Document(id=f"{parts[0].id}-m{i}", text=txt, meta=parts[0].meta))
        return out
