"""Filesystem data loader for RAGBee FW.

This module contains :class:`FileSystemLoader`, an implementation of
:class:`~src.ragbee_fw.core.ports.data_loader_port.DataLoaderPort` that
recursively scans a directory (or ingests a single file) and returns a
list of :class:`~src.ragbee_fw.core.models.document.Document` objects.

Supported file types
--------------------
- Plain‑text files with the ``.txt`` extension
- Markdown files with the ``.md`` extension

All other files are ignored.
"""

from pathlib import Path
from typing import List, Union

from charset_normalizer import from_bytes

from ragbee_fw.core.models.document import Document
from ragbee_fw.core.ports.data_loader_port import BaseDataLoader


class FileSystemLoader(BaseDataLoader):
    """Load text/markdown documents from the local filesystem.

    The loader accepts either a single path to a file or a directory.
    When a directory path is provided, it traverses the directory
    recursively (``Path.rglob``) to collect ``*.txt`` and ``*.md`` files.

    Each discovered file is read using the best‑guess text encoding from
    **charset-normalizer**. Files that cannot be read (e.g. permission
    errors, binary blobs) are skipped with a warning printed to ``stdout``.

    Args:
        path: A path to a text/markdown file or a directory containing
            such files. May be a ``str`` or :class:`pathlib.Path`.

    Returns:
        list[Document]: A list of :class:`~src.ragbee_fw.core.models.document.Document`
        instances where ``Document.text`` holds the file contents and
        ``Document.meta`` stores ``{"source": <file-path>, "encoding": <str>}``.

    Raises:
        RuntimeError: If *path* does not exist on the filesystem.

    Examples
    --------
    >>> loader = FileSystemLoader()
    >>> docs = loader.load("/data/wiki")
    >>> docs[0].meta
    {'source': '/data/wiki/ai.md', 'encoding': 'utf-8'}
    """

    def load(self, path: Union[str, Path]) -> List[Document]:
        """Read all eligible files under *path* and convert to Documents.

        Args:
            path: Filesystem location to load. If it is a directory, all
                ``*.txt`` and ``*.md`` files found recursively will be
                included. If it is a single file, only that file will be
                processed.

        Returns:
            list[Document]: A list of document objects representing the
            contents of each successfully read file.

        Raises:
            RuntimeError: If the supplied *path* does not exist.
        """
        path = Path(path)
        if not path.exists():
            raise RuntimeError(f"The directory '{path}' does not exist")
        if path.is_file():
            files = [path]
        else:
            files = list(path.rglob("*.txt")) + list(path.rglob("*.md"))

        docs: List[Document] = []
        for idx, file_path in enumerate(files):
            try:
                raw = file_path.read_bytes()
                guess = from_bytes(raw).best()
                enc = guess.encoding if guess else "utf-8"
                text = raw.decode(encoding=enc, errors="replace")
            except Exception as e:
                print(f"Error reading file {file_path.name}: {e}")
                continue

            docs.append(
                Document(
                    id=str(idx),
                    text=text,
                    meta={"source": str(file_path), "encoding": enc},
                )
            )
        return docs
