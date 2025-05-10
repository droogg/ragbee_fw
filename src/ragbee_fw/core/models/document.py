from dataclasses import dataclass
from typing import Optional


@dataclass
class Document:
    id: str
    text: str
    meta: Optional[dict] = None
