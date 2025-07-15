from ragbee_fw.infrastructure.di_conteiner.di_conteiner import (
    ADAPTER_REGISTRY,
    DIContainer,
)

from .config.load_config import load_config
from .core.services.answer_service import AnswerService
from .core.services.ingestion_service import IngestionService

__version__ = "0.1.0"
__all__ = [
    "__version__",
    "IngestionService",
    "AnswerService",
    "DIContainer",
    "ADAPTER_REGISTRY",
    "load_config",
]
