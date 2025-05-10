"""LLM port and base class for LLM adapters"""

from abc import ABC, abstractmethod
from typing import Optional, Protocol, Tuple, Union

from huggingface_hub import ChatCompletionOutput


class LLMPort(Protocol):
    def generate(
        self, prompt: str
    ) -> Union[str | None, Tuple[str | None, ChatCompletionOutput | str | None]]: ...


class BaseLLM(LLMPort, ABC):
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    @abstractmethod
    def generate(
        self, prompt: str
    ) -> Union[str | None, Tuple[str | None, ChatCompletionOutput | str | None]]:
        """Generating a response on prompt"""
        raise NotImplementedError("Subclasses must implement this method")
