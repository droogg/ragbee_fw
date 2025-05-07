"""LLM port and base class for LLM adapters"""

from abc import ABC, abstractmethod
from typing import Protocol


class LLMPort(Protocol):
    def generate(self, prompt: str) -> str: ...


class BaseLLM(LLMPort, ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generating a response on prompt"""
        raise NotImplementedError("Subclasses must implement this method")
