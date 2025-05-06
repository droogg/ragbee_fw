from abc import ABC, abstractmethod


class LLMPort(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generating a response on prompt"""
        raise NotImplementedError("Subclasses must implement this method")
