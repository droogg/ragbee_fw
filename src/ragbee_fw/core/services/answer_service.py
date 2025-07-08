"""
answer_service.py

This module implements the AnswerService, which coordinates retrieval of
relevant document fragments and generation of natural language answers via
an LLM. It relies on ports defined in the core layer, allowing interchangeable
retriever and LLM implementations.
"""

from typing import List

from ragbee_fw.core.models.document import Document
from ragbee_fw.core.ports.llm_port import LLMPort
from ragbee_fw.core.ports.retriever_port import RetrieverPort


class AnswerService:
    """Service for generating answers to user queries using a RAG workflow.

    AnswerService performs the following steps:
      1. Retrieves the top_k most relevant Document fragments for a query.
      2. Formats these fragments into a prompt context.
      3. Sends the prompt to an LLM via LLMPort.
      4. Returns the generated answer.

    Attributes:
        retriever: An implementation of RetrieverPort used to find relevant fragments.
        llm: An implementation of LLMPort used to generate answers.
    """

    def __init__(
        self,
        retriever: RetrieverPort,
        llm: LLMPort,
    ) -> None:
        """Initialize the AnswerService.

        Args:
            retriever: A RetrieverPort responsible for retrieving Document fragments.
            llm: An LLMPort responsible for generating text responses.
        """
        self.retriever = retriever
        self.llm = llm

    def generate_answer(
        self, query: str, top_k: int = 5, prompt_template: str | None = None
    ) -> str:
        """Generate a natural language answer for the given query.

        This method:
          1. Retrieves up to top_k Document fragments matching the query.
          2. Builds a prompt by concatenating the fragments.
          3. Optionally formats the prompt using a user-provided template.
          4. Invokes the LLM to generate and return an answer.

        Args:
            query: The user’s question as a string.
            top_k: The maximum number of fragments to retrieve for context.
            prompt_template: An optional format string with two placeholders:
                             `{context}` and `{query}`. If provided, it is used to
                             build the prompt. Example:
                             "Context:\n\n{context}\n\nPlease answer: {query}"

        Returns:
            A string containing the LLM-generated answer.

        Example:
            >>> from src.ragbee_fw.infrastructure.retriever.bm25_client import BM25Client
            >>> from src.ragbee_fw.infrastructure.llm_clients.huggingface_adapter import HuggingFaceInferenceAdapter
            >>> from src.ragbee_fw.core.services.answer_service import AnswerService
            >>> retriever = BM25Client.from_documents(chunks)
            >>> llm = HuggingFaceInferenceAdapter(model_name="gpt-3.5-turbo", token="…")
            >>> service = AnswerService(retriever, llm)
            >>> answer = service.generate_answer("What is RAG?", top_k=3)
            >>> print(answer)
        """
        docs: List[Document] = self.retriever.retrieve(query, top_k)
        context = "\n\n".join(f"[i + 1] {doc.text}" for i, doc in enumerate(docs))
        if prompt_template:
            prompt = prompt_template.format(context=context, query=query)
        else:
            prompt = (
                "Based on the following fragments:\n\n"
                f"{context}\n\n"
                f"Answer the question: «{query}»"
            )
        answer = self.llm.generate(prompt)
        return answer
