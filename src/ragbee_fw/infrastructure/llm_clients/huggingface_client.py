"""Adapters to connect to Hugging Face Inference API for LLM generation.

This module defines :class:`HuggingFaceInferenceAdapter`, a thin wrapper
around :class:`huggingface_hub.InferenceClient` that lets the application
work with any Hugging Face–hosted large‑language model (LLM) through
RAGBee's :class:`BaseLLM` port.
"""

from typing import Dict, List, Optional, Tuple, Union

from huggingface_hub import ChatCompletionOutput, InferenceClient

from ragbee_fw.core.ports.llm_port import BaseLLM


class HuggingFaceInferenceAdapter(BaseLLM):
    """Adapter that bridges :class:`BaseLLM` to the Hugging Face Inference API.

    The adapter wraps :class:`huggingface_hub.InferenceClient` and exposes a
    :py:meth:`generate` method that sends chat‑completion requests to the
    configured model hosted on the Hugging Face Inference Endpoints
    service.

    Args:
        model_name: The identifier of the HF model to query (e.g.
            ``"meta-llama/Llama-3-8b-instruct"``, or ``None`` if the
            project‑level default should be used).
        provider: Name of the inference provider. Pass ``"auto"`` to let
            Hugging Face pick the best available provider or ``None`` to stick
            with the default.
        token: A valid Hugging Face access token with permission to call the
            endpoint, or ``None`` to rely on the environment variable
            ``HF_TOKEN``.
        base_url: Fully qualified base URL to a self‑hosted
            text‑generation‑inference (TGI) instance. Leave ``None`` to target
            the official HF Inference API."""

    def __init__(
        self,
        model_name: str,
        token: Optional[str] | None = None,
        provider: Optional[str] = "auto",
        base_url: Optional[str] = None,
    ) -> None:
        super().__init__(model_name=model_name)
        self.client = InferenceClient(
            model=model_name,
            token=token,
            provider=provider,
            base_url=base_url,
        )

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 2048,
        return_full_response: bool = False,
    ) -> Union[str | None, Tuple[str | None, ChatCompletionOutput | str | None]]:
        """Generate a completion for *prompt*.

        Args:
            prompt: User prompt to send to the model.
            max_new_tokens: Maximum number of tokens the model is allowed to
                generate. Defaults to *256*.
            return_full_response: If *True*, return a tuple ``(text,
                raw_response)`` where *text* is the generated string (or
                *None* if absent) and *raw_response* is the full
                :class:`huggingface_hub.ChatCompletionOutput`. If *False*,
                return only *text*.

        Returns:
            Either the generated text or a tuple of *(text,
            ChatCompletionOutput)* depending on *return_full_response*.
        """
        messages: List[Dict[str, str]] = [{"role": "user", "content": prompt}]
        output: ChatCompletionOutput = self.client.chat.completions.create(
            messages=messages,
            max_tokens=max_new_tokens,
        )
        if not output.choices:
            raise RuntimeError("No choices returned from the model")
        text: Optional[str] = output.choices[0].message.content
        return (text, output) if return_full_response else text
