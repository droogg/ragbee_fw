from typing import Any, Dict, List, Literal, Optional, Union

from huggingface_hub import InferenceClient

from src.ragbee_fw.core.ports.llm_port import BaseLLM


class HuggingFaceInferenceAdapter(BaseLLM):
    def __init__(
        self,
        model_name: Optional[str],
        provider: Union[Literal["auto"], None],
        token: Optional[str],
        base_url: Optional[str] = None,
    ) -> None:
        super().__init__(model_name=model_name)
        self.client = InferenceClient(
            model=model_name,
            provider=provider,
            token=token,
            base_url=base_url,
        )

    def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        messages: List[Dict[str, str]] = [{"role": "user", "content": prompt}]
        output: Dict[str, Any] = self.client.chat.completions.create(
            messages=messages,
            max_tokens=max_new_tokens,
        )
        return output.choices[0].message.content
