from typing import Optional, Any

from models.base_llm_client import BaseLLMClient


class OllamaClient(BaseLLMClient):
    def __init__(self, base_url: str, model_name: str):
        import ollama

        self.client = ollama.Client(host=base_url)
        self.model_name = model_name

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        response_format: Optional[dict[str, Any]] = None,
    ) -> str:

        response = self.client.chat(
            model=self.model_name,
            messages=messages,
            options={"temperature": temperature, "num_ctx": 32768},
            format=response_format,
        )

        return response["message"]["content"]