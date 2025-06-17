import json
from typing import Optional, Any

from models.base_llm_client import BaseLLMClient


class DeepSeekClient(BaseLLMClient):
    def __init__(self, api_key: str, model_name: str):
        from openai import OpenAI
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com"
        )
        self.model_name = model_name

    def chat(self,
             messages: list[dict[str, str]],
             temperature: float = 0.7,
             response_format: Optional[dict[str, Any]] = None) -> str:

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            response_format={"type": "json_object"},
            stream=False
        )
        return response.choices[0].message.content