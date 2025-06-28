from typing import Any
from google import genai
import outlines
from openai import OpenAI

from models.base_llm_client import BaseLLMClient



class OpenRouterClient(BaseLLMClient):
    def __init__(self, api_key: str, model_name: str = "openai/gpt-4.1"):
        client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
        self.model = outlines.from_openai(
            client,
            model_name
        )

    def chat(self,
             messages: list[dict[str, str]],
             temperature: float = 0.7,
             response_format: Any = None) -> str:
        # Конвертируем формат сообщений для Gemini
        prompt = ""
        for msg in messages:
            if msg["role"] == "system":
                prompt += f"Инструкции: {msg['content']}\n\n"
            else:
                prompt += f"{msg['content']}\n\n"
        result = self.model(prompt, response_format, max_tokens=32000)
        return result