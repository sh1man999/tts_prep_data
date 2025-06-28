from typing import Optional, Any

import typer
from google import genai
import outlines

from models.base_llm_client import BaseLLMClient



class GeminiClient(BaseLLMClient):
    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash"):
        client = genai.Client(api_key=api_key)
        self.model = outlines.from_gemini(
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
        result = self.model(prompt, response_format, max_output_tokens=65000)
        return result