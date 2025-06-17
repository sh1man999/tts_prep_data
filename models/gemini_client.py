import json
from typing import Optional, Any
import google.generativeai as genai
from models.base_llm_client import BaseLLMClient


class GeminiClient(BaseLLMClient):
    def __init__(self, api_key: str, model_name: str = "gemini-pro"):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self.model_name = model_name

    def chat(self,
             messages: list[dict[str, str]],
             temperature: float = 0.7,
             response_format: Optional[dict[str, Any]] = None) -> str:
        # Конвертируем формат сообщений для Gemini
        prompt = ""
        for msg in messages:
            if msg["role"] == "system":
                prompt += f"Инструкции: {msg['content']}\n\n"
            else:
                prompt += f"{msg['content']}\n\n"

        if response_format:
            prompt += f"\nОтветь в формате JSON согласно схеме: {json.dumps(response_format)}"

        generation_config = genai.GenerationConfig(
            temperature=temperature,
            max_output_tokens=8192,
        )

        response = self.model.generate_content(
            prompt,
            generation_config=generation_config
        )

        return response.text