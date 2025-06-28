from entrypoint.config import GEMINI_TOKEN, OPENROUTER_TOKEN
from models.base_llm_client import BaseLLMClient
from models.deep_seek_client import DeepSeekClient
from models.gemini_client import GeminiClient
from models.llm_provider import LLMProvider
from models.ollama_client import OllamaClient
from models.openrouter_client import OpenRouterClient


# Фабрика для создания клиентов
def create_llm_client(provider: LLMProvider, **kwargs) -> BaseLLMClient:
    """
    Создает LLM клиент в зависимости от провайдера

    Args:
        provider: Тип провайдера (ollama, deepseek, gemini, openai)
        **kwargs: Параметры для инициализации клиента
    """
    if provider == LLMProvider.OLLAMA:
        return OllamaClient(
            model_name=kwargs.get("model_name", "qwen3:30b-a3b")
        )
    elif provider == LLMProvider.DEEPSEEK:
        return DeepSeekClient(
            api_key=GEMINI_TOKEN,
            model_name=kwargs.get("model_name", "deepseek-reasoner")
        )
    elif provider == LLMProvider.GEMINI:
        return GeminiClient(
            api_key=GEMINI_TOKEN,
            model_name=kwargs.get("model_name", "gemini-pro")
        )
    elif provider == LLMProvider.OPENROUTER:
        return OpenRouterClient(
            api_key=OPENROUTER_TOKEN,
            model_name=kwargs.get("model_name", "openai/gpt-4.1")
        )
    else:
        raise ValueError(f"Неподдерживаемый провайдер: {provider}")
