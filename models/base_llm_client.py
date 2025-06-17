from abc import ABC, abstractmethod
from typing import Optional, Any


class BaseLLMClient(ABC):
    """Базовый класс для всех LLM клиентов"""

    @abstractmethod
    def chat(self,
             messages: list[dict[str, str]],
             temperature: float = 0.7,
             response_format: Optional[dict[str, Any]] = None) -> str:
        """
        Отправляет сообщения в LLM и возвращает ответ

        Args:
            messages: Список сообщений в формате [{"role": "system/user", "content": "..."}]
            temperature: Температура генерации
            response_format: Схема ожидаемого формата ответа (для структурированного вывода)

        Returns:
            Строка с ответом модели
        """