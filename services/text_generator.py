# text_generator.py
import json
import os
from typing import List

import ollama
import typer
from pydantic import BaseModel, Field

from models.base_llm_client import BaseLLMClient
from models.dialogue_pair import DialoguePair

GENERATION_PROMPT = """
Ты - эксперт по генерации реалистичных диалогов между пользователем и ИИ-ассистентом на русском языке для тренировки систем Text-to-Speech (TTS).

Твоя задача: сгенерировать указанное количество пар "запрос пользователя - ответ ИИ" на заданную тему и вернуть результат в формате JSON.
    
Требования к диалогам:
- Естественность и реалистичность
- Разнообразие формулировок и длины ответов
- Информативность и полезность ответов ИИ
- Подходящий стиль для озвучивания

Примеры хороших диалогов:

Пример 1:
User: "Какие продукты помогают улучшить память?"
AI: "Для улучшения памяти полезны продукты, богатые омега-3 жирными кислотами - это жирная рыба, грецкие орехи, льняное семя. Также важны ягоды, особенно черника и голубика, которые содержат антиоксиданты. Темный шоколад с высоким содержанием какао стимулирует работу мозга. Не забывайте про зеленый чай и продукты с витаминами группы B - яйца, авокадо, цельнозерновые крупы."

Пример 2:
User: "Как правильно готовиться к марафону?"
AI: "Подготовка к марафону требует систематического подхода. Начните минимум за 16-20 недель до забега. Постепенно увеличивайте дистанцию - каждую неделю добавляйте не более 10%. Обязательно включите в программу длительные забеги по выходным, интервальные тренировки и дни отдыха. Уделите внимание правильному питанию - увеличьте потребление сложных углеводов. За неделю до марафона снизьте нагрузку. И конечно, подберите удобную обувь заранее."

Пример 3:
User: "Расскажи про пользу медитации"
AI: "Медитация - это практика, которая приносит множество преимуществ для физического и ментального здоровья. Регулярная медитация снижает уровень стресса и тревожности, улучшает концентрацию внимания и память. Исследования показывают, что она помогает нормализовать артериальное давление и улучшить качество сна. Даже 10-15 минут ежедневной практики могут значительно повысить эмоциональную устойчивость и общее самочувствие. Начать можно с простых дыхательных техник или использовать специальные приложения для медитации."

Твой ответ должен быть строго валидным JSON объектом:
{
  "pairs": [
    {
      "id": 1,
      "user_query": "Текст запроса пользователя",
      "ai_response": "Текст ответа ИИ"
    }
  ]
}

Не добавляй никаких пояснений - только JSON.
"""

NUMBERS_TO_WORDS_PROMPT = """
Ты - эксперт по адаптации текста для систем синтеза речи (TTS). 

Твоя задача: преобразовать все числа, цифры и специальные обозначения в тексте в их словесное представление на русском языке.

Правила преобразования:
1. Числа → слова:
   - 70% → семьдесят процентов
   - 1-2 часа → один-два часа
   - 10-15 минут → десять-пятнадцать минут
   - 16-20 недель → шестнадцать-двадцать недель

2. Специальные термины с цифрами:
   - омега-3 → омега-три
   - витамин B12 → витамин бэ двенадцать
   - витамин D3 → витамин дэ три
   - COVID-19 → ковид-девятнадцать

3. Англоязычные термины:
   - Wi-Fi → вай-фай
   - iOS → ай о эс
   - USB → ю эс би

4. Единицы измерения с числами:
   - 5 км → пять километров
   - 10 мг → десять миллиграммов
   - 2 л → два литра

5. Временные обозначения:
   - 24/7 → двадцать четыре на семь
   - 90-е годы → девяностые годы
   - XXI век → двадцать первый век

Важно: сохрани естественность текста и контекст. Если встретишь числа в составе устойчивых выражений, преобразуй их соответствующим образом.

Верни результат в формате JSON:
{
  "converted_text": "текст с замененными числами"
}
"""


class TextGeneratedLLMResult(BaseModel):
    pairs: List[DialoguePair] = Field(..., description="Пары запрос-ответ")

class ConvertedTextResult(BaseModel):
    converted_text: str = Field(..., description="Текст с замененными числами")


def convert_numbers_to_words(
    text: str, llm_client: BaseLLMClient, temperature: float = 0.3
) -> str:
    user_prompt = f"Преобразуй все числа и цифры в следующем тексте в их словесное представление:\n\n{text}"

    messages = [
        {"role": "system", "content": NUMBERS_TO_WORDS_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    response = llm_client.chat(
        messages=messages,
        temperature=temperature,
        response_format=ConvertedTextResult.model_json_schema(),
    )

    json_content = json.loads(response)
    return ConvertedTextResult.model_validate(json_content).converted_text


def generate_dialogue(
    topic: str,
    llm_client: BaseLLMClient,
    num_samples: int = 5,
    temperature: float = 0.7,
    convert_numbers: bool = True,
) -> List[DialoguePair]:
    if not topic or not topic.strip():
        raise ValueError("topic cannot be empty")

    user_prompt = f'Сгенерируй {num_samples} пар запрос-ответ на тему: "{topic}"'

    messages = [
        {"role": "system", "content": GENERATION_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    # Генерация диалогов
    response = llm_client.chat(
        messages=messages,
        temperature=temperature,
        response_format=TextGeneratedLLMResult.model_json_schema(),
    )

    json_content = json.loads(response)
    pairs = TextGeneratedLLMResult.model_validate(json_content).pairs

    # Обработка ai_response для замены цифр на слова
    if convert_numbers:
        typer.echo("Обработка ответов для замены цифр на слова...")
        for pair in pairs:
            pair.ai_response = convert_numbers_to_words(
                pair.ai_response,
                llm_client,
                temperature=0.3,  # Низкая температура для точности
            )

    return pairs


def generate_multiple_topics(
    topics_list: List[str],
    output_path: str,
    llm_client: BaseLLMClient,
    num_samples: int = 5,
    temperature: float = 0.7,
):
    """
    Генерирует диалоги для нескольких тем и сохраняет результаты в jsonl файл.
    """
    total_topics_count = len(topics_list)
    typer.echo(f"Начало генерации диалогов для {total_topics_count} тем...")

    for index, current_topic in enumerate(topics_list):
        typer.echo(
            f"\nОбработка темы {index + 1}/{total_topics_count}: '{current_topic}'..."
        )
        if not current_topic.strip():
            typer.echo("Пропуск пустой темы.")
            continue

        pairs: List[DialoguePair] = generate_dialogue(
            current_topic, llm_client, num_samples=num_samples, temperature=temperature
        )

        topic_filename = f"{current_topic.replace(' ', '_')[:30]}.jsonl"
        topic_filepath = os.path.join(output_path, topic_filename)
        topic_filepath_mode = "w" if not os.path.exists(topic_filepath) else "a"

        with open(topic_filepath, topic_filepath_mode) as topic_file:
            for pair in pairs:
                topic_file.write(pair.to_jsonl())

        typer.echo(typer.style(f"Диалоги для темы '{current_topic}' сохранены в {topic_filepath}", fg=typer.colors.GREEN))
