import json
import os
from typing import List

import ollama
import typer
from pydantic import BaseModel, Field

from models.dialogue_pair import DialoguePair

SYSTEM_PROMPT = """
Ты - эксперт по генерации реалистичных диалогов между пользователем и ИИ-ассистентом на русском языке для тренировки систем Text-to-Speech (TTS).
Твоя задача: сгенерировать указанное количество пар "запрос пользователя - ответ ИИ" на заданную тему и вернуть результат в формате JSON.

Требования к запросам пользователя:
- Разнообразные формулировки (вопросы, просьбы, уточнения)
- Разная степень конкретности и сложности
- Естественная разговорная форма
- Реалистичность запросов - такие запросы мог бы задать обычный пользователь ИИ по заданной теме

Требования к ответам ИИ:
- Естественность и плавность речи (подходит для TTS-озвучивания)
- Разнообразие по длине и структуре
- Характерные для ИИ речевые обороты и стилистика
- Информативность и полезность
- Логическая завершенность
- Весь текст ответа ИИ должен быть представлен исключительно словами на русском языке, имитируя естественное звучание при чтении вслух. Это означает:
    - Транслитерацию иностранных названий, брендов и аббревиатур (например, "iOS" -> "ай о эс", "YouTube" -> "ютуб", "Wi-Fi" -> "вай-фай").
    - Написание всех чисел прописью, включая целые, дробные, порядковые и даты. (Например: "42" -> "сорок два"; "01.03.2025" -> "первое марта две тысячи двадцать пятого года"; "15.1" -> "пятнадцать точка один"; "1/2" -> "одна вторая").
    - Полное словесное описание всех математических формул, выражений, отдельных символов и их компонентов:
        - Каждая буква, обозначающая переменную или константу (например, из латинского или греческого алфавита), произносится как соответствующее название буквы или звука на русском языке (например, "x" -> "икс", "r" -> "эр", "S" -> "эс", "π" -> "пи", "α" -> "альфа", "v" -> "вэ").
        - Все математические знаки, операторы (сложение, вычитание, равенство и т.д.) и символы (например, корень, интеграл) пишутся словами (например, "+" -> "плюс", "=" -> "равно", "√" -> "квадратный корень из", "/" (в дроби) -> "деленое на" или в составе дроби как "одна вторая").
        - Показатели степени (экспоненты) всегда произносятся словами (например, "x²" должно быть представлено как "икс в квадрате" или "икс во второй степени"; "r³" -> "эр в кубе" или "эр в третьей степени"). Индексы у переменных также произносятся словами (например "v1" -> "вэ один").
        - Формулы целиком должны быть преобразованы в связные русскоязычные фразы, полностью избегая использования математических символов (типа π, r, S, ²), цифр в их знаковой записи или буквенных обозначений переменных в исходном виде (кроме как в примерах транслитерации). Например, выражение "S = πr²" должно быть передано как "эс равно пи эр в квадрате". Другой пример: "a=(v₂-v₁)/t" должно стать "а равно вэ два минус вэ один деленное на тэ".

Дополнительные требования:
- В 30% выходных данных должны присутствовать англицизмы, произнесенные по-русски (например, "iOS" -> "ай о эс", "Wi-Fi" -> "вай-фай", "YouTube" -> "ютуб")
- В 30% выходных данных должны присутствовать числовые представления, произнесенные словами (например, "01.03.2025" -> "первое марта две тысячи двадцать пятого", "15.1" -> "пятнадцать точка один", "42" -> "сорок два")

Твой ответ должен быть строго валидным JSON объектом следующей структуры:
{
  "pairs": [
    {
      "id": 1,
      "user_query": "Текст запроса пользователя",
      "ai_response": "Текст ответа ИИ"
    },
    {
      "id": 2,
      "user_query": "Текст запроса пользователя",
      "ai_response": "Текст ответа ИИ"
    }
    // и так далее
  ]
}

Не добавляй никаких пояснений, преамбул или заключений - только чистый JSON объект.
/no_think
"""

class TextGeneratedLLMResult(BaseModel):
    pairs: List[DialoguePair] = Field(..., description="Пары запрос-ответ")


def generate_dialogue(
        topic: str,
        ollama_client: ollama.Client,
        model_name: str,
        num_samples: int = 5,
        temperature: float = 0.7
) -> List[DialoguePair]:
    if not topic or not topic.strip():
        raise ValueError("topic cannot be empty")
    user_prompt = f"Сгенерируй {num_samples} пар запрос-ответ на тему: \"{topic}\""

    response = ollama_client.chat(
        model=model_name,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        options={"temperature": temperature},
        format=TextGeneratedLLMResult.model_json_schema()
    )
    json_content = json.loads(response['message']['content'])
    return TextGeneratedLLMResult.model_validate(json_content).pairs


def generate_multiple_topics(
        topics_list: List[str],
        output_path: str,
        ollama_client: ollama.Client,
        model_name: str,
        num_samples: int = 5,
        temperature: float = 0.7
):
    """
    Генерирует диалоги для нескольких тем и сохраняет результаты в jsonl файл.
    """
    total_topics_count = len(topics_list)
    typer.echo(f"Начало генерации диалогов для {total_topics_count} тем...")

    for index, current_topic in enumerate(topics_list):
        typer.echo(f"\nОбработка темы {index + 1}/{total_topics_count}: '{current_topic}'...")
        if not current_topic.strip():
            typer.echo("Пропуск пустой темы.")
            continue

        pairs: List[DialoguePair] = generate_dialogue(
            current_topic,
            ollama_client,
            num_samples=num_samples,
            model_name=model_name,
            temperature=temperature
        )


        topic_filename = f"{current_topic.replace(' ', '_')[:30]}.jsonl"
        topic_filepath = os.path.join(output_path, topic_filename)
        topic_filepath_mode = 'w' if os.path.exists(topic_filepath) == 0 else 'a'
        with open(topic_filepath, topic_filepath_mode) as topic_file:
            for pair in pairs:
                topic_file.write(pair.to_jsonl())

        typer.echo(typer.style(f"Диалоги для темы '{current_topic}' сохранены в {topic_filepath}", fg=typer.colors.GREEN))