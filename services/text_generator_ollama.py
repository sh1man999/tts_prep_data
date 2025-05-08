import json
import os
from typing import List

import ollama
import pandas as pd
import typer
from pydantic import BaseModel, Field


class DialoguePair(BaseModel):
    id: int = Field(..., description="Уникальный идентификатор пары")
    user_query: str = Field(..., description="Запрос пользователя")
    ai_response: str = Field(..., description="Ответ ИИ на запрос")


class DialogueGeneration(BaseModel):
    topic: str = Field(..., description="Тема диалогов")
    pairs: List[DialoguePair] = Field(..., description="Пары запрос-ответ")


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

Твой ответ должен быть строго валидным JSON объектом следующей структуры:
{
  "topic": "Заданная тема",
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
    // и так далее до 5 пар
  ]
}

Не добавляй никаких пояснений, преамбул или заключений - только чистый JSON объект.
/no_think
"""


def generate_dialogue_with_ollama(
        topic: str,
        ollama_client: ollama.Client,
        num_samples: int = 5,
        model_name: str = "qwen3:30b",
        temperature: float = 0.7
) -> DialogueGeneration:
    if not topic or not topic.strip():
        return DialogueGeneration(
            topic="",
            pairs=[]
        )
    user_prompt = f"Сгенерируй {num_samples} пар запрос-ответ на тему: \"{topic}\""

    response = ollama_client.chat(
        model=model_name,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        options={"temperature": temperature}
    )
    content = response['message']['content']

    cleaned_content = content.strip()
    if "</think>" in cleaned_content:
        think_parts = cleaned_content.split("</think>")
        if len(think_parts) > 1:
            cleaned_content = think_parts[1]
        else:
            cleaned_content = think_parts[0]

    if cleaned_content.startswith("```json"):
        cleaned_content = cleaned_content[7:]
        if cleaned_content.endswith("```"):
            cleaned_content = cleaned_content[:-3]
    elif cleaned_content.startswith("```"):
        cleaned_content = cleaned_content[3:]
        if cleaned_content.endswith("```"):
            cleaned_content = cleaned_content[:-3]

    cleaned_content = cleaned_content.strip()

    json_data = json.loads(cleaned_content)
    result = DialogueGeneration(
        topic=json_data.get("topic", topic),
        pairs=[
            DialoguePair(
                id=pair.get("id", i + 1),
                user_query=pair.get("user_query", ""),
                ai_response=pair.get("ai_response", "")
            )
            for i, pair in enumerate(json_data.get("pairs", []))
        ]
    )
    return result


def generate_multiple_topics(
        topics_list: List[str],
        output_excel_path: str,
        ollama_client: ollama.Client,
        model_name: str,
        num_samples: int = 5,
        temperature: float = 0.7
):
    """
    Генерирует диалоги для нескольких тем и сохраняет результаты в Excel файл.
    """
    results_summary = []
    total_topics_count = len(topics_list)
    typer.echo(f"Начало генерации диалогов для {total_topics_count} тем...")

    for index, current_topic in enumerate(topics_list):
        typer.echo(f"\nОбработка темы {index + 1}/{total_topics_count}: '{current_topic}'...")
        if not current_topic.strip():
            typer.echo("Пропуск пустой темы.")
            continue

        dialogue_result = generate_dialogue_with_ollama(
            current_topic,
            ollama_client,
            num_samples=num_samples,
            model_name=model_name,
            temperature=temperature
        )


        results_summary.append({
            'Topic': current_topic,
            'Pairs_Generated': len(dialogue_result.pairs),
        })

        # Create dataframe with dialogue pairs for this topic
        pairs_data = []
        for i, pair in enumerate(dialogue_result.pairs):
            pairs_data.append({
                'Pair_Number': pair.id,
                'User_Query': pair.user_query,
                'AI_Response': pair.ai_response

            })

        topic_filename = f"{current_topic.replace(' ', '_')[:30]}.xlsx"
        topic_filepath = os.path.join(os.path.dirname(output_excel_path), topic_filename)

        # Save to topic-specific Excel file
        if pairs_data:
            pairs_df = pd.DataFrame(pairs_data)
            pairs_df.to_excel(topic_filepath, index=False)
            typer.echo(f"Диалоги для темы '{current_topic}' сохранены в {topic_filepath}")


        typer.echo(typer.style(
            f"Тема '{current_topic}' обработана успешно. Сгенерировано пар: {len(dialogue_result.pairs)}",
            fg=typer.colors.GREEN))
        if dialogue_result.pairs:
            first_pair = dialogue_result.pairs[0]
            typer.echo(
                f"Пример:\nПользователь: {first_pair.user_query[:70]}...\nИИ: {first_pair.ai_response[:70]}...")
    output_dir = os.path.dirname(output_excel_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with pd.ExcelWriter(output_excel_path, engine='openpyxl') as writer:
        topics_df = pd.DataFrame(results_summary)
        topics_df.to_excel(writer, sheet_name='Topics_Summary', index=False)

    typer.echo(typer.style(f"\nГенерация завершена. Результаты сохранены в {output_excel_path}",
                           fg=typer.colors.BRIGHT_GREEN))

