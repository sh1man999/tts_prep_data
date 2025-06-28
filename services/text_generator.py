# text_generator.py
import os
from typing import List, Generator
import typer
from pydantic import BaseModel, Field
from tqdm import tqdm

from models.base_llm_client import BaseLLMClient
from models.dialogue_pair import DialoguePair

GENERATION_PROMPT = """
Ты - эксперт по генерации реалистичных диалогов между пользователем и ИИ-ассистентом на русском языке для тренировки систем Text-to-Speech (TTS).

Твоя задача: сгенерировать указанное количество пар "запрос пользователя - ответ ИИ" на заданную тему и вернуть результат в формате JSON.

Требования к диалогам:
- Естественность и реалистичность
- Разнообразие формулировок и длины ответов (от кратких до развернутых, НО НЕ БОЛЕЕ 350 символов)
- Информативность и полезность ответов ИИ
- Подходящий стиль для озвучивания (избегай сложных терминов без пояснений)

Примеры хороших диалогов разной длины:

Пример 1:
User: "Какая погода будет завтра?"
AI: "К сожалению, я не имею доступа к актуальным данным о погоде. Рекомендую проверить прогноз погоды в интернете."

Пример 2:
User: "Что такое блокчейн простыми словами?"
AI: "Блокчейн - это цифровая книга записей, которую невозможно подделать. Каждая новая запись связана с предыдущей, образуя цепочку. Эта технология обеспечивает прозрачность и безопасность данных."

Пример 3:
User: "Какие продукты помогают улучшить память?"
AI: "Для улучшения памяти полезны продукты, богатые омега-три жирными кислотами - это жирная рыба, грецкие орехи, льняное семя. Также важны ягоды, особенно черника и голубика, которые содержат антиоксиданты. Темный шоколад с высоким содержанием какао стимулирует работу мозга. Не забывайте про зеленый чай и авокадо."

Пример 4:
User: "Помоги выбрать подарок"
AI: "С удовольствием помогу! Для кого выбираете подарок - мужчины или женщины? Какой возраст? И какой примерно бюджет вы рассматриваете для покупки?"

Пример 5:
User: "Как быстро заснуть?"
AI: "Попробуйте технику четыре-семь-восемь: вдохните через нос на четыре счета, задержите дыхание на семь, выдохните через рот на восемь. Повторите три-четыре раза. Также помогает проветрить комнату и выключить все экраны за час до сна."

Пример 6:
User: "Как сварить идеальные яйца всмятку?"
AI: "Вот простой способ: доведите воду до кипения, аккуратно опустите яйца комнатной температуры. Варите ровно шесть минут для жидкого желтка или семь минут для более густого. Сразу переложите в холодную воду на минуту. Очистите под холодной водой - скорлупа снимется легко."

Пример 7:
User: "Почему небо голубое?"
AI: "Небо выглядит голубым из-за рассеяния солнечного света в атмосфере. Молекулы воздуха рассеивают короткие волны синего цвета сильнее, чем длинные волны красного. Поэтому мы видим небо голубым в ясный день."

Пример 8:
User: "Чувствую себя уставшим весь день"
AI: "Понимаю, как это неприятно. Постоянная усталость может быть связана с недосыпом, стрессом или недостатком витаминов. Попробуйте наладить режим сна и добавить прогулки на свежем воздухе."

Пример 9:
User: "Какие навыки важны для программиста?"
AI: "Ключевые навыки программиста: логическое мышление, знание алгоритмов, умение решать задачи, работа в команде, английский язык для чтения документации. Также важны усидчивость и готовность постоянно учиться новому."

Пример 10:
User: "Болит голова, что делать?"
AI: "Попробуйте выпить стакан воды - часто головная боль связана с обезвоживанием. Помогает легкий массаж висков и затылка. Если боль сильная или частая, обязательно обратитесь к врачу."

ВАЖНО: Все ответы ИИ должны быть НЕ БОЛЕЕ 350 символов!

Твой ответ должен быть строго валидным JSON объектом:
{
  "pairs": [
    {
      "id": 1,
      "user_query": "Текст запроса пользователя",
      "ai_response": "Текст ответа ИИ (максимум 350 символов)"
    }
  ]
}

Не добавляй никаких пояснений - только JSON.
"""


class TextGeneratedLLMResult(BaseModel):
    pairs: List[DialoguePair] = Field(..., description="Пары запрос-ответ")


def generate_dialogue(
    topic: str,
    llm_client: BaseLLMClient,
    batch_size: int,
    num_samples: int = 5,
    temperature: float = 0.7,
) -> Generator[List[DialoguePair], None, None]:
    """
    Генератор диалогов батчами с использованием контекста предыдущих пар.
    Yield'ит батчи по мере генерации.
    """
    if not topic or not topic.strip():
        raise ValueError("topic cannot be empty")

    last_pair = None
    generated_count = 0

    # Прогресс бар для батчей
    with tqdm(
        total=num_samples, desc=f"Генерация для '{topic[:30]}'", unit="пар"
    ) as pbar:
        while generated_count < num_samples:
            # Вычисляем размер текущего батча
            current_batch_size = min(batch_size, num_samples - generated_count)

            # Формируем промпт с учетом последней пары
            if last_pair:
                user_prompt = (
                    f'Сгенерируй {current_batch_size} пар запрос-ответ на тему: "{topic}"\n\n'
                    f"Последняя сгенерированная пара:\n"
                    f"{last_pair.to_jsonl()}\n"
                    f"Продолжай в том же стиле, но создавай новые уникальные пары."
                )
            else:
                user_prompt = f'Сгенерируй {current_batch_size} пар запрос-ответ на тему: "{topic}"'

            messages = [
                {"role": "system", "content": GENERATION_PROMPT},
                {"role": "user", "content": user_prompt},
            ]

            # Генерируем батч
            response = llm_client.chat(
                messages=messages,
                temperature=temperature,
                response_format=TextGeneratedLLMResult,
            )

            batch_pairs = TextGeneratedLLMResult.model_validate_json(response).pairs

            # Сохраняем последнюю пару для следующего батча
            if batch_pairs:
                last_pair = batch_pairs[-1]
                generated_count += len(batch_pairs)

                # Обновляем прогресс
                pbar.update(len(batch_pairs))

                # Yield'им батч для немедленной обработки
                yield batch_pairs

def generate_multiple_topics(
    topics_list: List[str],
    output_path: str,
    llm_client: BaseLLMClient,
    batch_size: int,
    num_samples: int = 5,
    temperature: float = 0.7,
):
    """
    Генерирует диалоги для нескольких тем и сохраняет результаты в jsonl файлы.
    """
    total_topics_count = len(topics_list)
    typer.echo(f"Начало генерации диалогов для {total_topics_count} тем...")

    # Основной прогресс бар для тем
    for index, current_topic in enumerate(
        tqdm(topics_list, desc="Обработка тем", unit="тема")
    ):
        typer.echo(
            f"\n\nОбработка темы {index + 1}/{total_topics_count}: '{current_topic}'..."
        )

        if not current_topic.strip():
            typer.echo("Пропуск пустой темы.")
            continue

        # Создаем имя файла для темы
        topic_filename = f"{current_topic.replace(' ', '_')[:30]}.jsonl"
        topic_filepath = os.path.join(output_path, topic_filename)
        topic_filepath_mode = "w" if not os.path.exists(topic_filepath) else "a"

        # Генерируем и сохраняем диалоги батчами
        try:
            pairs_count = 0
            with open(topic_filepath, topic_filepath_mode) as topic_file:
                for batch_pairs in generate_dialogue(
                    topic=current_topic,
                    llm_client=llm_client,
                    batch_size=batch_size,
                    num_samples=num_samples,
                    temperature=temperature,
                ):
                    # Записываем каждую пару из батча
                    for pair in batch_pairs:
                        topic_file.write(pair.to_jsonl())
                        pairs_count += 1

                    # Принудительная запись в файл после каждого батча
                    topic_file.flush()

            typer.echo(
                typer.style(
                    f"✓ Диалоги для темы '{current_topic}' сохранены в {topic_filepath} "
                    f"({pairs_count} пар)",
                    fg=typer.colors.GREEN,
                )
            )

        except Exception as e:
            typer.echo(
                typer.style(
                    f"✗ Ошибка при обработке темы '{current_topic}': {str(e)}",
                    fg=typer.colors.RED,
                )
            )
            continue

    typer.echo(typer.style("\n✓ Генерация завершена!", fg=typer.colors.GREEN, bold=True))
