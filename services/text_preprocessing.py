import json
import os
from typing import Generator

import ollama
import pandas as pd
import typer
from pydantic import BaseModel

from models.base_llm_client import BaseLLMClient
from models.preprocessed_text import NeuralProcessedText

# Системный промпт
SYSTEM_PROMPT = """
You are an expert linguistic assistant specializing in preparing **Russian text** for Text-to-Speech (TTS) synthesis. Your task is to analyze input Russian text, assess its quality, process it for optimal TTS output, and provide an explanation for your quality assessment including any issues found in the original text.

Please perform the following steps:

1.  **Assess the input Russian text** for overall quality and suitability for TTS. This assessment should meticulously consider:
    * **Grammar and spelling:** Accuracy of language constructs and word forms.
    * **Clarity and coherence:** How easy the text is to understand and if it flows logically.
    * **Punctuation:** Correctness and effectiveness of punctuation for readability and TTS pausing.
    * **Need for normalization:** Presence of abbreviations, acronyms, numbers, special characters, non-standard expressions that require conversion to a pronounceable, explicit form.
    * **Lexical ambiguity:** Presence of homographs or other words that might be pronounced incorrectly by a TTS system without sufficient context or clarification.
    Assign a `quality_score` as a float between 0.0 (very low quality, many issues) and 1.0 (very high quality, few to no issues beyond potential minor TTS normalization). This score must reflect the state of the *original* input text.

2.  **Perform comprehensive text processing for Russian TTS to create the `processed_text`:**
    * **Correct all spelling and grammatical errors.**
    * **Normalize the text:**
        * Expand all abbreviations (e.g., "г.", "ул.", "др.") and acronyms (e.g., "СССР", "РФ") into their full, pronounceable Russian word forms as they would be spoken (e.g., "г." to "город" or "года" depending on context, "СССР" to "Союз Советских Социалистических Республик" or "эс-эс-эс-эр" if that's the intended spoken form).
        * Convert all numbers (cardinal, ordinal), full dates (day, month, year), and monetary values into their full Russian word equivalents, inflected correctly according to the grammatical context (e.g., "10 человек" to "десять человек", "25-го января 2023 г." to "двадцать пятого января две тысячи двадцать третьего года", "100$" to "сто долларов", "в 2002 г." to "в две тысячи втором году").
        * Replace special characters (e.g., %, @, #, &, +, -, *, /, <, >, ~ etc.) with their corresponding Russian word equivalents (e.g., "%" to "процентов", "@" to "собака", "+" to "плюс", ">100" to "больше ста", "~2кг" to "примерно два килограмма") or handle them contextually for correct spoken representation.
    * **Ensure correct and unambiguous punctuation** (commas, periods, question marks, exclamation points, colons, semicolons, dashes) to guide natural pauses and intonation in synthesized speech. This may involve adding, removing, or correcting punctuation.
    * **Address homographs and potential pronunciation ambiguities:** Clarify Russian words that have the same spelling but different pronunciations/meanings to ensure correct interpretation by the TTS system (e.g., differentiate between 'за́мок' - castle and 'замо́к' - lock). If the input text lacks sufficient context for the TTS to likely disambiguate correctly, you may need to subtly rephrase the ambiguous part or add minimal context, while strictly preserving the original core meaning. **Do not use explicit stress marks in the output `processed_text`.**
    * **Ensure overall clarity and unambiguity of phrasing** for spoken interpretation. Word order and grammatical constructions should be natural for spoken Russian.

3.  **Adaptation of `processed_text` based on `quality_score`:**
    * If the `quality_score` (determined in step 1 from the *original* text) is **less than 0.6**, you MUST **rewrite and adapt the text extensively** while performing the operations in step 2. This includes improving the text's overall structure, flow, clarity, and readability to make it highly suitable for TTS, beyond just fixing discrete errors. The original core meaning and intent must be strictly preserved.
    * If the `quality_score` is **0.6 or higher**, the `processed_text` should be the result of applying all transformations outlined in step 2. Substantial rewriting of the original content or style should be avoided; the focus is on meticulously preparing the existing text for optimal TTS output by correcting errors and normalizing elements.
    The result of this step is the `processed_text`.

4.  **Generate the `summary`**: This summary must be in Russian and explain the `quality_score` assigned in Step 1. It should briefly state the overall assessed quality of the *original* input text and then list the specific flaws ("косяки"), errors, or areas that required improvement or attention during processing. For example, mention issues like: "орфографические ошибки", "грамматические неточности", "необходимость нормализации чисел и сокращений", "неясные формулировки, требующие адаптации", "проблемы с пунктуацией", "наличие омографов без достаточного контекста, учтено при обработке".

Your response MUST be a single JSON object. This JSON object must contain exactly three keys:
-   `"quality_score"`: A float representing the assessed quality of the *original* input Russian text, as determined in Step 1.
-   `"processed_text"`: A string containing the fully processed (corrected, normalized, and potentially adapted) Russian text from Step 3, ready for TTS.
-   `"summary"`: The Russian string generated in Step 4, explaining the `quality_score` and detailing flaws in the *original* input.

Do not include any explanations, apologies, or conversational text outside of this JSON structure.

Example of a high-quality input (Russian):
User input text: "В 1998 г. проект стоил >100 тыс. руб. Мы читали про замок."
Your JSON output:
{
  "quality_score": 0.85,
  "processed_text": "В тысяча девятьсот девяносто восьмом году проект стоил больше ста тысяч рублей. Мы читали про замок.",
  "summary": "Оценка 0.85: Исходный текст хорошего качества, но требует стандартной нормализации для TTS. Основные аспекты, учтенные при обработке: необходимость расшифровки сокращений ('г.', 'тыс. руб.'), преобразования чисел и дат в словесную форму ('1998', '100'), обработки специального символа ('>'). Потенциальная омография слова 'замок' учтена, предполагая наиболее вероятное значение по общему контексту."
}

Example of a low-quality input needing adaptation (Russian):
User input text: "Маша пашла в магаз купит хлеб картшка и мн др за 50 р и еще ~2кг яблок"
Your JSON output:
{
  "quality_score": 0.2,
  "processed_text": "Маша пошла в магазин купить хлеб, картошку и многое другое за пятьдесят рублей, и ещё примерно два килограмма яблок.",
  "summary": "Оценка 0.2: Исходный текст очень низкого качества. Обнаружены и исправлены многочисленные 'косяки': множественные орфографические ошибки ('пашла', 'магаз', 'картшка'); грамматическая ошибка в глаголе ('купит'); большое количество ненормализованных сокращений и неформальных выражений ('мн др', 'р.', 'магаз'); числа ('50', '2') и единицы измерения ('кг') требовали преобразования в слова; специальный символ ('~') требовал интерпретации; пунктуация отсутствовала и была добавлена для корректного синтаксического деления и интонации."
}

Now, process the text I will provide.
/no_think
"""


class TextProcessedLLMResult(BaseModel):
    quality_score: float
    processed_text: str
    summary: str


class NeuralProcessedText(BaseModel):
    quality_score: float
    processed_text: str
    summary: str
    original_text: str

    def to_jsonl(self) -> str:
        """Конвертирует объект в JSONL строку"""
        return json.dumps(self.dict(), ensure_ascii=False) + "\n"


def process_text_with_llm(
    text_to_process: str, llm_client: BaseLLMClient, temperature: float = 0.3
) -> NeuralProcessedText:
    """
    Обрабатывает текст с помощью LLM

    Args:
        text_to_process: Текст для обработки
        llm_client: Клиент LLM для обработки
        temperature: Температура генерации

    Returns:
        NeuralProcessedText с результатами обработки
    """
    text_to_process = text_to_process.strip()
    if not text_to_process:
        raise ValueError("No text provided.")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": text_to_process},
    ]

    try:
        response = llm_client.chat(
            messages=messages,
            temperature=temperature,
            response_format=TextProcessedLLMResult.model_json_schema(),
        )

        result = json.loads(response)
        typer.echo(f"Оценка качества: {result.get('quality_score', 0.0)}")

        quality_score = float(result.get("quality_score", 0.0))
        summary = result.get("summary", "")
        processed_text = result.get("processed_text", text_to_process)

        return NeuralProcessedText(
            quality_score=quality_score,
            processed_text=processed_text,
            summary=summary,
            original_text=text_to_process,
        )
    except Exception as e:
        typer.echo(f"Ошибка обработки: {str(e)}", err=True)
        # В случае ошибки возвращаем исходный текст
        return NeuralProcessedText(
            quality_score=0.0,
            processed_text=text_to_process,
            summary="Ошибка обработки",
            original_text=text_to_process,
        )


def process_jsonl_file(
    jsonl_file_path: str,
    text_column_name: str,
    llm_client: BaseLLMClient,
) -> Generator[NeuralProcessedText, None, None]:
    """
    Обрабатывает JSONL файл построчно

    Args:
        jsonl_file_path: Путь к JSONL файлу
        text_column_name: Название колонки с текстом
        llm_client: Клиент LLM для обработки

    Yields:
        NeuralProcessedText для каждой обработанной строки
    """
    if not os.path.exists(jsonl_file_path):
        raise FileNotFoundError(f"Файл не найден: {jsonl_file_path}")

    with open(jsonl_file_path, "r", encoding="utf-8") as jsonl_file:
        lines = jsonl_file.readlines()
        total_rows = len(lines)
        typer.echo(f"Начало обработки {total_rows} строк из файла {jsonl_file_path}...")

        processed_count = 0
        skipped_count = 0

        for index, row in enumerate(lines):
            try:
                row_data = json.loads(row)

                # Проверяем наличие нужной колонки
                if text_column_name not in row_data:
                    typer.echo(
                        f"Предупреждение: колонка '{text_column_name}' не найдена в строке {index + 1}",
                        err=True,
                    )
                    skipped_count += 1
                    continue

                original_text = (
                    str(row_data[text_column_name])
                    if row_data[text_column_name]
                    else ""
                )

                typer.echo(f"\nОбработка строки {index + 1}/{total_rows}...")

                if not original_text.strip():
                    typer.echo("Пропуск пустой строки")
                    skipped_count += 1
                    continue

                result = process_text_with_llm(original_text, llm_client)

                processed_count += 1
                yield result

            except json.JSONDecodeError as e:
                typer.echo(
                    f"Ошибка парсинга JSON в строке {index + 1}: {str(e)}", err=True
                )
                skipped_count += 1
                continue
            except Exception as e:
                typer.echo(f"Ошибка обработки строки {index + 1}: {str(e)}", err=True)
                skipped_count += 1
                continue

        typer.echo(f"\nОбработка завершена. Обработано: {processed_count}, пропущено: {skipped_count}")

