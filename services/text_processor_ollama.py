import ollama
import pandas as pd
import json
import os

import typer

# Системный промпт для Ollama
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


def process_text_with_ollama(
        text_to_process: str,
        ollama_client: ollama.Client,
        model_name,
        temperature: float = 0.3
):
    """
    Отправляет текст в Ollama для оценки качества и обработки.
    Возвращает словарь с метрикой качества и обработанным текстом.
    """
    if not text_to_process or not text_to_process.strip():
        return {"quality_score": 0.0, "processed_text": "", "error": "Input text is empty"}

    try:
        response = ollama_client.chat(
            model=model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text_to_process},
            ],
            options={"temperature": temperature}
        )
        content = response['message']['content']

        # Попытка очистить ответ от возможных markdown-блоков JSON
        cleaned_content = content.strip()
        cleaned_content = cleaned_content.split("</think>")[1]
        if cleaned_content.startswith("```json"):
            cleaned_content = cleaned_content[7:]
            if cleaned_content.endswith("```"):
                cleaned_content = cleaned_content[:-3]
        elif cleaned_content.startswith("```"):
            cleaned_content = cleaned_content[3:]
            if cleaned_content.endswith("```"):
                cleaned_content = cleaned_content[:-3]

        cleaned_content = cleaned_content.strip()

        result = json.loads(cleaned_content)
        quality_score = float(result.get("quality_score", 0.0))
        summary = result.get("summary", "")
        processed_text = result.get("processed_text", text_to_process)  # Возвращаем исходный текст при ошибке ключа

        return {"quality_score": quality_score, "processed_text": processed_text, "summary": summary, "error": None}

    except json.JSONDecodeError as e:
        error_message = f"Failed to parse JSON response from Ollama: {e}\nRaw response: '{content}'"
        typer.echo(f"Ошибка декодирования JSON: {error_message}")
        return {"quality_score": 0.0, "processed_text": text_to_process, "error": error_message}
    except Exception as e:
        error_message = f"Error interacting with Ollama: {e}"
        typer.echo(f"Ошибка взаимодействия с Ollama: {error_message}")
        return {"quality_score": 0.0, "processed_text": text_to_process, "error": error_message}


def process_excel_file(
        input_excel_path: str,
        output_excel_path: str,
        text_column_name: str,
        ollama_client: ollama.Client,
        model_name: str = "llama3"
):
    """
    Читает Excel файл, обрабатывает текст из указанной колонки с помощью Ollama
    и сохраняет результаты в новый Excel файл.
    """
    try:
        df = pd.read_excel(input_excel_path)
        typer.echo(f"Файл {input_excel_path} успешно прочитан.")
    except FileNotFoundError:
        typer.echo(f"Ошибка: Входной файл не найден по пути {input_excel_path}")
        return
    except Exception as e:
        typer.echo(f"Ошибка чтения Excel файла: {e}")
        return

    if text_column_name not in df.columns:
        typer.echo(f"Ошибка: Колонка '{text_column_name}' не найдена в Excel файле.")
        typer.echo(f"Доступные колонки: {df.columns.tolist()}")
        return

    results = []
    total_rows = len(df)
    typer.echo(f"Начало обработки {total_rows} строк из файла {input_excel_path}...")

    for index, row in df.iterrows():
        original_text = str(row[text_column_name]) if pd.notna(row[text_column_name]) else ""

        typer.echo(f"\nОбработка строки {index + 1}/{total_rows}...")
        if not original_text.strip():
            typer.echo("Пропуск пустой строки.")
            results.append({
                'Original_Text': original_text,
                'Quality_Score': None,
                'Processed_Text': '',
                'Error': 'Input text was empty or NaN'
            })
            continue

        # print(f"Исходный текст: \"{original_text[:100]}...\"") # Для отладки можно выводить часть текста
        ollama_result = process_text_with_ollama(
            original_text,
            ollama_client,
            model_name=model_name
        )

        results.append({
            'Original_Text': original_text,
            'Quality_Score': ollama_result['quality_score'],
            'Processed_Text': ollama_result['processed_text'],
            'Summary': ollama_result['summary'],
            'Error': ollama_result['error']
        })
        if ollama_result['error']:
            typer.echo(f"Строка {index + 1} обработана с ошибкой: {ollama_result['error']}")
        else:
            typer.echo(f"Строка {index + 1} обработана. Качество: {ollama_result['quality_score']:.2f}")

    results_df = pd.DataFrame(results)
    try:
        # Создаем директорию для выходного файла, если она не существует
        os.makedirs(os.path.dirname(output_excel_path), exist_ok=True)
        results_df.to_excel(output_excel_path, index=False, engine='openpyxl')
        typer.echo(f"\nОбработка завершена. Результаты сохранены в {output_excel_path}")
    except Exception as e:
        typer.echo(f"Ошибка записи результатов в Excel файл: {e}")