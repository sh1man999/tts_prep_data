import os
from typing import Optional, List # Убраны Dict, Any, если они не нужны для других частей
                                # (оставлены на случай, если BASE_DIR или process_excel_file их используют неявно)

import typer
from typer import Typer
from typing_extensions import Annotated

from entrypoint.config import BASE_DIR # Предполагается, что BASE_DIR может быть нужен
from services.ollama_client import get_ollama_client
# Убран импорт DialogueGeneration, DialoguePair, save_dialogue_as_json
from services.text_generator_ollama import generate_multiple_topics
from services.text_processor_ollama import process_excel_file # Оставлен, т.к. был импортирован


app = Typer(help="Генератор диалогов для TTS с использованием Ollama.")


@app.command()
def generate(
        topic_arg: Annotated[Optional[str], typer.Option("--topic", help="Тема для генерации диалога. Используйте это или --topics-file.")] = None,
        topics_file_arg: Annotated[Optional[str], typer.Option("--topics-file", help="Файл со списком тем (по одной в строке). Используйте это или --topic.")] = None,
        samples: Annotated[int, typer.Option(prompt=True, help="Количество пар запрос-ответ для каждой темы.", show_default=True)] = 5,
        ollama_model: Annotated[str, typer.Option(prompt=True, show_default=True)] = "qwen3:30b-a3b",
        ollama_base_url: Annotated[str, typer.Option(prompt=True, show_default=True)] = "http://localhost:11434",
        output_excel: Annotated[str, typer.Option(prompt=True, help="Путь к Excel файлу для сохранения общих результатов.", show_default=True)] = "output/generated_dialogues.xlsx",
        temperature: Annotated[float, typer.Option(prompt=True, min=0.0, max=1.0, help="Температура генерации (0.0-1.0).", show_default=True)] = 0.7,
):
    typer.echo(typer.style(f"Параметры генерации:", bold=True))

    ollama_client = get_ollama_client(ollama_base_url, ollama_model)
    topics_to_process = []
    if topic_arg:
        topics_to_process = [topic_arg]
        typer.echo(f"  Используется одна тема из аргумента: '{topic_arg}'")
    elif topics_file_arg:
        if os.path.exists(topics_file_arg):
            with open(topics_file_arg, 'r', encoding='utf-8') as f:
                topics_to_process = [line.strip() for line in f if line.strip()]
            if not topics_to_process:
                typer.echo(typer.style(f"Файл тем '{topics_file_arg}' пуст. Используются дефолтные темы.",
                                       fg=typer.colors.YELLOW)) # Это сообщение может потребовать корректировки, если дефолтные темы не предусмотрены
            else:
                typer.echo(f"  Используются темы из файла: '{topics_file_arg}' ({len(topics_to_process)} тем)")
        else:
            typer.echo(typer.style(f"Файл тем '{topics_file_arg}' не найден.", # Убрано "Используются дефолтные темы."
                                   fg=typer.colors.RED))

    if not topics_to_process:
        typer.echo(typer.style("Темы для генерации не указаны. Пожалуйста, укажите тему через --topic или файл тем через --topics-file.", fg=typer.colors.RED))
        raise typer.Exit(code=1)

    generate_multiple_topics(
        topics_to_process,
        output_excel,
        ollama_client,
        num_samples=samples,
        model_name=ollama_model,
        temperature=temperature
    )
