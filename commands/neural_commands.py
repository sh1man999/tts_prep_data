import json
import os
import traceback
import uuid
from typing import Optional
import typer
from typer import Typer
from typing_extensions import Annotated

from entrypoint.config import BASE_DIR
from models.device import Device
from models.dialogue_pair import DialoguePair, DialogueResult
from models.llm_provider import LLMProvider
from services.llm_client import create_llm_client
from services.text_generator import generate_multiple_topics
from services.text_preprocessing import process_jsonl_file, generate_text_hash
from utils import get_available_gpus

app = Typer(help="Команды для обработки текста и генерации.")


@app.command()
def preprocess_file(
    jsonl_file_name: Annotated[
        str, typer.Option(help="Имя JSONL файла для обработки")
    ] = "Здоровое_питание.jsonl",
    output_file_name: Annotated[
        Optional[str],
        typer.Option(
            help="Имя выходного файла (если не указано, генерируется автоматически)"
        ),
    ] = None,
    batch_size: Annotated[int, typer.Option(prompt=True, show_default=True, help="Кол-во примеров в одном запросе")] = 50,
    provider: Annotated[
        LLMProvider, typer.Option(help="LLM провайдер")
    ] = LLMProvider.OLLAMA,
    model_name: Annotated[str, typer.Option(help="Название модели")] = "qwen3:30b-a3b",
    input_dir: Annotated[
        str, typer.Option(help="Директория с входными файлами")
    ] = "./datasets",
    output_dir: Annotated[
        str, typer.Option(help="Директория для сохранения результатов")
    ] = "./datasets/processed",
):
    """
    Обрабатывает JSONL файл с помощью выбранного LLM провайдера
    """
    # Формируем пути
    jsonl_file_path = os.path.join(input_dir, jsonl_file_name)

    # Проверяем существование входного файла
    if not os.path.exists(jsonl_file_path):
        typer.echo(
            typer.style(f"Файл не найден: {jsonl_file_path}", fg=typer.colors.RED)
        )
        raise typer.Exit(code=1)

    # Генерируем имя выходного файла если не указано
    if not output_file_name:
        base_name = os.path.splitext(jsonl_file_name)[0]
        output_file_name = f"{base_name}_processed_{uuid.uuid4().hex[:8]}.jsonl"

    output_file_path = os.path.join(output_dir, output_file_name)

    # Создаем директорию для выходного файла
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

    # Создаем LLM клиент
    client_kwargs = {"model_name": model_name}

    try:
        llm_client = create_llm_client(provider, **client_kwargs)
    except Exception as e:
        typer.echo(
            typer.style(f"Ошибка создания клиента: {str(e)}", fg=typer.colors.RED)
        )
        raise typer.Exit(code=1)

    typer.echo(f"\nВыходной файл: {output_file_path}")

    # Обрабатываем файл
    try:
        with open(output_file_path, "w", encoding="utf-8") as output_file:
            for batch in process_jsonl_file(jsonl_file_path, llm_client, batch_size):
                for item in batch:
                    id1 = generate_text_hash(item.user_query)
                    text1 = item.user_query
                    id2 = generate_text_hash(item.ai_response)
                    text2 = item.ai_response
                    output_file.write(DialogueResult(id=id1, text=text1).to_jsonl())
                    output_file.write(DialogueResult(id=id2, text=text2).to_jsonl())

        typer.echo(
            typer.style(
                f"\n✅ Обработка завершена!",
                fg=typer.colors.GREEN,
                bold=True,
            )
        )
        typer.echo(f"Результаты сохранены в: {output_file_path}")

    except Exception as e:
        typer.echo(
            typer.style(f"\n❌ Ошибка при обработке: {str(e)}", fg=typer.colors.RED)
        )
        raise typer.Exit(code=1)


@app.command()
def generate_text(
    topic: Annotated[
        Optional[str],
        typer.Option(
            help="Тема для генерации диалога. Используйте это или --topics-file.",
        ),
    ] = None,
    topics_file: Annotated[
        Optional[str],
        typer.Option(
            help="Файл со списком тем (по одной в строке). Используйте это или --topic.",
        ),
    ] = None,
    samples: Annotated[int, typer.Option(prompt=True, min=1, max=1000, show_default=True, help="Количество пар запрос-ответ для каждой темы.")] = 100,
    batch_size: Annotated[int, typer.Option(prompt=True, show_default=True, help="Кол-во примеров в одном запросе")] = 200,
    provider: Annotated[
        LLMProvider, typer.Option(show_default=True, help="LLM провайдер")
    ] = LLMProvider.OLLAMA,
    model_name: Annotated[Optional[str], typer.Option(help="Название модели")] = None,
    base_url: Annotated[
        str, typer.Option(help="Base URL (для Ollama)")
    ] = "http://localhost:11434",
    temperature: Annotated[float, typer.Option(prompt=True, min=0.0, max=1.0, help="Температура генерации (0.0-1.0).", show_default=True)] = 0.7,
):
    typer.echo(typer.style("Параметры генерации:", bold=True))
    typer.echo(f"  Провайдер: {provider.value}")
    typer.echo(f"  Модель: {model_name}")
    typer.echo(f"  Температура: {temperature}")
    typer.echo(f"  Количество примеров: {samples}")

    output_dir = os.path.join(BASE_DIR,"datasets")
    # Создаем директорию если не существует
    os.makedirs(output_dir, exist_ok=True)


    # Создаем LLM клиент
    client_kwargs = {}

    if model_name and model_name != '':
        client_kwargs["model_name"] = model_name
    if provider == LLMProvider.OLLAMA:
        client_kwargs["base_url"] = base_url

    try:
        llm_client = create_llm_client(provider, **client_kwargs)
    except Exception as e:
        typer.echo(
            typer.style(f"Ошибка создания клиента: {str(e)}", fg=typer.colors.RED)
        )
        raise typer.Exit(code=1)

    # Определяем темы для обработки
    topics_to_process = []
    if topic:
        topics_to_process = [topic]
        typer.echo(f"  Используется одна тема из аргумента: '{topic}'")
    elif topics_file:
        if os.path.exists(topics_file):
            with open(topics_file, "r", encoding="utf-8") as f:
                topics_to_process = [line.strip() for line in f if line.strip()]
            if topics_to_process:
                typer.echo(
                    f"  Используются темы из файла: '{topics_file}' ({len(topics_to_process)} тем)"
                )
            else:
                typer.echo(
                    typer.style(
                        f"Файл тем '{topics_file}' пуст.", fg=typer.colors.YELLOW
                    )
                )
        else:
            typer.echo(
                typer.style(
                    f"Файл тем '{topics_file}' не найден.", fg=typer.colors.RED
                )
            )

    if not topics_to_process:
        typer.echo(
            typer.style(
                "Темы для генерации не указаны. Пожалуйста, укажите тему через --topic или файл тем через --topics-file.",
                fg=typer.colors.RED,
            )
        )
        raise typer.Exit(code=1)

    # Генерируем диалоги
    try:
        generate_multiple_topics(
            topics_to_process,
            output_dir,
            llm_client,
            batch_size,
            num_samples=samples,
            temperature=temperature,
        )
        typer.echo(
            typer.style(
                "\n✅ Генерация успешно завершена!", fg=typer.colors.GREEN, bold=True
            )
        )
    except Exception as e:
        traceback.print_exc()
        typer.echo(
            typer.style(f"\n❌ Ошибка при генерации: {str(e)}", fg=typer.colors.RED)
        )
        raise typer.Exit(code=1)


@app.command()
def runorm_file(
        jsonl_file_name: Annotated[str, typer.Option(prompt=True, show_default=True)] = "Здоровое_питание.jsonl",
        device: Annotated[Device, typer.Option(prompt=True, show_default=True)] = Device.cuda
):
    from runorm import RUNorm
    typer.echo(get_available_gpus())
    normalizer = RUNorm()
    normalizer.load(model_size="big", device=device)
    jsonl_file_path = os.path.join(BASE_DIR, "payload_datasets", jsonl_file_name)
    dialogue_pairs = []
    with open(jsonl_file_path, "r", encoding='utf-8') as jsonl_file:
        for line in jsonl_file:
            pair = DialoguePair.model_validate(json.loads(line))
            pair.ai_response = normalizer.norm(pair.ai_response)
            dialogue_pairs.append(pair)
    with open(jsonl_file_path, "w", encoding='utf-8') as jsonl_file:
        lines = [pair.to_jsonl() for pair in dialogue_pairs]
        jsonl_file.writelines(lines)
