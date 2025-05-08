import os
from typing import Optional, Dict, List, Any

import typer
from typer import Typer
from typing_extensions import Annotated

from entrypoint.config import BASE_DIR
from services.ollama_client import get_ollama_client
from services.text_generator_ollama import generate_multiple_topics, DialogueGeneration, DialoguePair, \
    save_dialogue_as_json
from services.text_processor_ollama import process_excel_file


app = Typer(help="Генератор диалогов для TTS с использованием Ollama.")


@app.command()
def generate(
        topic_arg: Annotated[Optional[str], typer.Option("--topic", help="Тема для генерации диалога. Используйте это или --topics-file.")] = None,
        topics_file_arg: Annotated[Optional[str], typer.Option("--topics-file", help="Файл со списком тем (по одной в строке). Используйте это или --topic.")] = None,
        samples: Annotated[int, typer.Option(prompt=True, help="Количество пар запрос-ответ для каждой темы.", show_default=True)] = 5,
        ollama_model: Annotated[str, typer.Option(prompt=True, show_default="qwen3:30b-a3b")] = "qwen3:30b-a3b",
        ollama_base_url: Annotated[str, typer.Option(prompt=True, show_default="http://localhost:11434")] = "http://localhost:11434",
        output_excel: Annotated[str, typer.Option(prompt=True, help="Путь к Excel файлу для сохранения общих результатов.", show_default=True)] = "output/generated_dialogues.xlsx",
        output_json_dir: Annotated[str, typer.Option(prompt=True, help="Директория для JSON файлов (отдельный файл для каждой темы).", show_default=True)] = "output/json/",
        temperature: Annotated[float, typer.Option(prompt=True, min=0.0, max=1.0, help="Температура генерации (0.0-1.0).", show_default=True)] = 0.7,
):
    """
    Генерирует диалоги пользователь-ИИ на заданные темы с помощью Ollama.
    """
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
                                       fg=typer.colors.YELLOW))
            else:
                typer.echo(f"  Используются темы из файла: '{topics_file_arg}' ({len(topics_to_process)} тем)")
        else:
            typer.echo(typer.style(f"Файл тем '{topics_file_arg}' не найден. Используются дефолтные темы.",
                                   fg=typer.colors.RED))

    if not topics_to_process:  # Если ни тема, ни файл не были успешно загружены
        raise Exception("Темы не указаны !")


    # Генерируем диалоги для всех тем и сохраняем общий результат в Excel
    all_generated_pairs = generate_multiple_topics(
        topics_to_process,
        output_excel,
        ollama_client,
        num_samples=samples,
        model_name=ollama_model,
        temperature=temperature
    )

    # Дополнительно: генерируем и сохраняем каждую тему в отдельный JSON файл
    # Используем данные из all_generated_pairs, чтобы не генерировать заново,
    # а сгруппировать их по темам для сохранения.
    if output_json_dir:  # Проверяем, что директория указана
        os.makedirs(output_json_dir, exist_ok=True)  # Создаем директорию, если ее нет

        # Группируем сгенерированные пары по темам
        topic_to_pairs_map: Dict[str, List[Dict[str, Any]]] = {}
        for pair_data in all_generated_pairs:
            current_topic = pair_data['Topic']
            if current_topic not in topic_to_pairs_map:
                topic_to_pairs_map[current_topic] = []
            topic_to_pairs_map[current_topic].append({
                "id": pair_data['Pair_ID'],
                "user_query": pair_data['User_Query'],
                "ai_response": pair_data['AI_Response']
            })

        for current_topic, topic_pairs in topic_to_pairs_map.items():
            if not current_topic.strip():
                continue

            dialogue_gen_object = DialogueGeneration(
                topic=current_topic,
                pairs=[DialoguePair(**p) for p in topic_pairs]
            )

            safe_filename = "".join(c if c.isalnum() or c in [' ', '_'] else '_' for c in current_topic)
            safe_filename = safe_filename.replace(' ', '_').lower()
            if not safe_filename:  # Если имя файла получилось пустым
                safe_filename = f"topic_{sum(1 for t_data in all_generated_pairs if t_data['Topic'] == current_topic)}"  # просто уникальное имя
            json_path = os.path.join(output_json_dir, f"{safe_filename}.json")

            save_dialogue_as_json(dialogue_gen_object, json_path)
    else:
        typer.echo(typer.style("\nДиректория для JSON не указана, отдельные JSON файлы не будут сохранены.",
                               fg=typer.colors.YELLOW))



