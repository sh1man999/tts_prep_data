import os

import typer
from typer import Typer
from typing_extensions import Annotated

from entrypoint.config import BASE_DIR
from services.ollama_client import get_ollama_client
from services.text_processor_ollama import process_excel_file

app = Typer(help="Команды для обработки текста.")


@app.command()
def process_xlsx(
    output_patch: Annotated[str, typer.Option(prompt=True, show_default=os.path.join(BASE_DIR, "output"))] = os.path.join(BASE_DIR, "output"),
    ollama_model: Annotated[str, typer.Option(prompt=True, show_default="qwen3:30b-a3b")] = "qwen3:30b-a3b",
    ollama_base_url: Annotated[str, typer.Option(prompt=True, show_default="http://localhost:11434")] = "http://localhost:11434",
    text_column: Annotated[str, typer.Option(prompt=True, show_default="text")] = "text",
):
    base_dir_xlsx_data = os.path.join(BASE_DIR, "xlsx_data")
    os.makedirs(os.path.dirname(output_patch), exist_ok=True)
    ollama_client = get_ollama_client(ollama_base_url, ollama_model)
    xlsx_files_to_process = []
    for item_name in os.listdir(base_dir_xlsx_data):
        # Проверяем файлы с расширением .xlsx (без учета регистра)
        if item_name.lower().endswith(".xlsx"):
            xlsx_files_to_process.append(os.path.join(base_dir_xlsx_data, item_name))

    for input_file_path in xlsx_files_to_process:
        original_filename = os.path.basename(input_file_path)
        base_name, ext = os.path.splitext(original_filename)
        # Формируем имя выходного файла, добавляя суффикс _processed
        output_filename = f"{base_name}_processed{ext}"
        output_file_path = os.path.join(output_patch, output_filename)
        process_excel_file(input_file_path, output_file_path, text_column, ollama_client, ollama_model)
        typer.echo(typer.style(f"  Файл '{original_filename}' успешно обработан.", fg=typer.colors.GREEN))


