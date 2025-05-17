import os

import typer
from huggingface_hub import HfApi
from typer import Typer
from typing_extensions import Annotated

from entrypoint.config import BASE_DIR, HF_TOKEN
from utils import get_wav_duration, format_duration

app = Typer(help="Команды для загрузки аудио данных на hf.")


@app.command()
def upload_folder(
        input_path: Annotated[
            str, typer.Option(prompt=True, show_default=True)] = os.path.join(
            BASE_DIR, "output_elevenlabs"),
        hub_repository_id: Annotated[
            str, typer.Option(prompt=True, show_default=True)] = "Sh1man/elevenlabs",
):
    api = HfApi(token=HF_TOKEN)
    api.upload_large_folder(
        folder_path=input_path,
        repo_id=hub_repository_id,
        repo_type="dataset"
    )

@app.command()
def calculate_dataset_duration(
    input_path: Annotated[
        str, typer.Option(exists=True, file_okay=False, dir_okay=True, readable=True, resolve_path=True, help="Путь к папке с WAV файлами.")]
):
    """
    Рассчитывает и отображает общую продолжительность всех WAV файлов в указанной папке.
    """
    if not os.path.isdir(input_path):
        typer.echo(f"Ошибка: Указанный путь '{input_path}' не является директорией или не существует.")
        raise typer.Exit(code=1)

    total_duration_seconds = 0.0
    wav_files_count = 0

    typer.echo(f"Сканирование директории: {input_path}")

    for root, _, files in os.walk(input_path):
        for filename in files:
            if filename.lower().endswith(".wav"):
                file_path = os.path.join(root, filename)
                duration = get_wav_duration(file_path)
                if duration > 0:
                    total_duration_seconds += duration
                    wav_files_count += 1


    if wav_files_count == 0:
        typer.echo("WAV файлы не найдены в указанной директории.")
        raise typer.Exit()

    formatted_total_duration = format_duration(total_duration_seconds)

    print(f"\n--- Отчет о продолжительности датасета ---")
    print(f"Проанализировано WAV файлов: {wav_files_count}")
    print(f"Общая продолжительность: {formatted_total_duration}")
