import ollama
import typer
from ollama import ListResponse

def get_ollama_client(ollama_base_url: str, ollama_model: str):
    client = ollama.Client(host=ollama_base_url)
    model_data = client.list()  # type: ListResponse
    available_models = []
    for item in model_data.models:
        available_models.append(item.model)
    if not available_models:
        typer.echo("Доступные модели: Нет моделей или не удалось прочитать список.")
    else:
        typer.echo(f"Доступные модели: {', '.join(available_models)}")

    if ollama_model not in available_models:
        typer.echo(
            f"ВНИМАНИЕ: Модель '{ollama_model}' не найдена среди доступных моделей Ollama на {ollama_base_url}.")
        typer.echo(
            f"Пожалуйста, убедитесь, что модель загружена (например, 'ollama pull {ollama_model}') или выберите другую модель.")
        if not available_models:
            typer.echo("Нет доступных моделей для выбора. Загрузите модель через 'ollama pull <model_name>'.")
            exit(1)
    return client
