import os

import typer
from huggingface_hub import HfApi
from typer import Typer
from typing_extensions import Annotated

from entrypoint.config import BASE_DIR, HF_TOKEN

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
    api.upload_folder(
        folder_path=input_path,
        repo_id=hub_repository_id,
        repo_type="dataset"
    )
