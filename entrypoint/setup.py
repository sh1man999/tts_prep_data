from commands import text_processor_commands, text_generator_commands, text_to_elevenlabs_audio_cammands
from typer import Typer


def create_app() -> Typer:
    app = Typer()
    return app


def configure_app(app: Typer) -> None:
    app.add_typer(text_processor_commands.app, name="text_processor")
    app.add_typer(text_generator_commands.app, name="text_generator")
    app.add_typer(text_to_elevenlabs_audio_cammands.app, name="text_to_audio")
