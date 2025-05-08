from commands import text_processor_commands, text_generator_commands
from typer import Typer


def create_app() -> Typer:
    app = Typer()
    return app


def configure_app(app: Typer) -> None:
    app.add_typer(text_processor_commands.app, name="text_processor")
    app.add_typer(text_generator_commands.app, name="text_generator" )
