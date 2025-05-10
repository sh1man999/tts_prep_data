from commands import neural_commands, elevenlabs_commands, hf_commands
from typer import Typer


def create_app() -> Typer:
    app = Typer()
    return app


def configure_app(app: Typer) -> None:
    app.add_typer(neural_commands.app, name="neural")
    app.add_typer(elevenlabs_commands.app, name="elevenlabs")
    app.add_typer(hf_commands.app, name="hf")
