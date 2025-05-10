import typer
from elevenlabs import ElevenLabs, Voice

from entrypoint.config import ELEVENLABS_TOKEN


def get_client() -> ElevenLabs:
    if not ELEVENLABS_TOKEN:
        typer.echo("Not found ELEVENLABS_TOKEN")
        raise typer.Exit(1)
    return ElevenLabs(api_key=ELEVENLABS_TOKEN)


def get_voice(client: ElevenLabs, voice_name: str) -> Voice:
    voices_result = client.voices.search(search=voice_name)
    if len(voices_result.voices) == 0:
        raise ValueError(f"Voice '{voice_name}' not found")
    return voices_result.voices[0]
