import json
import os
import wave
from typing import Iterator

import typer
from elevenlabs import save
from elevenlabs import VoiceSettings
from typer import Typer
from typing_extensions import Annotated

from entrypoint.config import BASE_DIR
from exceptions import Limit
from models.row import BaseRow, HfRow
from models.voice import ElevenlabsVoice
from services.elevenlabs_service import get_client, get_voice

app = Typer(help="Команды для обработки текста.")


@app.command()
def jsonl_to_audio(
        input_file_name: Annotated[str, typer.Option(prompt=True, show_default=True)] = "den4ikai.jsonl",
        output_path: Annotated[
            str, typer.Option(prompt=True, show_default=True)] = os.path.join(
            BASE_DIR, "output_elevenlabs"),
        voice_name: Annotated[ElevenlabsVoice, typer.Option(prompt=True, show_default=True,
                                                            case_sensitive=False)] = ElevenlabsVoice.nuri,
        limit: Annotated[int, typer.Option(prompt=True, show_default=True)] = 5,
        audio_format: Annotated[str, typer.Option(prompt=True, show_default=True)] = ".wav",  # .wav .mp3
):
    input_file_path = os.path.join(BASE_DIR, "payload_datasets", input_file_name)
    source = input_file_name.replace(".jsonl", "")
    client = get_client()
    voice = get_voice(client, voice_name.value)
    typer.echo(f"voice '{voice.voice_id}' found")
    output_path = os.path.join(output_path, source)
    os.makedirs(output_path, exist_ok=True)
    output_metadata_file_path = os.path.join(output_path, "metadata.jsonl")

    # Временный файл, который станет новым input_file_path.
    # Он будет содержать все строки, которые не были успешно обработаны.
    temp_input_for_next_run_path = input_file_path + ".processing_temp"

    with open(input_file_path, "r", newline='', encoding="utf-8") as f:
        rows_to_process_from_input = list(f)

    output_file_mode = 'w' if os.path.exists(output_metadata_file_path) == 0 else 'a'

    try:
        with open(temp_input_for_next_run_path, "w", newline='', encoding="utf-8") as temp_input_file, \
                open(output_metadata_file_path, output_file_mode, newline='', encoding='utf-8') as output_file:

            if not rows_to_process_from_input:
                typer.echo("Исходный файл был пуст")

            for i, row in enumerate(rows_to_process_from_input):
                try:
                    base_row = BaseRow(**json.loads(row))

                    audio_dir_path = os.path.join(output_path, "audio")
                    os.makedirs(audio_dir_path, exist_ok=True)
                    audio_name = f"{source}_{base_row.id}{audio_format}"
                    relative_audio_path = os.path.join("audio", audio_name)
                    full_audio_path = os.path.join(audio_dir_path, audio_name)
                    audio_bytes = client.generate(
                        text=base_row.text,
                        voice=voice,
                        model="eleven_multilingual_v2",
                        output_format="pcm_48000" if audio_format == ".wav" else "mp3_44100_192",
                        voice_settings=VoiceSettings(
                            # Определяет, насколько стабилен голос и насколько случайным является каждое его поколение. Более низкие значения расширяют эмоциональный диапазон голоса. Более высокие значения могут привести к монотонному голосу с ограниченными эмоциями.
                            stability=0.8,
                            # Определяет, насколько точно ИИ должен придерживаться оригинального голоса при попытке его воспроизведения.
                            similarity_boost=1.0,
                            speed=1,
                            use_speaker_boost=True,
                            # Определяет преувеличение стиля голоса. Эта настройка пытается усилить стиль оригинального диктора. Она потребляет дополнительные вычислительные ресурсы и может увеличить задержку, если установить значение, отличное от 0.
                            style=0
                        )
                    )
                    if isinstance(audio_bytes, Iterator):
                        audio_bytes = b"".join(audio_bytes)
                    if audio_format == ".wav":
                        with wave.open(full_audio_path, 'wb') as wavfile:
                            # Установка параметров WAV файла:
                            # (nchannels, sampwidth, framerate, nframes, comptype, compname)
                            # nchannels: 1 для моно, 2 для стерео
                            # sampwidth: ширина сэмпла в байтах (1 для 8-бит, 2 для 16-бит, 3 для 24-бит)
                            # framerate: частота дискретизации (например, 48000)
                            # nframes: количество кадров (0 если неизвестно заранее, будет обновлено при закрытии)
                            # comptype: тип сжатия ('NONE' для PCM)
                            # compname: описание сжатия ('not compressed' для PCM)

                            nchannels = 1  # ПРЕДПОЛОЖЕНИЕ: Моно
                            sampwidth = 2  # ПРЕДПОЛОЖЕНИЕ: 16 бит (2 байта на сэмпл)
                            framerate = 48000  # Соответствует вашему output_format
                            nframes = 0  # Будет вычислено автоматически при записи всех кадров
                            comptype = 'NONE'
                            compname = 'NONE'

                            wavfile.setparams((nchannels, sampwidth, framerate, nframes, comptype, compname))
                            wavfile.writeframes(audio_bytes)
                    else:
                        save(audio_bytes, full_audio_path)
                    hf_row = HfRow(**base_row.model_dump(), file_name=relative_audio_path, source=source,
                                   style="default", voice=voice.name)
                    output_file.write(hf_row.to_jsonl())
                    if i == limit:
                        raise Limit("limit reached")
                except Limit as e:
                    for k in range(i + 1, len(rows_to_process_from_input)):
                        temp_input_file.write(rows_to_process_from_input[k])

                    # Передаем ошибку дальше, чтобы прервать выполнение и зафиксировать состояние
                    raise e

                except Exception as e_row_processing:
                    # Записываем эту "проблемную" строку в temp_input_for_next_run_path
                    temp_input_file.write(base_row.to_jsonl())

                    # Также записываем все ОСТАВШИЕСЯ строки из первоначального списка
                    # в temp_input_for_next_run_path, так как они еще не были обработаны.
                    for k in range(i + 1, len(rows_to_process_from_input)):
                        temp_input_file.write(rows_to_process_from_input[k])

                    # Передаем ошибку дальше, чтобы прервать выполнение и зафиксировать состояние
                    raise e_row_processing
    except Exception as e_fatal:
        typer.echo(e_fatal, err=True)
        raise typer.Exit(code=1) from e_fatal
    finally:
        os.replace(temp_input_for_next_run_path, input_file_path)
