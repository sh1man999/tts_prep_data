import csv
import os
import wave
from typing import Iterator

from elevenlabs import save, VoiceSettings
import typer
from elevenlabs import ElevenLabs
from typer import Typer
from typing_extensions import Annotated

from entrypoint.config import BASE_DIR, ELEVENLABS_TOKEN
from exceptions import Limit

app = Typer(help="Команды для обработки текста.")


@app.command()
def tsv_to_elevenlabs_audio(
        input_file_path: Annotated[str, typer.Option(prompt=True, show_default=True)] = os.path.join(
            BASE_DIR, "output", "denchik_dt.tsv"),
        output_path: Annotated[
            str, typer.Option(prompt=True, show_default=True)] = os.path.join(
            BASE_DIR, "output_elevenlabs"),
        voice: Annotated[str, typer.Option(prompt=True, show_default=True)] = "Arcades", # Matilda | Dorothy | Arcades | Soft Female Russian voice
        limit: Annotated[int, typer.Option(prompt=True, show_default=True)] = 5,
        audio_format: Annotated[str, typer.Option(prompt=True, show_default=True)] = ".wav" # .wav .mp3
):
    if not ELEVENLABS_TOKEN:
        typer.echo("Not found ELEVENLABS_TOKEN")
        raise typer.Exit(1)
    client = ElevenLabs(api_key=ELEVENLABS_TOKEN)
    voices_result = client.voices.search(search=voice)

    if len(voices_result.voices) == 0:
        typer.echo(f"voice '{voice}' not found")
        raise typer.Exit(1)
    voice = voices_result.voices[0]
    typer.echo(f"voice '{voice.voice_id}' found")

    os.makedirs(output_path, exist_ok=True)
    output_metadata_file_path = os.path.join(output_path, "elevenlabs_audio.tsv")

    # Временный файл, который станет новым input_file_path.
    # Он будет содержать заголовок и все строки, которые не были успешно обработаны.
    temp_input_for_next_run_path = input_file_path + ".processing_temp"

    with open(input_file_path, "r", newline='', encoding="utf-8") as r_input_file:
        reader = csv.DictReader(r_input_file, delimiter="\t")
        input_header_fields = reader.fieldnames
        rows_to_process_from_input = list(reader)

    # Определяем, нужен ли заголовок в выходном файле метаданных
    create_output_header = not os.path.exists(output_metadata_file_path) or os.path.getsize(
        output_metadata_file_path) == 0
    output_file_mode = 'w' if create_output_header else 'a'

    try:
        with open(temp_input_for_next_run_path, "w", newline='', encoding="utf-8") as w_next_input_file, \
                open(output_metadata_file_path, output_file_mode, newline='', encoding='utf-8') as wr_output_file:

            writer_next_input = csv.writer(w_next_input_file, delimiter='\t')
            output_writer = csv.writer(wr_output_file, delimiter='\t')

            if input_header_fields:
                writer_next_input.writerow(input_header_fields)  # Пишем заголовок в новый "входной" файл

            if create_output_header and input_header_fields:  # Не пишем заголовок в выходной, если входной пуст
                output_writer.writerow(["id", "path", "text", "source", "style", "speaker"])

            if not input_header_fields and not rows_to_process_from_input:
                typer.echo("Исходный файл был пуст или без заголовка")

            elif input_header_fields:  # Только если есть заголовок, есть что обрабатывать или сохранять
                for i, row_dict in enumerate(rows_to_process_from_input):
                    try:
                        source = row_dict.get('source')
                        id = row_dict.get('id')
                        text = row_dict.get('text')

                        audio_dir_path = os.path.join(output_path, "audio", source)
                        os.makedirs(audio_dir_path, exist_ok=True)
                        audio_name = f"{source}_{id}{audio_format}"
                        relative_audio_path = os.path.join("audio", source, audio_name)
                        full_audio_path = os.path.join(audio_dir_path, audio_name)
                        audio_bytes = client.generate(
                            text=text,
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
                                compname = 'not compressed'

                                wavfile.setparams((nchannels, sampwidth, framerate, nframes, comptype, compname))
                                wavfile.writeframes(audio_bytes)
                        else:
                            save(audio_bytes, full_audio_path)

                        output_writer.writerow([id, relative_audio_path, text, source, "default", voice.name])
                        if i == limit:
                            raise Limit("limit reached")
                    except Limit as e:
                        for k in range(i + 1, len(rows_to_process_from_input)):
                            writer_next_input.writerow(
                                [rows_to_process_from_input[k].get(h, "") for h in input_header_fields])

                        # Передаем ошибку дальше, чтобы прервать выполнение и зафиксировать состояние
                        raise e

                    except Exception as e_row_processing:
                        # Записываем эту "проблемную" строку в temp_input_for_next_run_path
                        writer_next_input.writerow([row_dict.get(h, "") for h in input_header_fields])

                        # Также записываем все ОСТАВШИЕСЯ строки из первоначального списка
                        # в temp_input_for_next_run_path, так как они еще не были обработаны.
                        for k in range(i + 1, len(rows_to_process_from_input)):
                            writer_next_input.writerow(
                                [rows_to_process_from_input[k].get(h, "") for h in input_header_fields])

                        # Передаем ошибку дальше, чтобы прервать выполнение и зафиксировать состояние
                        raise e_row_processing
    except Exception as e_fatal:
        typer.echo(e_fatal, err=True)
        raise typer.Exit(code=1) from e_fatal
    finally:
        os.replace(temp_input_for_next_run_path, input_file_path)
