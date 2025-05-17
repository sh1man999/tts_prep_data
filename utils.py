import contextlib
import wave

import torch


def get_available_gpus():
    if torch.cuda.is_available():
        return [(i, torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]
    else:
        return []


def get_wav_duration(file_path: str) -> float:
    with contextlib.closing(wave.open(file_path, 'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)
        return duration

def format_duration(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    milliseconds = int((seconds - int(seconds)) * 1000)
    if hours > 0:
        return f"{hours}ч {minutes}м {secs}с {milliseconds}мс"
    elif minutes > 0:
        return f"{minutes}м {secs}с {milliseconds}мс"
    else:
        return f"{secs}с {milliseconds}мс"