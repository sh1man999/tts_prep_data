## Neural commands
```ssh
python run.py neural generate-text --topic "Колл-центр" --provider "gemini" --model-name "gemini-2.5-flash"
```
```ssh
python run.py neural postprocess-file --jsonl-file-name "Колл-центр.jsonl" --provider "gemini" --model-name "gemini-2.5-flash"
python run.py neural postprocess-file --jsonl-file-name "Колл-центр.jsonl" --provider "openrouter" --model-name "openai/gpt-4.1"
```
```ssh
python run.py neural runorm-file
```
```ssh
python run.py neural generate-text --topics-file themes.txt
```
## Elevenlabs commands
```ssh
python run.py elevenlabs jsonl-to-audio
```
## HuggingFace commands
```ssh
python run.py hf upload-folder
```
```ssh
python run.py hf calculate-dataset-duration --input-path /home/sh1man/development/python/neural/tts_prep_data/output_elevenlabs/den4ikai/audio
```

Требования к аудио на выходе
ElevenLabs default voices: https://elevenlabs.io/docs/product/voices/default-voices
1. Минимум 1 час голоса для одного спикера
2. Минимальная длина звука составляет 0,8 секунды
3. Максимальная длина текста(фонем) составляет <= 510 токенов `Assert Len (Phonemes) <= 510`
4. аудио не более 45 секунд 

# Описания

payload_datasets - туда перекладываем файл для генерации аудио