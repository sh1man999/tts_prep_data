## Neural commands
```ssh
python run.py neural preprocess_file
```
```ssh
python run.py neural generate-text --topic "Здоровое питание"
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

Требования к аудио на выходе
ElevenLabs default voices: https://elevenlabs.io/docs/product/voices/default-voices
1. Минимум 1 час голоса для одного спикера
2. Минимальная длина звука составляет 0,8 секунды
3. Максимальная длина текста(фонем) составляет <= 510 токенов `Assert Len (Phonemes) <= 510`
4. аудио не более 45 секунд 