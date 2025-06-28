import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).parent.parent

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
ELEVENLABS_TOKEN = os.getenv("ELEVENLABS_TOKEN")
GEMINI_TOKEN = os.getenv("GEMINI_TOKEN")
OPENROUTER_TOKEN = os.getenv("OPENROUTER_TOKEN")