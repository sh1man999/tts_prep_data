import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).parent.parent

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
ELEVENLABS_TOKEN = os.getenv("ELEVENLABS_TOKEN")