# src/audio_utils.py

import io
from typing import Optional

from openai import OpenAI

# This will automatically use OPENAI_API_KEY from st.secrets / env
client = OpenAI()


def text_to_speech_bytes(text: str) -> bytes:
    """
    Convert assistant text into an MP3 audio byte stream using OpenAI TTS.
    """
    if not text:
        return b""

    # If gpt-4o-mini-tts is not enabled for you, switch model="tts-1"
    response = client.audio.speech.create(
        model="gpt-4o-mini-tts",
        voice="alloy",
        input=text,
        format="mp3",
    )

    # In the new OpenAI Python client, response is already raw bytes
    # (if not, use response.read())
    try:
        audio_bytes = bytes(response)
    except TypeError:
        # fallback if response is a stream-like object
        audio_bytes = response.read()

    return audio_bytes


def speech_to_text_from_bytes(audio_bytes: bytes) -> Optional[str]:
    """
    Send raw audio bytes to OpenAI STT and return the transcribed text.
    """
    if not audio_bytes:
        return None

    file_obj = io.BytesIO(audio_bytes)
    file_obj.name = "audio.webm"  # name hint; format is not too strict

    # If gpt-4o-mini-transcribe is available, you can use that instead of whisper-1
    transcription = client.audio.transcriptions.create(
        model="whisper-1",
        file=file_obj,
        response_format="json",
    )

    text = getattr(transcription, "text", None)
    return text.strip() if text else None
