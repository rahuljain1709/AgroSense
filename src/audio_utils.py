# src/audio_utils.py

import io
from typing import Optional

from openai import OpenAI

client = OpenAI()  # Uses OPENAI_API_KEY from env


# ---------- SPEECH → TEXT (STT) ----------

def speech_to_text_from_bytes(audio_bytes: bytes) -> str:
    """
    Take raw audio bytes (from mic_recorder) and return transcript text.
    """
    if not audio_bytes:
        return ""

    try:
        # Wrap bytes in a file-like object for the API
        audio_file = io.BytesIO(audio_bytes)
        audio_file.name = "recording.webm"  # extension just needs to be valid

        resp = client.audio.transcriptions.create(
            model="gpt-4o-mini-transcribe",  # or "gpt-4o-mini-audio" variant
            file=audio_file,
              language="hi",  
            response_format="text",
        )
        # resp is usually a string or an object with `.text`
        if isinstance(resp, str):
            return resp.strip()
        if hasattr(resp, "text"):
            return resp.text.strip()
        return ""
    except Exception as e:
        print("STT error:", e)
        return ""


# ---------- TEXT → SPEECH (TTS) ----------

def text_to_speech_bytes(text: str) -> Optional[bytes]:
    """
    Convert text to speech using OpenAI TTS and return audio bytes (mp3).
    """
    text = (text or "").strip()
    if not text:
        return b""

    try:
        # NOTE: `format=` is NOT valid.
        # Use `response_format` or rely on default (mp3).
        response = client.audio.speech.create(
            model="gpt-4o-mini-tts",   # or "tts-1" if you prefer
            voice="alloy",
            input=text,
            response_format="mp3",     # explicit, works with new SDK
        )
        audio_bytes = response.read()
        # In the current SDK, response.audio contains the raw bytes
        #audio_bytes = getattr(response, "audio", None)

        # Fallbacks in case the SDK version behaves slightly differently
        if audio_bytes is None:
            if isinstance(response, (bytes, bytearray)):
                return bytes(response)
            if hasattr(response, "to_bytes"):
                return response.to_bytes()
            return b""

        return audio_bytes
    except Exception as e:
        print("TTS error:", e)
        return b""
