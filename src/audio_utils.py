# src/audio_utils.py

import io
from typing import Optional

from openai import OpenAI

# Uses OPENAI_API_KEY from environment
client = OpenAI()


# =========================================================
# 🔊 SPEECH → TEXT (STT)
# =========================================================

def speech_to_text_from_bytes(audio_bytes: bytes) -> str:
    """
    Convert raw audio bytes into Hindi/Hinglish/English text.
    Avoids Urdu/Nastaliq script as much as possible.
    """
    if not audio_bytes:
        return ""

    try:
        audio_file = io.BytesIO(audio_bytes)
        audio_file.name = "recording.webm"

        resp = client.audio.transcriptions.create(
            model="gpt-4o-mini-transcribe",
            file=audio_file,
            language="hi",  # bias towards Hindi instead of Urdu
            prompt=(
                "Transcribe the speech into either Hindi in Devanagari, "
                "Hinglish (Hindi written with English letters), or English. "
                "Do NOT use Urdu / Arabic / Nastaliq script."
            ),
            response_format="text",
        )

        # Handle response variations
        if isinstance(resp, str):
            return resp.strip()
        if hasattr(resp, "text"):
            return resp.text.strip()

        return ""

    except Exception as e:
        print("STT error:", e)
        return ""


# =========================================================
# 🔉 TEXT → SPEECH (TTS)
# =========================================================

def text_to_speech_bytes(text: str) -> Optional[bytes]:
    """
    Convert text into MP3 audio using OpenAI's TTS model.
    Returns raw audio bytes.
    """
    text = (text or "").strip()
    if not text:
        return b""

    try:
        response = client.audio.speech.create(
            model="gpt-4o-mini-tts",  # or "tts-1" if you prefer
            voice="alloy",
            input=text,
            response_format="mp3",
        )

        # New SDK: raw mp3 bytes are in .audio
        audio_bytes = getattr(response, "audio", None)

        if audio_bytes:
            return audio_bytes

        # Fallbacks for other SDK behaviours
        if isinstance(response, (bytes, bytearray)):
            return bytes(response)
        if hasattr(response, "to_bytes"):
            return response.to_bytes()

        return b""

    except Exception as e:
        print("TTS error:", e)
        return b""
