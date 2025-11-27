# src/audio_utils.py

import io
from typing import Optional
from openai import OpenAI

client = OpenAI()  # Uses OPENAI_API_KEY from env


# =========================================================
# 🔊 SPEECH → TEXT (STT)
# =========================================================

def speech_to_text_from_bytes(audio_bytes: bytes) -> str:
    """
    Convert raw audio bytes into Hindi/Hinglish/English text.
    Avoids Urdu/Nastaliq script.
    """
    if not audio_bytes:
        return ""

    try:
        audio_file = io.BytesIO(audio_bytes)
        audio_file.name = "recording.webm"

        resp = client.audio.transcriptions.create(
            model="gpt-4o-mini-transcribe",
            file=audio_file,
            
            # Force Hindi language bias so transcription doesn't drift into Urdu
            language="hi",

            # Important prompt to enforce no Urdu output
            prompt=(
                "Transcribe the speech into either Hindi (Devanagari), Hinglish "
                "(Hindi written in Latin script), or English. "
                "⚠️ Do NOT use Urdu or Arabic/Nastaliq script even if the "
                "pronunciation sounds like Urdu. "
                "Prefer Devanagari Hindi when unsure."
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
    Correct handling of output bytes.
    """
    text = (text or "").strip()
    if not text:
        return b""

    try:
        response = client.audio.speech.create(
            model="gpt-4o-mini-tts",   # You can switch to "tts-1" for higher quality
            voice="alloy",
            input=text,
            response_format="mp3",
        )

        # New SDK returns audio bytes inside `.audio`
        audio_bytes = getattr(response, "audio", None)

        if audio_bytes:
            return audio_bytes

        # Fallback: older SDK versions may return bytes directly
        if isinstance(response, (bytes, bytearray)):
            return bytes(response)

        if hasattr(response, "to_bytes"):
            return response.to_bytes()

        return b""

    except Exception as e:
        print("TTS error:", e)
        return b""

