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
    Try to avoid Urdu/Nastaliq script in output.
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

        if isinstance(resp, str):
            return resp.strip()
        if hasattr(resp, "text"):
            return resp.text.strip()

        return ""

    except Exception as e:
        print("STT error:", repr(e))
        return ""


# =========================================================
# 🔉 TEXT → SPEECH (TTS)
# =========================================================

def _extract_audio_bytes(response) -> Optional[bytes]:
    """
    Helper to pull raw audio bytes out of different SDK response shapes.
    """
    # 1) Some SDKs directly return bytes
    if isinstance(response, (bytes, bytearray)):
        return bytes(response)

    # 2) Some responses are file-like (BinaryAPIResponse) with .read()
    if hasattr(response, "read"):
        try:
            return response.read()
        except Exception:
            pass

    # 3) Some expose .audio with bytes
    audio_bytes = getattr(response, "audio", None)
    if isinstance(audio_bytes, (bytes, bytearray)):
        return bytes(audio_bytes)

    # 4) Some have .to_bytes()
    if hasattr(response, "to_bytes"):
        try:
            return response.to_bytes()
        except Exception:
            pass

    return None


def text_to_speech_bytes(text: str) -> Optional[bytes]:
    """
    Convert text into MP3 audio using OpenAI TTS.
    Tries both the "old" and "new" API styles for compatibility.
    """
    text = (text or "").strip()
    if not text:
        return b""

    # First try: some SDKs expect `format="mp3"` and return a streaming object
    try:
        resp = client.audio.speech.create(
            model="gpt-4o-mini-tts",   # if this model is unavailable, you can try "tts-1"
            voice="alloy",
            input=text,
            format="mp3",             # ⚠️ this is what raised TypeError earlier on some versions
        )
        audio_bytes = _extract_audio_bytes(resp)
        if audio_bytes:
            return audio_bytes
    except TypeError as e:
        # This is the case we saw before: "got unexpected keyword argument 'format'"
        print("TTS TypeError with format=mp3 style:", repr(e))
    except Exception as e:
        # Any other error from this first attempt
        print("TTS error (first attempt):", repr(e))

    # Second try: newer SDK style (no `format` argument, returns bytes directly)
    try:
        resp2 = client.audio.speech.create(
            model="gpt-4o-mini-tts",   # or "tts-1" if needed
            voice="alloy",
            input=text,
        )
        audio_bytes2 = _extract_audio_bytes(resp2)
        if audio_bytes2:
            return audio_bytes2
    except Exception as e2:
        print("TTS error (second attempt):", repr(e2))

    # If we reach here, TTS failed
    return b""
