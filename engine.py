"""Fixed TTS module for Namma Vanni — Sarvam (primary) + Edge TTS (fallback)."""

import asyncio
import base64
import concurrent.futures
import logging
import os

import edge_tts
import requests

logger = logging.getLogger(__name__)

SARVAM_API_KEY: str = os.getenv("SARVAM_API_KEY", "")
SARVAM_TTS_URL = "https://api.sarvam.ai/text-to-speech"
TTS_OUTPUT_PATH = "verify.mp3"

SARVAM_TTS_LANG_MAP: dict[str, str] = {
    "kn": "kn-IN",
    "hi": "hi-IN",
    "en": "en-IN",
}

EDGE_TTS_VOICE_MAP: dict[str, str] = {
    "kn": "kn-IN-SapnaNeural",
    "hi": "hi-IN-SwaraNeural",
    "en": "en-IN-NeerjaNeural",
}
EDGE_TTS_FALLBACK_VOICE = "en-IN-NeerjaNeural"


def _normalize_lang(lang: str) -> str:
    """Normalize language code to 2-char lowercase key."""
    return lang.strip().lower()[:2]


def _sarvam_tts(text: str, lang: str) -> bytes | None:
    """
    Call Sarvam TTS API (bulbul:v1).
    Returns raw WAV/MP3 bytes on success, None on any failure.
    """
    lang_key = _normalize_lang(lang)
    target_language_code = SARVAM_TTS_LANG_MAP.get(lang_key, "kn-IN")

    payload = {
        "inputs": [text],
        "target_language_code": target_language_code,
        "speaker": "meera",
        "pitch": 0,
        "pace": 1.0,
        "loudness": 1.5,
        "speech_sample_rate": 8000,
        "enable_preprocessing": True,
        "model": "bulbul:v1",
    }
    headers = {
        "api-subscription-key": SARVAM_API_KEY,
        "Content-Type": "application/json",
    }

    try:
        logger.info("[SARVAM TTS] Requesting lang=%s (%s), chars=%d",
                    lang_key, target_language_code, len(text))
        res = requests.post(SARVAM_TTS_URL, headers=headers, json=payload, timeout=20)
        res.raise_for_status()

        data = res.json()
        audio_b64 = (data.get("audios") or [None])[0]
        if not audio_b64:
            logger.warning("[SARVAM TTS] Empty audios array in response.")
            return None

        audio_bytes = base64.b64decode(audio_b64)
        logger.info("[SARVAM TTS] Success — %d bytes received.", len(audio_bytes))
        return audio_bytes

    except requests.HTTPError as e:
        logger.warning("[SARVAM TTS] HTTP %s: %s", e.response.status_code, e.response.text[:200])
    except requests.Timeout:
        logger.warning("[SARVAM TTS] Request timed out.")
    except Exception as e:
        logger.warning("[SARVAM TTS] Unexpected error: %s", e)

    return None


async def _edge_tts_coroutine(text: str, voice: str, output_path: str) -> None:
    """Async Edge TTS synthesis — saves directly to output_path."""
    communicator = edge_tts.Communicate(text, voice)
    await communicator.save(output_path)


def _run_edge_tts(text: str, voice: str, output_path: str) -> bool:
    """
    Run Edge TTS safely whether or not an event loop is already running.
    Returns True on success, False on failure.
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Running inside FastAPI / Streamlit — use a thread pool
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(
                    asyncio.run, _edge_tts_coroutine(text, voice, output_path)
                )
                future.result(timeout=30)
        else:
            loop.run_until_complete(_edge_tts_coroutine(text, voice, output_path))
        return True
    except RuntimeError:
        try:
            asyncio.run(_edge_tts_coroutine(text, voice, output_path))
            return True
        except Exception as e:
            logger.warning("[EDGE TTS] asyncio.run fallback failed: %s", e)
    except concurrent.futures.TimeoutError:
        logger.warning("[EDGE TTS] Timed out after 30s.")
    except Exception as e:
        logger.warning("[EDGE TTS] Error: %s", e)

    return False


def _edge_tts(text: str, lang: str, output_path: str) -> bool:
    """
    Try Edge TTS with the correct voice for lang, then fallback voice.
    Returns True if audio was saved successfully.
    """
    lang_key = _normalize_lang(lang)
    primary_voice = EDGE_TTS_VOICE_MAP.get(lang_key, EDGE_TTS_FALLBACK_VOICE)

    logger.info("[EDGE TTS] Trying voice=%s for lang=%s", primary_voice, lang_key)
    if _run_edge_tts(text, primary_voice, output_path):
        logger.info("[EDGE TTS] Success with voice=%s", primary_voice)
        return True

    if primary_voice != EDGE_TTS_FALLBACK_VOICE:
        logger.warning("[EDGE TTS] Primary voice failed, trying fallback=%s", EDGE_TTS_FALLBACK_VOICE)
        if _run_edge_tts(text, EDGE_TTS_FALLBACK_VOICE, output_path):
            logger.info("[EDGE TTS] Fallback voice succeeded.")
            return True

    logger.error("[EDGE TTS] All voices failed for lang=%s.", lang_key)
    return False


def generate_tts(text: str, lang: str, output_path: str = TTS_OUTPUT_PATH) -> str | None:
    """
    Synthesise text to speech.

    Priority:
      1. Sarvam TTS (bulbul:v1, native Indian voices)
      2. Edge TTS  (Microsoft Neural voices, offline-capable)
      3. Return None if both fail

    Args:
        text: Text to synthesise.
        lang: 2-char language code ('kn', 'hi', 'en').
        output_path: Destination file path for the MP3.

    Returns:
        output_path on success, None if both engines fail.
    """
    if not text.strip():
        logger.warning("[TTS] Empty text — skipping synthesis.")
        return None

    lang_key = _normalize_lang(lang)
    logger.info("[TTS] Starting synthesis: lang=%s, chars=%d, out=%s",
                lang_key, len(text), output_path)

    # ── 1. Sarvam TTS ────────────────────────────────────────────────────────
    audio_bytes = _sarvam_tts(text, lang_key)
    if audio_bytes:
        try:
            with open(output_path, "wb") as f:
                f.write(audio_bytes)
            logger.info("[TTS] ✓ Sarvam TTS saved to %s (%d bytes)", output_path, len(audio_bytes))
            return output_path
        except OSError as e:
            logger.error("[TTS] Failed to write Sarvam audio to disk: %s", e)

    # ── 2. Edge TTS fallback ─────────────────────────────────────────────────
    logger.info("[TTS] Falling back to Edge TTS...")
    if _edge_tts(text, lang_key, output_path):
        logger.info("[TTS] ✓ Edge TTS saved to %s", output_path)
        return output_path

    # ── 3. Both failed ───────────────────────────────────────────────────────
    logger.error("[TTS] ✗ Both Sarvam and Edge TTS failed. No audio generated.")
    return None
