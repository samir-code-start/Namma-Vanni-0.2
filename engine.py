"""engine.py — Namma Vanni AI pipeline: STT, LLM analysis, TTS, feedback logging, mock mode."""

import asyncio
import concurrent.futures
import csv
import json
import logging
import os
import re
import requests
from datetime import datetime

import edge_tts
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Secrets: support both .env (local) and st.secrets (Streamlit Cloud)
# ---------------------------------------------------------------------------
def _get_secret(key: str, default: str = "") -> str:
    val = os.getenv(key, "")
    if val:
        return val
    try:
        import streamlit as st
        return st.secrets.get(key, default)
    except Exception:
        return default

def translate_to_english(text: str) -> str:
    """Translates any text to English via Sarvam LLM. Safe fallback to original."""
    if not text.strip(): return ""
    try:
        return _sarvam_chat(
            messages=[{"role": "user", "content": f"Translate to English ONLY: '{text}'"}],
            temperature=0.1, max_tokens=100,
        )
    except Exception: return text  # Fail-safe

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MOCK_MODE: bool = os.getenv("MOCK_MODE", "False").lower() == "true"
SARVAM_API_KEY: str = _get_secret("SARVAM_API_KEY")
SARVAM_STT_URL = "https://api.sarvam.ai/speech-to-text-translate"
GROQ_API_KEY: str = _get_secret("GROQ_API_KEY")
LLM_MODEL = "llama-3.3-70b-versatile"
GROQ_LLM_URL = "https://api.groq.com/openai/v1/chat/completions"

KANADA_FIXES = {
    "ನಮ್ವ": "ನಮ್ಮ",
    "ವಣಿ": "ವಾಣಿ",
    "ತುಂಬಾ": "ತುಂಬಾ",
    "ಹಾಳಾಗಿದೆ": "ಹಾಳಾಗಿದೆ",
    "ರಸ್ತೆ": "ರಸ್ತೆ",
    "ಪಾಣಿ": "ಪಾಣಿ",
    "ಕೇಂದ್ರ": "ಕೇಂದ್ರ",
    "ಫೋನ್": "ಫೋನ್"
}

FEEDBACK_FILE = "feedback.csv"
FEEDBACK_HEADERS = [
    "timestamp", "language", "raw_text", "ai_issue",
    "confidence", "sentiment", "citizen_response",
    "agent_correction", "handover", "feedback_weight",
]

TTS_VOICE_MAP: dict[str, str] = {
    "kn": "kn-IN-VarunNeural",
    "hi": "hi-IN-MadhurNeural",
    "en": "en-IN-NeerjaNeural",
}
TTS_FALLBACK_VOICE = "en-IN-NeerjaNeural"
TTS_OUTPUT_PATH = "verify.mp3"

# Sarvam TTS & Translate config
SARVAM_TTS_URL = "https://api.sarvam.ai/text-to-speech"
SARVAM_TRANSLATE_URL = "https://api.sarvam.ai/translate"
SARVAM_STT_ORIGINAL_URL = "https://api.sarvam.ai/speech-to-text"
SARVAM_TTS_LANG_MAP = {
    "kn": "kn-IN",
    "hi": "hi-IN",
    "en": "en-IN",
}

# ---------------------------------------------------------------------------
# Mock payloads
# ---------------------------------------------------------------------------
_MOCK_TRANSCRIPT = "ನಮ್ಮ ಊರಿ ರಸ್ತೆ ತುಂಬಾ ಕೆಟ್ಟಿದೆ, ಅಧಿಕಾರಿಗಳನ್ನು ಕಳುಹಿಸಿ"

_MOCK_ANALYSIS: dict = {
    "language": "kn",
    "normalized_issue": "The road in the caller's village is severely damaged and needs official inspection.",
    "confidence": 0.92,
    "sentiment": "urgent",
    "verification_prompt": "Your village road is badly damaged and you need officials to visit. Did I understand correctly? Say Yes or No.",
    "handover": False,
}

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = """You are Namma Vanni, an expert AI analyst for Karnataka's 1092 Civic Helpline.

🎯 CORE TASKS:
1. DOMAIN TAXONOMY: Map fragmented speech to issues like water leakage, street lights, garbage, etc.
2. SEMANTIC EXTRACTION: Extract PRIMARY issue. Partial understanding > failure.
3. CONFIDENCE RUBRIC:
   - High (0.85-1.0): Clear issue + location
   - Medium (0.60-0.84): Clear issue, missing location
   - Low (0.30-0.59): Ambiguous or dialect-heavy
   - Critical (<0.30): Incoherent
4. DYNAMIC VERIFICATION: Generate a SPECIFIC clarification question tailored to the issue.
5. SENTIMENT DETECTION: Analyze tone, urgency.

OUTPUT STRICT JSON ONLY:
{
  "language": "en|kn|hi",
  "confidence": 0.0-1.0,
  "sentiment": "calm|confused|urgent|distressed|angry|fear",
  "normalized_issue": "Clean 1-line summary",
  "verification_prompt": "Natural question clarifying the specific issue. End with: 'Did I understand correctly? Say Yes or No.' Max 20 words.",
  "handover": true|false
}

DIALECT AWARENESS (Karnataka):
- North Karnataka dialects: "enu" → "ēnu" (what), "barri" → "banni" (come)
- Bangalore Urban: Code-mixed Kannada-English ("current hogide" = power cut)
- Old Mysuru: Formal Kannada with "appa/amma" honorifics
- Hindi-belt migrants: Hinglish mixed with Kannada words
- Common civic expressions:
  "current hogide" = power cut, "neer bandilla" = no water supply
  "gutter overflow" = drainage blockage, "kasa collect aagilla" = garbage not collected
  "road kharab" = road damaged, "light illa" = no street light

ISSUE CATEGORIES (Karnataka 1092):
- ROAD: potholes, damaged roads, flooding, waterlogging
- WATER: supply disruption, contamination, leakage, bore well
- ELECTRICITY: power cuts, street lights, transformer failure
- GARBAGE: collection missed, illegal dumping, burning
- DRAINAGE: overflow, blockage, sewage leak
- SAFETY: crime, harassment, emergency, accidents
- GOVERNMENT: corruption, missing services, documentation issues

GUARDRAILS:
- If confidence < 0.7 -> handover=true
- If sentiment in [distressed, angry, fear] -> handover=true"""

# ---------------------------------------------------------------------------
# Groq LLM helper (OpenAI-compatible API with Bearer token)
# ---------------------------------------------------------------------------

def _sarvam_chat(messages: list, temperature: float = 0.1, max_tokens: int = 500) -> str:
    """Make a chat completion call to Groq LLM endpoint (llama-3.3-70b-versatile)."""
    res = requests.post(
        GROQ_LLM_URL,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {GROQ_API_KEY}",
        },
        json={
            "model": LLM_MODEL,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
        timeout=30,
    )
    res.raise_for_status()
    data = res.json()
    return data["choices"][0]["message"]["content"].strip()

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
_VALID_LANGUAGES = {"kn", "hi", "en"}
_VALID_SENTIMENTS = {"calm", "confused", "urgent", "distressed", "angry", "fear"}

def _strip_fences(raw: str) -> str:
    """Remove markdown/code fences from a raw string."""
    return re.sub(r"```(?:json)?\s*|\s*```", "", raw, flags=re.IGNORECASE).strip()

def _enforce_guardrails(data: dict) -> dict:
    """Validate schema fields and enforce handover guardrail rules."""
    language = data.get("language", "en")
    if language not in _VALID_LANGUAGES:
        language = "en"

    sentiment = data.get("sentiment", "calm")
    if sentiment not in _VALID_SENTIMENTS:
        sentiment = "calm"

    try:
        confidence = float(data.get("confidence", 0.5))
        confidence = max(0.0, min(1.0, confidence))
    except (TypeError, ValueError):
        confidence = 0.5

    normalized_issue = str(data.get("normalized_issue", "Unclear report. Needs agent clarification.")).strip()
    if not normalized_issue or normalized_issue.lower() == "issue could not be determined":
        normalized_issue = "Unclear report. Needs agent clarification."

    verification_prompt = str(
        data.get(
            "verification_prompt",
            "Could you please repeat your issue? Did I understand correctly? Say Yes or No.",
        )
    ).strip()

    handover: bool = bool(data.get("handover", False))
    if confidence < 0.7:
        handover = True
    elif sentiment in {"distressed", "angry", "fear"} or normalized_issue == "Unclear report. Needs agent clarification.":
        handover = True

    return {
        "language": language,
        "normalized_issue": normalized_issue,
        "confidence": confidence,
        "sentiment": sentiment,
        "verification_prompt": verification_prompt,
        "handover": handover,
    }

async def _tts_coroutine(text: str, voice: str, output_path: str) -> None:
    """Async coroutine: synthesise speech with edge-tts and save to disk."""
    communicator = edge_tts.Communicate(text, voice)
    await communicator.save(output_path)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_confirmation(transcript: str) -> dict:
    """Robustly parses multilingual yes/no & summarizes longer inputs."""
    t = transcript.strip().lower()
    if not t: return {"intent": "unclear", "summary": ""}
    
    # Multi-language affirmation/negation patterns
    yes_tokens = ["yes","yeah","yep","correct","right","exactly","agreed","okay","ok",
                  "haan","han","hān","ji","theek","sahi","haa","bilkul","sari","sha","hana"]
    no_tokens = ["no","nahi","nahin","naahi","galat","bhool","phir se","wapos","na",
                 "wrong","incorrect","missed","repeat","again","try again",
                 "illa","illai","alla","muddu","kadliya","galti","bharosa nahi"]
                
    yes_hits = [w for w in yes_tokens if w in t]
    no_hits = [w for w in no_tokens if w in t]
    # Partial-correct patterns ("yes but...", "almost", "haan par...")
    partial_tokens = ["but","almost","mostly","partially","partly","half","kinda","sort of",
                      "not fully","not exactly","haan par","haan lekin","sari aadre","yes but",
                      "thoda","kuch","aadre"]
    partial_hits = [w for w in partial_tokens if w in t]
    
    # Check partial FIRST ("yes but..." should be partial, not confirmed)
    if len(partial_hits) > 0 and len(yes_hits) > 0:
        return {"intent": "partial", "summary": t[:80]}
    
    if len(yes_hits) > 0 and len(no_hits) == 0:
        return {"intent": "confirmed", "summary": t[:80]}
    if len(no_hits) > 0 and len(yes_hits) == 0:
        return {"intent": "denied", "summary": t[:80]}
    if len(t) > 20 or (len(yes_hits) > 0 and len(no_hits) > 0):
        # Longer/mixed input → route to quick LLM intent extraction
        try:
            system = "You are a helpline assistant. Respond ONLY with JSON: {\"intent\":\"confirmed\"|\"denied\"|\"partial\"|\"unclear\",\"summary\":\"one-line clarification of user's exact meaning\"}. 'partial' means the citizen said something like 'yes but...' or 'almost correct'. Analyze this spoken reply:"
            raw = _sarvam_chat(
                messages=[{"role": "system", "content": system}, {"role": "user", "content": t}],
                temperature=0.1, max_tokens=80,
            )
            m = re.search(r'\{.*\}', raw, re.DOTALL)
            if m: return json.loads(m.group())
        except Exception: pass
        return {"intent": "unclear", "summary": t[:80]}
    return {"intent": "unclear", "summary": t[:80]}

def normalize_kannada(text: str) -> str:
    """Normalize Kannada text using KANADA_FIXES."""
    for wrong, right in KANADA_FIXES.items():
        text = text.replace(wrong, right)
    return text

def normalize_transcript(text: str, lang: str) -> str:
    """Cross-language ASR drift correction for Kannada/Hindi/English."""
    if not text: return ""
    lang = lang.lower()[:2]
    KN_FIXES = {"ನಮ್ವ":"ನಮ್ಮ", "ವಣಿ":"ವಾಣಿ", "ತುಂಬಾ":"ತುಂಬಾ", "ಹಾಳಾಗಿದೆ":"ಹಾಳಾಗಿದೆ", "ರಸ್ತೆ":"ರಸ್ತೆ", "ಪಾಣಿ":"ಪಾಣಿ", "ಕೇಂದ್ರ":"ಕೇಂದ್ರ", "ಫೋನ್":"ಫೋನ್", "ಬೇಕು":"ಬೇಕು", "ಸೇವೆ":"ಸೇವೆ"}
    HI_FIXES = {"क्यो":"क्यों", "कहा":"कहाँ", "ठिक":"ठीक", "सही":"सही", "रस्ता":"रास्ता", "बात":"बात", "दिया":"दिया", "लिए":"लिए", "में":"में", "पानि":"पानी", "सफाय":"सफाई"}
    EN_FIXES = {"teh":"the", "plz":"please", "thk":"thank", "recieve":"receive", "adress":"address", "wont":"won't", "cant":"can't", "im":"I'm", "waterline":"water line", "paniline":"pani line", "bandh kr do":"shut off"}
    fixes = {"kn": KN_FIXES, "hi": HI_FIXES, "en": EN_FIXES}
    current = fixes.get(lang, EN_FIXES)
    out = text
    for k, v in current.items(): out = out.replace(k, v)
    return out.strip().replace("  ", " ").replace("\n", " ")

def transcribe_audio(audio_path: str) -> tuple[str, str, str]:
    """Returns (english_text, detected_language_code, original_transcript).
    
    original_transcript: raw text in citizen's language via Sarvam /speech-to-text (non-translate).
    Falls back to empty string if Sarvam STT original fails.
    """
    # Mock Mode Check
    if os.getenv("MOCK_MODE", "").lower() == "true":
        return "The road in our village is very bad, please send officials", "kn", _MOCK_TRANSCRIPT

    print(f"[STT] Transcribing+translating via Sarvam: {audio_path}", flush=True)
    if not os.path.isfile(audio_path) or os.path.getsize(audio_path) < 50:
        return "", "en", ""

    try:
        with open(audio_path, "rb") as f:
            res = requests.post(
                SARVAM_STT_URL,
                files={"file": ("audio.wav", f, "audio/wav")},
                headers={"api-subscription-key": SARVAM_API_KEY},
                timeout=20,
            )

        res.raise_for_status()
        data = res.json()
        # The translate endpoint returns English text directly
        english_text = (data.get("transcript") or data.get("text") or "").strip()

        # Detect original language for TTS voice routing
        raw_lang = (data.get("language_code") or data.get("detected_language") or "kn").split("-")[0][:2]
        if raw_lang not in _VALID_LANGUAGES:
            raw_lang = "kn"

        original_text = _sarvam_stt_original(audio_path)

        print(f"[STT+TRANSLATE SUCCESS] Lang: {raw_lang} | Len: {len(english_text)}", flush=True)
        return english_text, raw_lang, original_text

    except Exception as e:
        print(f"[STT ERROR] Sarvam Failed: {e}", flush=True)
        return "", "en", ""  # Graceful failure

def extract_json(text: str) -> dict:
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try: return json.loads(match.group())
        except json.JSONDecodeError as e: raise ValueError(f"Invalid JSON: {e}")
    raise ValueError("No JSON object found in LLM response")

def analyze_transcript(text: str) -> dict:
    """Analyze citizen transcript via Sarvam LLM and return structured JSON."""
    if MOCK_MODE:
        return _MOCK_ANALYSIS

    logging.info(f"[LLM INPUT] {text[:80]}{'...' if len(text)>80 else ''}")

    try:
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": f"CITIZEN TRANSCRIPT: '{text}'"}
        ]
        raw = _sarvam_chat(messages=messages, temperature=0.1, max_tokens=500)
        clean_raw = _strip_fences(raw)
        parsed = extract_json(clean_raw)
        return _enforce_guardrails(parsed)
    except Exception as e:
        logging.error(f"[LLM FAIL] {e}")
        return _enforce_guardrails({
            "language": "en",
            "normalized_issue": "Unable to parse request. Please repeat.",
            "confidence": 0.1,
            "sentiment": "confused",
            "verification_prompt": "I didn't catch that. Could you please say it again?",
            "handover": True,
        })

def re_analyze_transcript(new_text: str, previous_analysis: dict, feedback_type: str = "denied") -> dict:
    """Re-analyze with context from a previous denied/partial attempt.
    
    Args:
        new_text: New citizen transcript (English).
        previous_analysis: Previous ai_data dict that was rejected/partial.
        feedback_type: "denied" or "partial" — changes the LLM context framing.
    """
    if MOCK_MODE:
        return _MOCK_ANALYSIS

    prev_issue = previous_analysis.get("normalized_issue", "")
    if feedback_type == "partial":
        context = (
            f"Previous AI analysis was PARTIALLY CORRECT. "
            f"Previous AI summary: '{prev_issue}'. "
            f"Citizen's clarification/correction: '{new_text}'. "
            f"Refine the analysis incorporating the citizen's feedback. "
            f"Keep what was correct and fix what was wrong."
        )
    else:
        context = (
            f"Previous AI analysis was DENIED by the citizen. "
            f"Previous AI summary: '{prev_issue}'. "
            f"Citizen's new recording/correction: '{new_text}'. "
            f"Re-analyze from scratch incorporating the citizen's feedback."
        )

    logging.info(f"[RE-ANALYZE] Type: {feedback_type} | New input: {new_text[:60]}")
    try:
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": context},
        ]
        raw = _sarvam_chat(messages=messages, temperature=0.1, max_tokens=500)
        parsed = extract_json(_strip_fences(raw))
        return _enforce_guardrails(parsed)
    except Exception as e:
        logging.error(f"[RE-ANALYZE FAIL] {e}")
        return _enforce_guardrails({
            "language": previous_analysis.get("language", "en"),
            "normalized_issue": prev_issue or "Unable to re-analyze. Needs agent.",
            "confidence": 0.3,
            "sentiment": "confused",
            "verification_prompt": "I'm still having trouble understanding. Could you explain once more?",
            "handover": True,
        })

def _sarvam_tts(text: str, lang: str) -> bytes | None:
    """Call Sarvam TTS API. Returns audio bytes or None on failure."""
    target_lang = SARVAM_TTS_LANG_MAP.get(lang.lower().strip(), "en-IN")
    try:
        res = requests.post(
            SARVAM_TTS_URL,
            headers={
                "Content-Type": "application/json",
                "api-subscription-key": SARVAM_API_KEY,
            },
            json={
                "inputs": [text],
                "target_language_code": target_lang,
                "speaker": "meera",
                "model": "bulbul:v2",
            },
            timeout=20,
        )
        res.raise_for_status()
        data = res.json()
        # Response contains base64 audio
        import base64
        audio_b64 = data.get("audios", [None])[0]
        if audio_b64:
            logger.info("[SARVAM TTS] Success for lang=%s", target_lang)
            return base64.b64decode(audio_b64)
    except Exception as e:
        logger.warning("[SARVAM TTS] Failed: %s — will fallback to Edge TTS", e)
    return None

def _sarvam_translate(text: str, source_lang: str, target_lang: str) -> str:
    """Translate text via Sarvam /translate API. Returns translated text or original on failure."""
    try:
        res = requests.post(
            SARVAM_TRANSLATE_URL,
            headers={
                "Content-Type": "application/json",
                "api-subscription-key": SARVAM_API_KEY,
            },
            json={
                "input": text,
                "source_language_code": source_lang,
                "target_language_code": target_lang,
            },
            timeout=15,
        )
        res.raise_for_status()
        translated = res.json().get("translated_text", "").strip()
        if translated:
            logger.info("[SARVAM TRANSLATE] %s -> %s OK", source_lang, target_lang)
            return translated
    except Exception as e:
        logger.warning("[SARVAM TRANSLATE] Failed: %s — returning original", e)
    return text  # Fail-safe: return original

def generate_tts(text: str, lang: str) -> str:
    """Synthesise verification prompt via Sarvam TTS (primary) or Edge TTS (fallback)."""
    if MOCK_MODE:
        logger.info("[MOCK] generate_tts() -> writing stub verify.mp3.")
        with open(TTS_OUTPUT_PATH, "wb") as f:
            f.write(b"")  # zero-byte stub
        return TTS_OUTPUT_PATH

    logger.info("TTS: lang=%s, chars=%d", lang, len(text))

    # Try Sarvam TTS first (native Indian voices)
    audio_bytes = _sarvam_tts(text, lang)
    if audio_bytes:
        with open(TTS_OUTPUT_PATH, "wb") as f:
            f.write(audio_bytes)
        logger.info("[SARVAM TTS] Saved to %s", TTS_OUTPUT_PATH)
        return TTS_OUTPUT_PATH

    # Fallback to Edge TTS
    logger.info("[EDGE TTS FALLBACK] Trying Edge TTS...")
    voice = TTS_VOICE_MAP.get(lang.lower().strip(), TTS_FALLBACK_VOICE)
    try:
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    future = pool.submit(asyncio.run, _tts_coroutine(text, voice, TTS_OUTPUT_PATH))
                    future.result(timeout=30)
            else:
                loop.run_until_complete(_tts_coroutine(text, voice, TTS_OUTPUT_PATH))
        except RuntimeError:
            asyncio.run(_tts_coroutine(text, voice, TTS_OUTPUT_PATH))

        logger.info("TTS saved to %s", TTS_OUTPUT_PATH)
        return TTS_OUTPUT_PATH

    except Exception as exc:
        logger.warning("TTS failed with voice %s: %s — retrying with fallback.", voice, exc)
        try:
            asyncio.run(_tts_coroutine(text, TTS_FALLBACK_VOICE, TTS_OUTPUT_PATH))
            logger.info("TTS fallback succeeded.")
            return TTS_OUTPUT_PATH
        except Exception as fallback_exc:
            logger.error("TTS fallback also failed: %s", fallback_exc)
            return TTS_OUTPUT_PATH

def _sarvam_stt_original(audio_path: str) -> str:
    """Get original-language transcript via Sarvam /speech-to-text (non-translate)."""
    try:
        with open(audio_path, "rb") as f:
            res = requests.post(
                SARVAM_STT_ORIGINAL_URL,
                files={"file": ("audio.wav", f, "audio/wav")},
                headers={"api-subscription-key": SARVAM_API_KEY},
                data={"language_code": "unknown"},
                timeout=20,
            )
        res.raise_for_status()
        data = res.json()
        original = (data.get("transcript") or data.get("text") or "").strip()
        logger.info("[SARVAM STT ORIGINAL] Got %d chars", len(original))
        return original
    except Exception as e:
        logger.warning("[SARVAM STT ORIGINAL] Failed: %s", e)
        return ""

def process_audio(audio_path: str) -> dict:
    logging.info(f"[PROCESS] Starting for {audio_path}")
    raw_text, lang, original_text = transcribe_audio(audio_path)
    logging.info(f"[STT OUTPUT] Lang: {lang}, Text Length: {len(raw_text)}")
    
    if not raw_text.strip():
        logging.warning("[STT] Empty transcript. Returning fallback.")
        fallback = _enforce_guardrails({
            "language": lang, 
            "normalized_issue": "I couldn't hear clearly. Please speak again.",
            "confidence": 0.2, 
            "sentiment": "confused", 
            "verification_prompt": "Please try recording again.", 
            "handover": False
        })
        return {**fallback, "raw_text": "", "original_text": "", "verify_tts_path": "verify.mp3"}
                
    # raw_text is already English (from speech-to-text-translate)
    ai_data = analyze_transcript(raw_text)
    
    # Phase 5: Translate verification prompt to citizen's language
    verification_prompt = ai_data.get("verification_prompt", "")
    citizen_lang = ai_data.get("language", "kn")
    tts_text = verification_prompt
    
    # Sarvam bulbul:v2 rejects English — always route to Kannada if citizen spoke English
    tts_lang = citizen_lang if citizen_lang != "en" else "kn"
    if verification_prompt:
        target_lang_code = SARVAM_TTS_LANG_MAP.get(tts_lang, "kn-IN")
        translated_prompt = _sarvam_translate(verification_prompt, "en-IN", target_lang_code)
        ai_data["verification_prompt_translated"] = translated_prompt
        tts_text = translated_prompt
        citizen_lang = tts_lang  # ensure TTS uses Indic voice
    
    tts_path = generate_tts(tts_text, citizen_lang)
    
    final = {**ai_data, "raw_text": raw_text, "original_text": original_text, "verify_tts_path": tts_path}
    logging.info(f"[PIPELINE OK] Confidence: {final['confidence']}, Handover: {final['handover']}")
    return final

def log_feedback(data: dict) -> None:
    """Append a feedback row to feedback.csv; creates file with headers if it does not exist."""
    file_exists = os.path.isfile(FEEDBACK_FILE)
    try:
        with open(FEEDBACK_FILE, "a", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=FEEDBACK_HEADERS, extrasaction="ignore")
            if not file_exists:
                writer.writeheader()
                logger.info("Created %s with headers.", FEEDBACK_FILE)
            
            # Phase 8: Calculate confidence-weighted feedback signal
            response = data.get("citizen_response", "")
            try:
                confidence = float(data.get("confidence", 0.5))
            except (TypeError, ValueError):
                confidence = 0.5

            if response == "Confirmed":
                weight = round(0.5 + (confidence * 0.5), 2)   # 0.5-1.0 strong positive
            elif response == "Partial":
                weight = round(0.3 * confidence, 2)            # 0.0-0.3 weak positive
            elif response == "Handover":
                weight = round(-0.5 * confidence, 2)           # negative signal
            else:
                weight = 0.0

            row = {
                "timestamp": data.get("timestamp", datetime.utcnow().isoformat()),
                "language": data.get("language", ""),
                "raw_text": data.get("raw_text", ""),
                "ai_issue": data.get("normalized_issue", ""),
                "confidence": data.get("confidence", ""),
                "sentiment": data.get("sentiment", ""),
                "citizen_response": data.get("citizen_response", ""),
                "agent_correction": data.get("agent_correction", ""),
                "handover": data.get("handover", ""),
                "feedback_weight": weight,
            }
            writer.writerow(row)
            logger.info("Feedback logged to %s (weight=%.2f).", FEEDBACK_FILE, weight)
    except Exception as exc:
        logger.error("log_feedback() failed: %s", exc)

