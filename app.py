import os
import tempfile
import pandas as pd
import streamlit as st

from engine import (
    MOCK_MODE,
    FEEDBACK_FILE,
    log_feedback,
    parse_confirmation,
    process_audio,
    transcribe_audio,
    re_analyze_transcript,
    generate_tts,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Namma Vanni — 1092 AI Helpline", page_icon="📞", layout="wide")

# ---------------------------------------------------------------------------
# Dark Command Center — Global Design System
# ---------------------------------------------------------------------------
def init_global_styles():
    st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');
:root {
  --bg-base: #0d0f14; --bg-surface: #161921; --bg-surface-2: #1e2230;
  --border-subtle: rgba(255,255,255,0.07); --border-medium: rgba(255,255,255,0.12);
  --text-primary: #f0f2f8; --text-secondary: #9da5be; --text-muted: #6b7080;
  --accent-red: #e8445a; --accent-green: #1a9e5c; --accent-blue: #3b8af5; --accent-yellow: #d4a017;
}
[data-testid="stAppViewContainer"] { background-color: var(--bg-base) !important; color: var(--text-primary) !important; }
[data-testid="stHeader"] { background-color: var(--bg-base) !important; }
[data-testid="stVerticalBlock"] > div { padding-top: 0 !important; }
.block-container { padding-top: 1rem !important; padding-bottom: 2rem !important; max-width: 900px; }
h1, h2, h3, h4, label, p, span, li { font-family: 'Syne', sans-serif !important; color: var(--text-primary) !important; }
.font-mono { font-family: 'Space Mono', monospace !important; letter-spacing: -0.5px; }

/* Nav Bar */
.nav-bar { display:flex; justify-content:space-between; align-items:center; padding:16px 0; border-bottom:1px solid var(--border-subtle); margin-bottom:24px; margin-top:8px; }
.nav-brand-group { display:flex; flex-direction:column; }
.nav-icon-bg { width:38px; height:38px; background:linear-gradient(135deg,#e8445a,#ff7b54); border-radius:10px; display:flex; align-items:center; justify-content:center; font-size:18px; }
.nav-badge { background:rgba(232,68,90,0.15); color:var(--accent-red); border:1px solid rgba(232,68,90,0.3); font-size:11px; font-family:'Space Mono'; padding:4px 10px; border-radius:6px; text-transform:uppercase; letter-spacing:1px; }

/* Cards */
.card { background:var(--bg-surface); border:1px solid var(--border-subtle); border-radius:12px; padding:18px; margin-bottom:12px; }
.section-label { font-size:11px; text-transform:uppercase; letter-spacing:0.8px; color:var(--text-muted) !important; font-weight:700; margin-bottom:8px; display:block; }

/* Pills */
.pill { display:inline-flex; align-items:center; gap:6px; padding:5px 12px; border-radius:100px; font-size:12px; font-weight:600; }
.lang-pill { background:rgba(96,108,200,0.18); color:#8896e8 !important; border:1px solid rgba(96,108,200,0.25); }
.conf-pill { background:rgba(59,164,243,0.15); color:#58aef5 !important; border:1px solid rgba(59,164,243,0.25); }
.emotion-pill { border-radius:100px; font-size:12px; font-weight:600; padding:5px 12px; display:inline-flex; align-items:center; gap:6px; }
.emotion-calm { background:rgba(61,214,140,0.15); color:#3dd68c !important; border:1px solid rgba(61,214,140,0.25); }
.emotion-confused { background:rgba(212,160,23,0.15); color:#d4a017 !important; border:1px solid rgba(212,160,23,0.25); }
.emotion-urgent { background:rgba(255,123,84,0.15); color:#ff7b54 !important; border:1px solid rgba(255,123,84,0.25); }
.emotion-distressed { background:rgba(232,68,90,0.15); color:#e8445a !important; border:1px solid rgba(232,68,90,0.3); }
.emotion-angry { background:rgba(200,30,30,0.2); color:#ff4444 !important; border:1px solid rgba(200,30,30,0.35); }
.emotion-fear { background:rgba(138,43,226,0.15); color:#9b59b6 !important; border:1px solid rgba(138,43,226,0.25); }
.dot-blink::before { content:''; width:6px; height:6px; border-radius:50%; background:currentColor; display:inline-block; animation:blink 1s infinite; }
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.3} }
@keyframes pulse-ring { 0%{transform:scale(1);opacity:0.6} 100%{transform:scale(1.4);opacity:0} }

/* Buttons */
.stButton > button { width:100% !important; border-radius:10px !important; font-weight:700 !important; font-family:'Syne' !important; border:1px solid var(--border-medium) !important; background:var(--bg-surface) !important; color:var(--text-secondary) !important; transition:all 0.2s ease; }
.stButton > button:hover { background:var(--bg-surface-2) !important; border-color:var(--accent-blue) !important; color:var(--text-primary) !important; }
.stButton > button[kind="primary"] { background:var(--accent-red) !important; color:white !important; border-color:var(--accent-red) !important; }
.stButton > button[kind="primary"]:hover { background:#c7374b !important; }

/* Inputs & Expanders */
[data-testid="stAudioInput"] > div { background:var(--bg-surface) !important; border:1px solid var(--border-subtle) !important; border-radius:10px !important; }
[data-testid="stTextArea"] textarea { background:var(--bg-surface) !important; color:var(--text-secondary) !important; border:1px solid var(--border-subtle) !important; border-radius:10px !important; font-family:'Syne' !important; }
[data-testid="stExpander"] { background:var(--bg-surface) !important; border:1px solid var(--border-subtle) !important; border-radius:10px !important; }
[data-testid="stExpander"] summary span { color:var(--text-secondary) !important; }
[data-testid="stExpander"] .stCodeBlock code { background:var(--bg-base) !important; color:var(--text-secondary) !important; }

/* Alerts */
.stAlert { border-radius:10px !important; }
div[data-testid="stCaptionContainer"] p { color:var(--text-muted) !important; }

/* Toast */
[data-testid="stToast"] { background:var(--bg-surface) !important; color:var(--text-primary) !important; border:1px solid var(--border-subtle) !important; }

/* Dataframe */
[data-testid="stDataFrame"] { background:var(--bg-surface) !important; border:1px solid var(--border-subtle) !important; border-radius:10px !important; }

/* Spinner */
.stSpinner > div { color:var(--text-muted) !important; }
</style>""", unsafe_allow_html=True)

init_global_styles()

# ---------------------------------------------------------------------------
# UI Helpers
# ---------------------------------------------------------------------------
SENTIMENT_MAP = {
    "calm": ("emotion-calm", "🟢", "Calm"),
    "confused": ("emotion-confused", "🟡", "Confused"),
    "urgent": ("emotion-urgent", "🟠", "Urgent"),
    "distressed": ("emotion-distressed", "🔴", "Distressed"),
    "angry": ("emotion-angry", "🟥", "Angry"),
    "fear": ("emotion-fear", "😨", "Fear"),
}

def _sentiment_pill(sentiment):
    cls, icon, label = SENTIMENT_MAP.get(sentiment, SENTIMENT_MAP["calm"])
    blink = " dot-blink" if sentiment in ("distressed", "angry") else ""
    return f"<span class='emotion-pill {cls}{blink}'>{icon} {label}</span>"

def _conf_pill(conf):
    return f"<span class='pill conf-pill'>{conf * 100:.0f}%</span>"

def _lang_pill(lang):
    return f"<span class='pill lang-pill'>{lang.upper()}</span>"

def _autoplay_audio(file_path: str):
    """Inject base64 audio with autoplay attribute — works in Streamlit Cloud."""
    import base64
    import os
    if not file_path or not os.path.isfile(file_path) or os.path.getsize(file_path) == 0:
        st.warning("⚠️ Voice playback unavailable — read the AI summary above.")
        return
    with open(file_path, "rb") as f:
        audio_b64 = base64.b64encode(f.read()).decode()
    st.markdown(
        f'<audio autoplay controls style="width:100%;margin-top:8px;border-radius:8px;">'
        f'<source src="data:audio/mp3;base64,{audio_b64}" type="audio/mp3">'
        f'</audio>',
        unsafe_allow_html=True,
    )

def _render_header():
    st.markdown('''<div class="nav-bar"><div class="nav-brand-group"><div style="display:flex;align-items:center;gap:12px;"><div class="nav-icon-bg">📞</div><h1 style="margin:0;font-family:'Syne',sans-serif;font-weight:800;font-size:18px;color:var(--text-primary);line-height:1.2;">Namma Vanni — <span style="color:var(--accent-red)">1092</span> AI Helpline</h1></div><p style="margin:6px 0 0 50px;font-size:12px;color:var(--text-secondary);letter-spacing:0.3px;">Voice-to-voice citizen assistant for Karnataka</p></div><div class="nav-badge">LIVE</div></div>''', unsafe_allow_html=True)

def parse_smart_confirmation(transcript: str) -> dict:
    """Handles conversational Yes/No/Partial (e.g., 'Yes but the location is wrong')."""
    t = transcript.strip().lower()

    pos_cluster = ["yes", "yeah", "yep", "correct", "right", "okay", "ok", "haan", "hān", "sari", "sha"]
    neg_cluster = ["no", "nahi", "galat", "wrong", "naahi", "illa", "thappilla", "kadliya"]
    partial_cluster = ["but", "almost", "mostly", "partially", "partly", "not fully",
                       "not exactly", "haan par", "haan lekin", "sari aadre", "yes but",
                       "thoda", "kuch", "aadre"]

    pos_hits = [w for w in pos_cluster if w in t]
    neg_hits = [w for w in neg_cluster if w in t]
    partial_hits = [w for w in partial_cluster if w in t]

    # Check partial FIRST ("yes but..." should be partial, not confirmed)
    if len(partial_hits) > 0 and len(pos_hits) > 0:
        return {"intent": "partial"}

    if len(pos_hits) >= 1 and len(neg_hits) == 0: return {"intent": "confirmed"}
    if len(neg_hits) >= 1 and len(pos_hits) == 0: return {"intent": "denied"}

    positive_phrases = ["sounds right", "did i understand correctly", "that is correct", "you got it", "exactly", "bilkul sahi"]
    negative_phrases = ["not exactly", "wrong", "missed", "try again", "not what i said"]

    if any(p in t for p in positive_phrases): return {"intent": "confirmed"}
    if any(p in t for p in negative_phrases): return {"intent": "denied"}

    return {"intent": "unclear"}

# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------
_DEFAULTS = {"stage": "input_record", "ai_data": None, "attempts": 0}
for key, val in _DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = val

def _reset():
    """Reset session state to initial values."""
    for key, val in _DEFAULTS.items():
        st.session_state[key] = val

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
_render_header()
if MOCK_MODE:
    st.caption("⚙️ Running in **MOCK MODE** — no API calls.")

# ═══════════════════════════════════════════════════════════════════════════
# STAGE: input_record
# ═══════════════════════════════════════════════════════════════════════════
if st.session_state.stage == "input_record":
    # Pulsing mic hero
    st.markdown('''<div style="display:flex;flex-direction:column;align-items:center;margin:40px 0 20px 0;">
        <div style="position:relative;width:150px;height:150px;display:flex;justify-content:center;align-items:center;">
            <div style="position:absolute;width:120px;height:120px;border-radius:50%;background:rgba(232,68,90,0.08);border:2px solid rgba(232,68,90,0.2);animation:pulse-ring 2.2s infinite;"></div>
            <div style="position:absolute;width:150px;height:150px;border-radius:50%;border:1px solid rgba(232,68,90,0.08);animation:pulse-ring 2.2s infinite 0.4s;"></div>
            <div style="width:64px;height:64px;border-radius:50%;background:var(--accent-red);display:flex;align-items:center;justify-content:center;font-size:24px;z-index:2;box-shadow:0 0 30px rgba(232,68,90,0.3);">🎤</div>
        </div>
        <h2 style="margin-top:24px;font-weight:800;">Start Emergency Call</h2>
        <p style="color:var(--text-muted) !important;font-size:14px;margin-top:4px;">Speak clearly. The AI will triage your issue in real-time.</p>
    </div>''', unsafe_allow_html=True)

    audio_bytes = st.audio_input("Speak your concern or upload audio")

    # Stats grid
    st.markdown('''<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-top:24px;">
        <div class="card" style="margin:0;text-align:center;"><span class="section-label">Active Calls</span><br><span style="font-size:20px;font-weight:700;color:var(--accent-red) !important;">14</span></div>
        <div class="card" style="margin:0;text-align:center;"><span class="section-label">Resolved Today</span><br><span style="font-size:20px;font-weight:700;color:var(--accent-green) !important;">238</span></div>
        <div class="card" style="margin:0;text-align:center;"><span class="section-label">Avg Response</span><br><span style="font-size:20px;font-weight:700;">1m 12s</span></div>
    </div>''', unsafe_allow_html=True)

    if audio_bytes is not None:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes.getvalue())
            wav_path = tmp.name

        with st.spinner("🔄 Processing your voice — transcribing & analysing…"):
            result = process_audio(wav_path)

        os.unlink(wav_path)

        st.session_state.ai_data = result
        st.session_state.attempts = 0
        st.session_state.stage = "verify"
        st.rerun()

# ═══════════════════════════════════════════════════════════════════════════
# STAGE: verify
# ═══════════════════════════════════════════════════════════════════════════
elif st.session_state.stage == "verify":
    data = st.session_state.ai_data or {}
    if not data:
        st.error("No analysis data found. Returning to start.")
        _reset()
        st.rerun()

    lang = data.get("language", "en")
    tts_path = data.get("verify_tts_path", "")
    sentiment = data.get("sentiment", "calm")
    conf = data.get("confidence", 0)

    # --- Analysis pills ---
    st.markdown(f'''<div class="card"><span class="section-label">Caller Analysis</span>
        <div style="display:flex;gap:8px;margin-top:4px;flex-wrap:wrap;">
            {_lang_pill(lang)} {_conf_pill(conf)} {_sentiment_pill(sentiment)}
        </div>
    </div>''', unsafe_allow_html=True)

    # --- AI Summary ---
    st.markdown(f'''<div class="card"><span class="section-label">AI Summary</span>
        <h3 style="margin:4px 0 0 0;font-weight:600;">{data.get("normalized_issue", "—")}</h3>
    </div>''', unsafe_allow_html=True)

    # --- Voice Prompt ---
    base_prompt = data.get("verification_prompt", "")
    translated_prompt = data.get("verification_prompt_translated", "")
    normalized_issue = data.get("normalized_issue", "")
    appended_prompt = f"{base_prompt} I heard you say '{normalized_issue}'. Is this correct? Say Yes or No."

    st.markdown(f'''<div class="card" style="display:flex;gap:12px;align-items:start;">
        <div style="width:32px;height:32px;min-width:32px;background:rgba(96,108,200,0.2);border-radius:8px;display:flex;align-items:center;justify-content:center;">🤖</div>
        <div><span class="section-label">AI Voice Prompt (English)</span><p style="margin:0;font-style:italic;color:var(--text-secondary) !important;line-height:1.5;">"{appended_prompt}"</p></div>
    </div>''', unsafe_allow_html=True)

    if translated_prompt:
        lang_label = {"kn": "ಕನ್ನಡ", "hi": "हिन्दी"}.get(lang, lang.upper())
        st.markdown(f'''<div class="card" style="display:flex;gap:12px;align-items:start;border-left:3px solid var(--accent-blue);">
            <div style="width:32px;height:32px;min-width:32px;background:rgba(59,138,245,0.2);border-radius:8px;display:flex;align-items:center;justify-content:center;">🗣️</div>
            <div><span class="section-label">Voice Playback ({lang_label})</span><p style="margin:0;font-style:italic;color:var(--text-primary) !important;line-height:1.5;">"{translated_prompt}"</p></div>
        </div>''', unsafe_allow_html=True)

    # --- PLAYBACK (no key= args to prevent TypeError) ---
    st.markdown(
        '<p style="font-size:12px;color:var(--text-muted) !important;margin-bottom:4px;">🔊 AI is speaking — listen and then respond below</p>',
        unsafe_allow_html=True,
    )
    _autoplay_audio(tts_path)

    # --- RECORDING ---
    conf_audio = st.audio_input("🎙️ Say 'Yes' or 'No'...", key="final_mic_ver")
    if conf_audio is not None:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(conf_audio.getvalue())
            conf_wav_path = tmp.name

        with st.spinner("🤔 Understanding your response…"):
            raw_text, _, _ = transcribe_audio(conf_wav_path)

            # ── LAYER 1: LLM (primary — handles all languages, dialects, free speech) ──
            result = {"intent": "unclear", "summary": ""}
            try:
                llm_result = parse_confirmation(raw_text)
                if llm_result.get("intent") in ("confirmed", "denied", "partial"):
                    result = llm_result  # LLM gave a clear answer — use it
            except Exception:
                pass  # LLM failed — fall through to keywords

            # ── LAYER 2: Keyword matching (fallback — works offline, covers KN/HI/EN) ──
            if result["intent"] == "unclear":
                result = parse_smart_confirmation(raw_text)

            st.session_state.attempts += 1

        os.unlink(conf_wav_path)

        # ── LAYER 3: Route based on final intent ──
        if result["intent"] == "confirmed":
            st.session_state.stage = "agent_ready"
            st.rerun()

        elif result["intent"] == "partial":
            st.session_state.stage = "partial_refine"
            st.rerun()

        elif result["intent"] == "denied":
            if st.session_state.attempts >= 2 or data.get("handover", False):
                st.session_state.stage = "handover"
            else:
                st.session_state.stage = "re_record"
            st.rerun()

        else:
            # Both LLM and keywords failed — show friendly retry UI, NO st.rerun()
            summary = result.get("summary", "") or raw_text[:60]
            st.markdown(
                f'''<div class="card" style="border-left:3px solid var(--accent-yellow);">
                  <div style="display:flex;gap:10px;align-items:start;">
                    <div style="width:28px;height:28px;min-width:28px;background:rgba(212,160,23,0.2);
                         border-radius:8px;display:flex;align-items:center;justify-content:center;">🤖</div>
                    <div>
                      <span class="section-label">AI Heard</span>
                      <p style="color:var(--text-secondary) !important;margin:0 0 8px 0;">"{summary}"</p>
                      <p style="font-size:12px;color:var(--text-muted) !important;margin:0;">
                        Please respond with one of these:
                      </p>
                      <div style="display:flex;gap:8px;margin-top:8px;flex-wrap:wrap;">
                        <span class="pill" style="background:rgba(61,214,140,0.15);color:#3dd68c !important;border:1px solid rgba(61,214,140,0.25);">✅ Yes / ಹೌದು / हाँ</span>
                        <span class="pill" style="background:rgba(232,68,90,0.15);color:#e8445a !important;border:1px solid rgba(232,68,90,0.3);">❌ No / ಇಲ್ಲ / नहीं</span>
                        <span class="pill" style="background:rgba(212,160,23,0.15);color:#d4a017 !important;border:1px solid rgba(212,160,23,0.25);">🔄 Yes but... / ಆದರೆ / लेकिन</span>
                      </div>
                    </div>
                  </div>
                </div>''',
                unsafe_allow_html=True,
            )
            # NO st.rerun() — user sees the pills and re-records naturally

# ═══════════════════════════════════════════════════════════════════════════
# STAGE: partial_refine — citizen said "partially correct"
# ═══════════════════════════════════════════════════════════════════════════
elif st.session_state.stage == "partial_refine":
    data = st.session_state.ai_data or {}
    if not data:
        _reset()
        st.rerun()

    # Show what AI understood
    st.markdown('''<div style="background:rgba(212,160,23,0.08);border:1px solid rgba(212,160,23,0.2);border-radius:12px;padding:14px 18px;margin-bottom:20px;display:flex;align-items:center;gap:8px;">
        <span style="color:var(--accent-yellow);">🔄</span>
        <span style="font-size:14px;font-weight:700;color:var(--accent-yellow) !important;">Partially Understood — Please Clarify</span>
    </div>''', unsafe_allow_html=True)

    st.markdown(f'''<div class="card" style="border-left:3px solid var(--accent-yellow);">
        <span class="section-label">AI's Current Understanding</span>
        <h3 style="font-weight:600;margin:4px 0 0 0;">{data.get('normalized_issue', '—')}</h3>
    </div>''', unsafe_allow_html=True)

    st.markdown('''<div class="card">
        <span class="section-label">What to do</span>
        <p style="color:var(--text-secondary) !important;margin:4px 0 0 0;">Please record what was different or incorrect. The AI will refine its understanding.</p>
    </div>''', unsafe_allow_html=True)

    correction_audio = st.audio_input("🎙️ Record your correction...", key="partial_mic")
    if correction_audio is not None:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(correction_audio.getvalue())
            corr_wav_path = tmp.name

        with st.spinner("🔄 Refining AI analysis with your correction..."):
            correction_text, _, _ = transcribe_audio(corr_wav_path)
            os.unlink(corr_wav_path)
            if correction_text.strip():
                refined = re_analyze_transcript(correction_text, data, feedback_type="partial")
                tts_path = generate_tts(refined.get("verification_prompt", ""), refined.get("language", "kn"))
                refined["raw_text"] = data.get("raw_text", "") + " [CORRECTION] " + correction_text
                refined["verify_tts_path"] = tts_path
                st.session_state.ai_data = refined
                st.session_state.stage = "verify"
                st.rerun()
            else:
                st.warning("Couldn't hear the correction. Please try again.")

# ═══════════════════════════════════════════════════════════════════════════
# STAGE: re_record — citizen denied, re-record with context preserved
# ═══════════════════════════════════════════════════════════════════════════
elif st.session_state.stage == "re_record":
    data = st.session_state.ai_data or {}
    if not data:
        _reset()
        st.rerun()

    # Show what went wrong
    st.markdown('''<div style="background:rgba(232,68,90,0.08);border:1px solid rgba(232,68,90,0.2);border-radius:12px;padding:14px 18px;margin-bottom:20px;display:flex;align-items:center;gap:8px;">
        <span style="color:var(--accent-red);">❌</span>
        <span style="font-size:14px;font-weight:700;color:var(--accent-red) !important;">Not Correct — Please Re-explain</span>
    </div>''', unsafe_allow_html=True)

    st.markdown(f'''<div class="card" style="border-left:3px solid var(--accent-red);">
        <span class="section-label">AI's Wrong Understanding</span>
        <h3 style="font-weight:600;margin:4px 0 0 0;text-decoration:line-through;color:var(--text-muted) !important;">{data.get('normalized_issue', '—')}</h3>
    </div>''', unsafe_allow_html=True)

    st.markdown('''<div class="card">
        <span class="section-label">What to do</span>
        <p style="color:var(--text-secondary) !important;margin:4px 0 0 0;">Please re-explain your issue. The AI will try again with more context.</p>
    </div>''', unsafe_allow_html=True)

    retry_audio = st.audio_input("🎙️ Re-explain your issue...", key="retry_mic")
    if retry_audio is not None:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(retry_audio.getvalue())
            retry_wav_path = tmp.name

        with st.spinner("🔄 Re-analyzing with your feedback..."):
            retry_text, _, _ = transcribe_audio(retry_wav_path)
            os.unlink(retry_wav_path)
            if retry_text.strip():
                refined = re_analyze_transcript(retry_text, data, feedback_type="denied")
                tts_path = generate_tts(refined.get("verification_prompt", ""), refined.get("language", "kn"))
                refined["raw_text"] = retry_text
                refined["verify_tts_path"] = tts_path
                st.session_state.ai_data = refined
                st.session_state.stage = "verify"
                st.rerun()
            else:
                st.warning("Couldn't hear you. Please try again.")

# ═══════════════════════════════════════════════════════════════════════════
# STAGE: decision (transient — routes immediately)
# ═══════════════════════════════════════════════════════════════════════════
elif st.session_state.stage == "decision":
    st.session_state.attempts += 1
    data = st.session_state.ai_data or {}

    if st.session_state.attempts >= 2 or data.get("handover", False):
        st.session_state.stage = "handover"
    else:
        st.warning("⚠️ Not understood. Please try again.")
        st.session_state.stage = "input_record"
    st.rerun()

# ═══════════════════════════════════════════════════════════════════════════
# STAGE: agent_ready
# ═══════════════════════════════════════════════════════════════════════════
elif st.session_state.stage == "agent_ready":
    data = st.session_state.ai_data or {}
    sentiment = data.get("sentiment", "calm")
    conf = data.get("confidence", 0)
    lang = data.get("language", "—")
    issue = data.get('normalized_issue', '—')

    # Success banner
    st.markdown('''<div style="background:rgba(61,214,140,0.08);border:1px solid rgba(61,214,140,0.2);border-radius:12px;padding:14px 18px;margin-bottom:20px;display:flex;align-items:center;gap:8px;">
        <span style="color:var(--accent-green);">✅</span>
        <span style="font-size:14px;font-weight:700;color:var(--accent-green) !important;">Issue Verified — Agent Dashboard Active</span>
    </div>''', unsafe_allow_html=True)

    # Metadata + Issue grid
    st.markdown(f'''<div style="display:grid;grid-template-columns:220px 1fr;gap:12px;margin-bottom:20px;">
        <div class="card"><span class="section-label">Metadata</span>
            <div style="margin-bottom:8px;">{_sentiment_pill(sentiment)}</div>
            <div style="margin-bottom:8px;">{_conf_pill(conf)}</div>
            <div>{_lang_pill(lang)}</div>
        </div>
        <div class="card" style="border-left:3px solid var(--accent-green);">
            <span class="section-label">Verified Issue</span>
            <h2 style="font-weight:800;margin:4px 0 0 0;font-size:20px;">{issue}</h2>
        </div>
    </div>''', unsafe_allow_html=True)

    raw_transcript = data.get("raw_text", "")
    original_transcript = data.get("original_text", "")
    lang = data.get("language", "en")
    lang_label = {"kn": "ಕನ್ನಡ (Kannada)", "hi": "हिन्दी (Hindi)"}.get(lang, lang.upper())

    if original_transcript:
        # Side-by-side: original language + English
        st.markdown(f'''<div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:16px;">
            <div style="background:var(--bg-surface);border:1px solid var(--border-subtle);border-left:3px solid var(--accent-blue);border-radius:12px;padding:20px;">
                <span class="section-label">🗣️ Original ({lang_label})</span>
                <p style="font-family:'Space Mono',monospace;font-size:13px;line-height:1.6;color:var(--text-secondary) !important;margin-top:8px;white-space:pre-wrap;word-break:break-word;">
                    {original_transcript}
                </p>
            </div>
            <div style="background:var(--bg-surface);border:1px solid var(--border-subtle);border-left:3px solid var(--accent-green);border-radius:12px;padding:20px;">
                <span class="section-label">📄 English Translation</span>
                <p style="font-family:'Space Mono',monospace;font-size:13px;line-height:1.6;color:var(--text-secondary) !important;margin-top:8px;white-space:pre-wrap;word-break:break-word;">
                    {raw_transcript}
                </p>
            </div>
        </div>''', unsafe_allow_html=True)
    else:
        # Fallback: single transcript (English only)
        st.markdown(f'''<div style="background:var(--bg-surface);border:1px solid var(--border-subtle);border-radius:12px;padding:20px;margin-top:16px;">
            <span class="section-label">📄 Raw Transcript</span>
            <p style="font-family:'Space Mono',monospace;font-size:14px;line-height:1.5;color:var(--text-secondary) !important;margin-top:8px;white-space:pre-wrap;word-break:break-word;">
                {raw_transcript}
            </p>
        </div>''', unsafe_allow_html=True)

    agent_note = st.text_area(
        "Agent Notes / Corrections",
        value=data.get("normalized_issue", ""),
        help="Edit if the AI summary needs correction.",
    )

    if st.button("✅ Log & Close Call", width="stretch"):
        feedback = dict(data)
        feedback["citizen_response"] = "Confirmed"
        feedback["agent_correction"] = agent_note if agent_note != data.get("normalized_issue", "") else ""
        log_feedback(feedback)
        st.toast("📋 Feedback logged!", icon="✅")
        _reset()
        st.rerun()

    # Feedback log
    st.markdown('<div style="margin-top:24px;"><span class="section-label" style="font-size:13px;">📊 Feedback Log</span></div>', unsafe_allow_html=True)
    if os.path.isfile(FEEDBACK_FILE):
        try:
            df = pd.read_csv(FEEDBACK_FILE)
            st.dataframe(df, use_container_width=True)
        except Exception:
            st.caption("Feedback file is empty or unreadable.")
    else:
        st.caption("No feedback recorded yet.")

# ═══════════════════════════════════════════════════════════════════════════
# STAGE: handover
# ═══════════════════════════════════════════════════════════════════════════
elif st.session_state.stage == "handover":
    data = st.session_state.ai_data or {}
    sentiment = data.get("sentiment", "calm")
    conf = data.get("confidence", 0)
    issue = data.get('normalized_issue', '—')

    # Red alert banner
    st.markdown('''<div style="background:rgba(232,68,90,0.08);border:1px solid rgba(232,68,90,0.25);border-radius:12px;padding:18px 20px;margin-bottom:20px;display:flex;gap:16px;align-items:start;">
        <div style="width:40px;height:40px;min-width:40px;background:rgba(232,68,90,0.15);border-radius:10px;display:flex;align-items:center;justify-content:center;color:var(--accent-red);font-size:18px;">⚠️</div>
        <div>
            <span style="font-size:16px;font-weight:800;color:var(--accent-red) !important;display:block;margin-bottom:4px;">HUMAN AGENT TAKING OVER</span>
            <span style="font-size:13px;color:rgba(232,68,90,0.7) !important;">AI could not confidently verify the caller's issue. A human agent will now handle this call directly.</span>
        </div>
    </div>''', unsafe_allow_html=True)

    # Metadata + Summary grid
    st.markdown(f'''<div style="display:grid;grid-template-columns:220px 1fr;gap:12px;margin-bottom:16px;">
        <div class="card"><span class="section-label">Metadata</span>
            <div style="margin-bottom:8px;">{_sentiment_pill(sentiment)}</div>
            <div>{_conf_pill(conf)}</div>
        </div>
        <div class="card" style="border-left:3px solid var(--accent-red);">
            <span class="section-label">Last AI Summary</span>
            <h2 style="font-weight:700;margin:4px 0 0 0;font-size:20px;">{issue}</h2>
        </div>
    </div>''', unsafe_allow_html=True)

    raw_text = data.get("raw_text", "No transcript available.")
    st.markdown(f'''<div style="background: var(--bg-surface); border: 1px solid var(--border-subtle); border-radius: 12px; padding: 20px; margin-bottom: 16px;">
        <span class="section-label">📄 Raw Transcript</span>
        <p style="font-family: 'Space Mono', monospace; font-size: 13px; line-height: 1.6; color: var(--text-secondary) !important; margin-top: 12px; white-space: pre-wrap; word-break: break-word; max-height: 150px; overflow-y: auto;">
            {raw_text}
        </p>
    </div>''', unsafe_allow_html=True)

    if st.button("🔄 End Call", width="stretch"):
        feedback = dict(data)
        feedback["citizen_response"] = "Handover"
        feedback["agent_correction"] = ""
        log_feedback(feedback)
        st.toast("📋 Call logged as handover.", icon="🔄")
        _reset()
        st.rerun()
