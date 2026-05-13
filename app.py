import streamlit as st
import pandas as pd
import joblib
import re
import time
import json
from openai import OpenAI
from dotenv import load_dotenv
import base64
import os

# ─── PAGE CONFIG ────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Car Advisor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed",
)

load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if OPENROUTER_API_KEY:
    try:
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=OPENROUTER_API_KEY,
        )
        chat_available = True
    except Exception as e:
        chat_available = False
        client = None
else:
    chat_available = False
    client = None

# ─── GLOBAL CSS ─────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── Reset & Base ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"] {
    background: #080C14 !important;
    font-family: 'DM Sans', sans-serif;
    color: #E2E8F0;
}

/* Hide default Streamlit chrome */
#MainMenu, footer, header,
[data-testid="collapsedControl"],
[data-testid="stSidebarNav"] { display: none !important; }

/* Remove default padding */
[data-testid="stAppViewContainer"] > .main { padding: 0 !important; }
.block-container { padding: 2.5rem !important; max-width: 100% !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0F1520; }
::-webkit-scrollbar-thumb { background: #2D3748; border-radius: 3px; }

/* ── Top Nav Bar ── */
.nav-bar {
    position: sticky; top: 0; z-index: 999;
    background: rgba(8, 12, 20, 0.92);
    backdrop-filter: blur(20px);
    border-bottom: 1px solid rgba(255,255,255,0.06);
    padding: 0;
    margin-bottom: 1.5rem;
    height: 60px;
    display: flex; align-items: center; justify-content: space-between;
}
.nav-logo {
    font-family: 'Syne', sans-serif;
    font-weight: 800; font-size: 1.2rem;
    color: #FFFFFF; letter-spacing: -0.5px;
}
.nav-logo span { color: #FF4B4B; }
.nav-tabs { display: flex; gap: 0.25rem; }
.nav-tab {
    padding: 0.45rem 1.1rem;
    border-radius: 8px;
    font-size: 0.85rem; font-weight: 500;
    cursor: pointer; transition: all 0.2s;
    color: #718096; background: transparent; border: none;
    font-family: 'DM Sans', sans-serif;
}
.nav-tab.active {
    background: rgba(255,75,75,0.15);
    color: #FF4B4B;
    border: 1px solid rgba(255,75,75,0.3);
}
.nav-tab:hover:not(.active) { color: #E2E8F0; background: rgba(255,255,255,0.05); }

/* ── Page Wrapper ── */
.page-wrapper { max-width: 1200px; margin: 0 auto; padding: 0; }

/* ── Hero ── */
.hero-label {
    font-size: 0.7rem; font-weight: 600; letter-spacing: 3px;
    color: #FF4B4B; text-transform: uppercase; margin-bottom: 0.75rem;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: clamp(2rem, 4vw, 3rem);
    font-weight: 800; line-height: 1.15;
    color: #FFFFFF; margin-bottom: 0.75rem;
}
.hero-title em { color: #FF4B4B; font-style: italic; }
.hero-sub {
    font-size: 0.95rem; color: #718096;
    max-width: 520px; line-height: 1.6; margin-bottom: 2rem;
}

/* ── Form Card ── */
.form-card {
    background: #0F1520;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px;
    padding: 2rem;
    margin-bottom: 1.5rem;
}
.form-section-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.75rem; font-weight: 700;
    letter-spacing: 2px; text-transform: uppercase;
    color: #4A5568; margin-bottom: 1.25rem;
}

/* ── Streamlit widget overrides ── */
[data-testid="stSelectbox"] label,
[data-testid="stSlider"] label,
[data-testid="stNumberInput"] label,
[data-testid="stTextArea"] label {
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    color: #718096 !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
    font-family: 'DM Sans', sans-serif !important;
}

[data-testid="stSelectbox"] > div > div,
[data-testid="stNumberInput"] input,
[data-testid="stTextArea"] textarea {
    background: #141B27 !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 10px !important;
    color: #E2E8F0 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.9rem !important;
}

[data-testid="stSelectbox"] > div > div:focus-within,
[data-testid="stNumberInput"] input:focus,
[data-testid="stTextArea"] textarea:focus {
    border-color: rgba(255,75,75,0.5) !important;
    box-shadow: 0 0 0 3px rgba(255,75,75,0.1) !important;
}

/* Slider track */
[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] {
    background: #FF4B4B !important;
}
[data-testid="stSlider"] [data-baseweb="slider"] div[style*="background"] {
    background: #FF4B4B !important;
}

/* ── CTA Button ── */
[data-testid="stButton"] > button[kind="primary"] {
    background: linear-gradient(135deg, #FF4B4B, #E53E3E) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.85rem 2.5rem !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    letter-spacing: 0.5px !important;
    width: 100% !important;
    height: 3.2rem !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 20px rgba(255,75,75,0.3) !important;
}
[data-testid="stButton"] > button[kind="primary"]:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 30px rgba(255,75,75,0.45) !important;
}

/* Secondary buttons */
[data-testid="stButton"] > button:not([kind="primary"]) {
    background: rgba(255,255,255,0.04) !important;
    color: #A0AEC0 !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.82rem !important;
    transition: all 0.2s !important;
}
[data-testid="stButton"] > button:not([kind="primary"]):hover {
    background: rgba(255,255,255,0.08) !important;
    color: #E2E8F0 !important;
    border-color: rgba(255,255,255,0.15) !important;
}

/* ── Result Cards ── */
.result-section { animation: fadeInUp 0.4s ease both; }
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(16px); }
    to   { opacity: 1; transform: translateY(0); }
}

.price-hero-card {
    background: linear-gradient(135deg, #0F1520 0%, #141B2D 100%);
    border: 1px solid rgba(0, 212, 255, 0.2);
    border-radius: 16px;
    padding: 2rem 2rem 1.5rem;
    position: relative;
    overflow: hidden;
}
.price-hero-card::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, #00D4FF, transparent);
}
.price-label {
    font-size: 0.68rem; font-weight: 700;
    letter-spacing: 2.5px; text-transform: uppercase;
    color: #00D4FF; margin-bottom: 0.5rem;
}
.price-value {
    font-family: 'Syne', sans-serif;
    font-size: clamp(2rem, 4vw, 2.8rem);
    font-weight: 800; color: #FFFFFF;
    line-height: 1; margin-bottom: 0.4rem;
}
.price-standard {
    font-size: 0.82rem; color: #4A5568;
}
.price-standard span { color: #718096; }
.price-badge {
    display: inline-flex; align-items: center; gap: 0.4rem;
    padding: 0.3rem 0.85rem;
    border-radius: 20px;
    font-size: 0.78rem; font-weight: 600;
    margin-top: 0.75rem;
}
.price-badge.premium {
    background: rgba(72,187,120,0.15);
    border: 1px solid rgba(72,187,120,0.3);
    color: #48BB78;
}
.price-badge.discount {
    background: rgba(255,75,75,0.12);
    border: 1px solid rgba(255,75,75,0.25);
    color: #FF4B4B;
}
.price-badge.neutral {
    background: rgba(160,174,192,0.1);
    border: 1px solid rgba(160,174,192,0.2);
    color: #A0AEC0;
}

/* ── Score Card ── */
.score-card {
    background: #0F1520;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
    height: 100%;
}
.score-number {
    font-family: 'Syne', sans-serif;
    font-size: 3.5rem; font-weight: 800;
    line-height: 1;
}
.score-label {
    font-size: 0.72rem; font-weight: 600;
    letter-spacing: 2px; text-transform: uppercase;
    color: #4A5568; margin-top: 0.3rem; margin-bottom: 1rem;
}
.condition-badge {
    display: inline-block;
    padding: 0.4rem 1rem;
    border-radius: 20px;
    font-size: 0.82rem; font-weight: 600;
}
.badge-excellent { background: rgba(72,187,120,0.15); border: 1px solid rgba(72,187,120,0.35); color: #48BB78; }
.badge-good      { background: rgba(0,212,255,0.12);  border: 1px solid rgba(0,212,255,0.3);  color: #00D4FF; }
.badge-fair      { background: rgba(246,224,94,0.12); border: 1px solid rgba(246,224,94,0.3); color: #F6E05E; }
.badge-poor      { background: rgba(255,75,75,0.12);  border: 1px solid rgba(255,75,75,0.25); color: #FF4B4B; }

/* ── Info Cards (Trend / Demand) ── */
.info-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 0.75rem; margin-top: 1rem; }
.info-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 10px;
    padding: 0.85rem 1rem;
}
.info-card-label { font-size: 0.68rem; color: #4A5568; font-weight: 600; letter-spacing: 1px; text-transform: uppercase; margin-bottom: 0.25rem; }
.info-card-value { font-size: 0.95rem; font-weight: 600; color: #E2E8F0; }

/* ── Upgrade Suggestion Cards ── */
.upgrade-section-title {
    font-family: 'Syne', sans-serif;
    font-size: 1rem; font-weight: 700;
    color: #FFFFFF; margin-bottom: 1rem;
    display: flex; align-items: center; gap: 0.5rem;
}
.upgrade-card {
    background: #0F1520;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.6rem;
    display: flex; align-items: center; justify-content: space-between;
    transition: border-color 0.2s;
}
.upgrade-card:hover { border-color: rgba(255,75,75,0.25); }
.upgrade-info h4 { font-size: 0.92rem; font-weight: 600; color: #E2E8F0; margin-bottom: 0.2rem; }
.upgrade-info p  { font-size: 0.78rem; color: #4A5568; }
.lift-badge {
    background: rgba(72,187,120,0.12);
    border: 1px solid rgba(72,187,120,0.25);
    color: #48BB78;
    border-radius: 8px;
    padding: 0.25rem 0.6rem;
    font-size: 0.72rem; font-weight: 700;
    white-space: nowrap;
}
.lift-label { font-size: 0.55rem; display: block; color: #2F855A; text-transform: uppercase; letter-spacing: 1px; }

/* ── Divider ── */
.section-divider {
    border: none;
    border-top: 1px solid rgba(255,255,255,0.06);
    margin: 2rem 0;
}

/* ── Chat Screen ── */
.chat-hero { margin-bottom: 1.5rem; }
.chat-hero h1 {
    font-family: 'Syne', sans-serif;
    font-size: 1.8rem; font-weight: 800; color: #FFFFFF;
    margin-bottom: 0.3rem;
}
.chat-hero p { font-size: 0.85rem; color: #4A5568; }

.quick-chips { display: flex; flex-wrap: wrap; gap: 0.5rem; margin-bottom: 1.25rem; }
.quick-chip {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 20px;
    padding: 0.35rem 0.9rem;
    font-size: 0.8rem; color: #A0AEC0; cursor: pointer;
    transition: all 0.15s;
}
.quick-chip:hover { background: rgba(255,75,75,0.1); border-color: rgba(255,75,75,0.3); color: #FF4B4B; }

/* Chat messages */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    padding: 0.25rem 0 !important;
}
[data-testid="stChatMessage"][data-testid*="user"] > div {
    background: rgba(255,75,75,0.08) !important;
    border: 1px solid rgba(255,75,75,0.15) !important;
    border-radius: 12px !important;
}
[data-testid="stChatMessage"]:not([data-testid*="user"]) > div {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: 12px !important;
}

/* Chat input */
[data-testid="stChatInput"] {
    background: #0F1520 !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 12px !important;
}
[data-testid="stChatInput"] textarea {
    background: transparent !important;
    color: #E2E8F0 !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* ── Metric overrides ── */
[data-testid="stMetric"] {
    background: transparent !important;
}
[data-testid="stMetricLabel"] { color: #4A5568 !important; font-size: 0.75rem !important; }
[data-testid="stMetricValue"] { color: #FFFFFF !important; font-family: 'Syne', sans-serif !important; }

/* ── Error / Info / Success ── */
[data-testid="stAlert"] {
    background: rgba(255,255,255,0.03) !important;
    border-radius: 10px !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
}

/* ── Spinner ── */
[data-testid="stSpinner"] { color: #FF4B4B !important; }

/* ── Helper for inline HTML sections ── */
.no-margin-collapse > * { margin: 0 !important; padding: 0 !important; }
</style>
""", unsafe_allow_html=True)


# ─── MODEL LOADING ───────────────────────────────────────────────
@st.cache_resource
def load_models():
    try:
        pm  = joblib.load("models/price_model_optimized.pkl")
        im  = joblib.load("models/inspector_model.pkl")
        vec = joblib.load("models/tfidf_vectorizer.pkl")
        return pm, im, vec
    except Exception:
        return None, None, None

price_model, inspector_model, vectorizer = load_models()


@st.cache_data
def load_car_titles():
    try:
        df = pd.read_csv("./data/MASTER_CAR_DATASET.csv")
        return sorted(df["title_version"].astype(str).unique().tolist())
    except Exception:
        return ["Honda Civic Oriel 1.8 i-VTEC CVT 2020"]

car_options = load_car_titles()


# ─── AI RECOMMENDER ──────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def get_ai_recommendations(car_title, user_desc):
    if not chat_available:
        return []
    try:
        prompt = f"""
        Act as a Pakistani Used Car Dealer Expert. 
        I am selling a: {car_title}
        Current Condition/Description: "{user_desc}"
        DETECTED ISSUES: {', '.join(tags)}
        
        Identify 3 specific, high-return physical upgrades 
        or fixes that would maximize the resale value of 
        this SPECIFIC car in the Pakistani market.
        
        Rules:
        1. Do NOT suggest things already mentioned 
           in the description.
        2. Be specific (e.g., instead of "Sound system", 
           say "Android Panel").
        3. For each item, give a "Lift Score" (1-10) of 
           how much it helps the price.
        4. Return ONLY a valid JSON list like this:
        [
          {{"mod": "Android Panel", 
            "reason": "High demand in Family cars", 
            "lift": 9}},
          {{"mod": "Detailing/Compound", 
            "reason": "Removes scratches", 
            "lift": 7}}
        ]
        5. If TAG_FILE_ISSUE is present, focus suggestions on legal resolution, not cosmetic upgrades.
        """

        response = client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a Pakistani used car market expert. Always respond with valid JSON only. No markdown, no explanation."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.7,
            max_tokens=500,
        )

        text = response.choices[0].message.content
        text = text.replace("```json", "").replace("```", "").strip()
        recommendations = json.loads(text)
        return recommendations

    except Exception as e:
        return [
            {"mod": "Detailing & Polishing", 
             "reason": "Universal value booster", 
             "lift": 5},
            {"mod": "High-Quality Pictures", 
             "reason": "Increases click-through rate", 
             "lift": 8}
        ]


# ── Negation helper ──────────────────────────────────────────────
def _negated(text, match_start, window=30):
    prefix = text[max(0, match_start - window): match_start]
    return bool(re.search(
        r"\b(no|not|never|without|zero|none|dont|don't|nahi|nahin|bilkul)\b",
        prefix
    ))

# ── Count touching pieces ────────────────────────────────────────
def _count_touching_pieces(text):
    """
    Extracts the number of pieces mentioned in touching/touch context.
    Returns integer count or None if not found.
    """
    word_to_num = {
        "one":1,"two":2,"three":3,"four":4,"five":5,
        "six":6,"seven":7,"eight":8,"nine":9,"ten":10,
        "ek":1,"do":2,"teen":3,"char":4,"panch":5
    }
    # Pattern: "4 piece touch" or "four pieces" or "4-5 piece"
    patterns = [
        r"\b(\d+)\s*(?:to|-|or)\s*(\d+)\s*(?:piece|pc|pcs|panel|touch)",
        r"\b(\d+)\s*(?:piece|pc|pcs|panel|touch)",
        r"\b(one|two|three|four|five|six|seven|eight|nine|ten|ek|do|teen|char|panch)"
          r"\s*(?:to|-|or)?\s*(?:one|two|three|four|five|six|seven|eight|nine|ten)?"
          r"\s*(?:piece|pc|pcs|panel|touch)",
    ]
    # Range pattern — take the higher number (worst case)
    m = re.search(
        r"\b(\d+)\s*(?:to|-|or)\s*(\d+)\s*(?:piece|pc|pcs|panel|touch)", text
    )
    if m:
        return max(int(m.group(1)), int(m.group(2)))

    # Single number
    m = re.search(r"\b(\d+)\s*(?:piece|pc|pcs|panel|touch)", text)
    if m:
        return int(m.group(1))

    # Word number
    for word, num in word_to_num.items():
        if re.search(rf"\b{word}\s*(?:piece|pc|pcs|panel|touch)", text):
            return num

    return None


def smart_condition_tagger(text):
    if not isinstance(text, str):
        return ""

    original_lower = text.lower()
    text = original_lower
    tags  = set()
    # severity_notes will store extra context for the penalty engine
    notes = {}

    # ── TIER 1: CATASTROPHIC FILE ISSUES ────────────────────────
    tier1_patterns = [
        r"(file|files)\s*(is\s*)?(miss|lost|dup|fake|stolen|incomplete)",
        r"(missing|duplicate|fake|stolen)\s*(file|ownership)",
        r"\bncp\b",
        r"non[\s-]?custom[\s-]?paid",
        r"open\s*letter",
        r"(registration|reg)\s*(stolen|fake|tampered)",
    ]
    for p in tier1_patterns:
        m = re.search(p, text)
        if m and not _negated(text, m.start()):
            tags.add("TAG_FILE_ISSUE")
            break

    # ── TIER 2: LEGAL / ADMINISTRATIVE ──────────────────────────
    # Duplicate book
    book_patterns = [
        r"(book|logbook|log[\s-]?book)\s*(dup|duplicate|miss|fake|lost)",
        r"(smart\s*card|card)\s*(dup|duplicate|miss|fake|lost)",
        r"duplicate\s*(book|card|smart)",
    ]
    for p in book_patterns:
        m = re.search(p, text)
        if m and not _negated(text, m.start()):
            tags.add("TAG_BOOK_ISSUE")
            break

    # Token unpaid — SEPARATE from file issue, lower penalty
    token_patterns = [
        r"token\s*(unpaid|due|pending|not\s*paid|nahi\s*diya)",
        r"(tax|taxes)\s*(unpaid|due|pending|overdue)",
    ]
    for p in token_patterns:
        m = re.search(p, text)
        if m and not _negated(text, m.start()):
            tags.add("TAG_TOKEN_UNPAID")
            break

    # Court case / on money (on money = registration/tax discrepancy)
    legal_misc = [
        r"court\s*(case|matter|challan)",
        r"\bon[\s-]?money\b",           # on-money = underpaid tax
        r"(registration|reg)\s*(issue|problem|mismatch|city\s*mismatch)",
    ]
    for p in legal_misc:
        m = re.search(p, text)
        if m and not _negated(text, m.start()):
            tags.add("TAG_LEGAL_MISC")
            break

    # ── TIER 3: STRUCTURAL DAMAGE ───────────────────────────────
    structural_patterns = [
        r"(roof|pillar[s]?|a[\s-]?pillar|b[\s-]?pillar|c[\s-]?pillar)"
          r"\s*(paint|painted|damage|shower|dent|bent|replaced|accidental)",
        r"(chassis|frame)\s*(damage|bent|twisted|straightened|repaired)",
        r"total\s*(loss|damage)",
        r"flood[\s-]?(affected|damage|hit)",
        r"fire[\s-]?(damage|affected)",
        r"(major|heavy|severe|serious)\s*(accident|accidental|hit|crash)",
        r"complete\s*(shower|repaint|paint\s*job)\s*(roof|pillar|top)",
        r"(bonnet|boot)\s*(replaced|new|changed)",   # bonnet replaced = front impact
    ]
    for p in structural_patterns:
        m = re.search(p, text)
        if m and not _negated(text, m.start()):
            tags.add("TAG_MAJOR_ACCIDENT")
            break

    # ── TIER 4: MECHANICAL ──────────────────────────────────────
    mech_patterns = [
        (r"(engine|motor)\s*(change|changed|swap|swapped|replace|replaced|overhaul)",
         "engine_replaced"),
        (r"(lifan|chinese|local|3rd[\s-]?party)\s*(engine|motor)",
         "engine_replaced"),
        (r"(engine|motor)\s*(noise|knock|smoke|leak|issue|problem|burning)",
         "engine_issue"),
        (r"(gearbox|gear\s*box|transmission)\s*(change|issue|noise|problem|slip|gone)",
         "gearbox"),
        (r"over\s*heat(ing)?",                  "overheating"),
        (r"white\s*smoke",                       "engine_issue"),
        (r"oil\s*(leak|burning)",                "engine_issue"),
        (r"(timing\s*chain|timing\s*belt)\s*(issue|problem|noise|done|changed)",
         "engine_issue"),
        (r"(ac|air\s*con(dition)?)\s*(not\s*working|faulty|dead|gone|issue)",
         "ac_issue"),
    ]
    for p, note in mech_patterns:
        m = re.search(p, text)
        if m and not _negated(text, m.start()):
            tags.add("TAG_MECHANICAL_BAD")
            notes["mechanical"] = note
            break

    # ── TIER 5 & 6: COSMETIC — THE NUANCED PART ─────────────────
    # Only evaluate cosmetic if no structural damage already
    if "TAG_MAJOR_ACCIDENT" not in tags:

        # Full shower / complete repaint (worst cosmetic)
        full_shower_patterns = [
            r"\bshower(ed|ing)?\b",
            r"full\s*(body\s*)?(paint|painted|respray|repaint)",
            r"complete\s*(repaint|paint\s*job|colour\s*change)",
            r"colour\s*change",
            r"\brepaint(ed|ing)?\b",
            r"fresh\s*(look|paint|color|colour)",
        ]
        is_full_shower = False
        for p in full_shower_patterns:
            m = re.search(p, text)
            if m and not _negated(text, m.start()):
                is_full_shower = True
                break

        # Sides spray (medium cosmetic — 4-6 panels)
        sides_spray_patterns = [
            r"sides?\s*(spray|painted|repaint|done|shower)",
            r"(left|right|both)\s*side[s]?\s*(spray|paint|done)",
            r"panel\s*(beat|beaten|beater)",
            r"dent[s]?\s*(on|at|near)?\s*(door|fender|quarter|side)",
        ]
        is_sides_spray = False
        for p in sides_spray_patterns:
            m = re.search(p, text)
            if m and not _negated(text, m.start()):
                is_sides_spray = True
                break

        # Count pieces for numeric touch mentions
        piece_count = _count_touching_pieces(text)

        # Now determine the correct cosmetic tier
        if is_full_shower:
            tags.add("TAG_COSMETIC_FULL")      # −10%
        elif is_sides_spray:
            tags.add("TAG_COSMETIC_MED")        # −8%
        elif piece_count is not None:
            notes["piece_count"] = piece_count
            if piece_count >= 6:
                tags.add("TAG_COSMETIC_MED")    # −8% (6+ pieces ≈ sides spray)
            elif piece_count >= 4:
                tags.add("TAG_MINOR_HEAVY")     # −4% (4-5 pieces)
            elif piece_count >= 1:
                tags.add("TAG_MINOR_LIGHT")     # −2% (1-3 pieces)
        else:
            # Text-based minor indicators (no piece count)
            minor_text_patterns = [
                r"\bminor\s*(touch|scratch|dent|work|repair)\b",
                r"\bsmall\s*(touch|scratch|dent)\b",
                r"\b(light|slight)\s*(scratch|dent|touch)\b",
                r"\btouchup\b",
                r"\btouch[\s-]?up\b",
                r"\bminor\b",
            ]
            for p in minor_text_patterns:
                m = re.search(p, text)
                if m and not _negated(text, m.start()):
                    tags.add("TAG_MINOR_LIGHT")  # −2% default for vague minor
                    break

    # ── OWNERSHIP CHAIN ──────────────────────────────────────────
    ownership_m = re.search(
        r"\b(\d+)(st|nd|rd|th)\s*owner\b"
        r"|\b(fourth|fifth|sixth|seventh|4th|5th|6th|7th)\s*owner\b",
        text
    )
    if ownership_m:
        raw = ownership_m.group(0)
        num_m = re.search(r"\d+", raw)
        word_map = {"fourth":4,"fifth":5,"sixth":6,"seventh":7}
        count = int(num_m.group()) if num_m else next(
            (v for k,v in word_map.items() if k in raw), None
        )
        if count and count >= 4 and not _negated(text, ownership_m.start()):
            tags.add("TAG_HIGH_OWNERSHIP")
            notes["owners"] = count

    # ── EXCELLENT / GENUINE ──────────────────────────────────────
    negative_tags = {
        "TAG_FILE_ISSUE","TAG_BOOK_ISSUE","TAG_TOKEN_UNPAID",
        "TAG_LEGAL_MISC","TAG_MAJOR_ACCIDENT","TAG_MECHANICAL_BAD",
        "TAG_COSMETIC_FULL","TAG_COSMETIC_MED","TAG_MINOR_HEAVY",
        "TAG_MINOR_LIGHT","TAG_HIGH_OWNERSHIP"
    }
    if not tags.intersection(negative_tags):
        excellent_patterns = [
            r"total\s*genuine",
            r"bumper\s*(to|-)\s*bumper\s*(genuine|original)",
            r"\bsealed\b",
            r"\bno\s*(touch|paint|accident|damage|work)\b",
            r"never\s*(touch|paint|accident)",
            r"mint\s*(condition)?",
            r"showroom\s*(condition|fresh)",
            r"zero\s*(touch|accident|work)",
        ]
        for p in excellent_patterns:
            if re.search(p, text):
                tags.add("TAG_EXCELLENT")
                break

        if "TAG_EXCELLENT" not in tags:
            if re.search(r"\bgenuine\b|\boriginal\b", text):
                tags.add("TAG_GENUINE")

    tag_str = " ".join(sorted(tags))
    return original_lower + " " + tag_str, tags, notes

def apply_penalty(tagged_desc, tags, notes, final_score):
    """
    Returns (penalty_multiplier, status_msg, badge_class, final_score)
    Priority: higher severity tags override lower ones.
    """
    penalty = 1.0
    status_msg = "Standard Market Condition"
    badge_class = "badge-fair"

    # Tier 1 — Catastrophic
    if "TAG_FILE_ISSUE" in tags:
        penalty = 0.65
        status_msg = "⚠️ Missing / Duplicate File"
        final_score = min(final_score, 5.0)
        badge_class = "badge-poor"

    elif "TAG_MAJOR_ACCIDENT" in tags:
        penalty = 0.75
        status_msg = "❌ Structural Accident Detected"
        final_score = min(final_score, 6.0)
        badge_class = "badge-poor"    

    # Tier 2 — Major Legal
    elif "TAG_BOOK_ISSUE" in tags:
        penalty = 0.88
        status_msg = "⚠️ Duplicate Book / Smart Card"
        final_score = min(final_score, 7.0)
        badge_class = "badge-fair"

    elif "TAG_MECHANICAL_BAD" in tags:
        mech = notes.get("mechanical", "")
        if mech == "engine_replaced":
            penalty = 0.82
            status_msg = "🔧 Engine Replaced / Swapped"
        else:
            penalty = 0.90
            status_msg = "🔧 Mechanical Issue Detected"
        final_score = min(final_score, 7.0)
        badge_class = "badge-fair"

    # Tier 3 — Cosmetic (graduated)
    elif "TAG_COSMETIC_FULL" in tags:
        penalty = 0.90
        status_msg = "🎨 Full Shower / Complete Repaint"
        final_score = min(final_score, 7.5)
        badge_class = "badge-fair"

    elif "TAG_COSMETIC_MED" in tags:
        pieces = notes.get("piece_count", "")
        piece_str = f" ({pieces} Pieces)" if pieces else " (Sides Spray)"
        penalty = 0.92
        status_msg = f"🎨 Medium Cosmetic Work{piece_str}"
        final_score = min(final_score, 8.0)
        badge_class = "badge-fair"

    elif "TAG_MINOR_HEAVY" in tags:
        pieces = notes.get("piece_count", "4-5")
        penalty = 0.96
        status_msg = f"🖌️ {pieces} Piece Touchup"
        badge_class = "badge-good"

    elif "TAG_MINOR_LIGHT" in tags:
        pieces = notes.get("piece_count", "1-2")
        penalty = 0.98
        status_msg = f"🖌️ Minor Touch ({pieces} Piece)"
        badge_class = "badge-good"

    # Tier 4 — Administrative
    elif "TAG_TOKEN_UNPAID" in tags:
        penalty = 0.95
        status_msg = "📋 Token / Tax Unpaid"
        final_score = min(final_score, 8.5)
        badge_class = "badge-fair"

    elif "TAG_LEGAL_MISC" in tags:
        penalty = 0.93
        status_msg = "📋 Minor Legal Issue (On-Money / City Mismatch)"
        badge_class = "badge-fair"

    elif "TAG_HIGH_OWNERSHIP" in tags:
        owners = notes.get("owners", "4th+")
        penalty = 0.95
        status_msg = f"👥 High Ownership Chain ({owners} Owner)"
        final_score = min(final_score, 8.0)
        badge_class = "badge-fair"

    # Tier 5 — Positive
    elif "TAG_EXCELLENT" in tags:
        penalty = 1.05
        status_msg = "💎 Bumper-to-Bumper Genuine"
        badge_class = "badge-excellent"

    elif "TAG_GENUINE" in tags:
        penalty = 1.02
        status_msg = "✅ Genuine Condition"
        badge_class = "badge-good"

    return penalty, status_msg, badge_class, final_score

# ─── NAV STATE ───────────────────────────────────────────────────
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "estimator"

def get_base64_of_bin_file(bin_file):
    if os.path.exists(bin_file):
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    return ""

# ─── NAV BAR ─────────────────────────────────────────────────────
tab_est = "active" if st.session_state.active_tab == "estimator" else ""
tab_chat = "active" if st.session_state.active_tab == "chat" else ""

# Get the base64 string for the logo
logo_base64 = get_base64_of_bin_file("./app_assets/logo.png")
# Format it for the img src attribute
logo_src = f"data:image/png;base64,{logo_base64}" if logo_base64 else ""

st.markdown(f"""
<div class="nav-bar">
  <div class="nav-logo">
    <img src="{logo_src}" alt="Logo" style="height: 5rem; width: auto;">
  </div>
  <div class="nav-tabs">
    <button class="nav-tab {tab_est}"  onclick="void(0)">💰 Price Estimator</button>
    <button class="nav-tab {tab_chat}" onclick="void(0)">🤖 AI Mechanic</button>
  </div>
</div>
""", unsafe_allow_html=True)

# Actual tab switching via Streamlit buttons (hidden behind nav HTML)
col_nav1, col_nav2, col_nav3 = st.columns([1.25, 1.25, 7.5])
with col_nav1:
    if st.button("💰 Estimator", key="nav_est"):
        st.session_state.active_tab = "estimator"
        st.rerun()
with col_nav2:
    if st.button("🤖 AI Chat", key="nav_chat"):
        st.session_state.active_tab = "chat"
        st.rerun()

# hide those helper buttons visually
st.markdown("""
<style>
div[data-testid="column"]:nth-child(1) button,
div[data-testid="column"]:nth-child(2) button {
    opacity: 0 !important;
    position: absolute !important;
    pointer-events: auto !important;
    width: 100% !important;
    height: 100% !important;
    top: -60px !important;
}
div[data-testid="column"]:nth-child(1),
div[data-testid="column"]:nth-child(2) {
    position: absolute !important;
    top: 0 !important;
    height: 60px !important;
    z-index: 1000 !important;
    padding: 0 !important;
}
div[data-testid="column"]:nth-child(1) { width: 1000px !important; }
div[data-testid="column"]:nth-child(2) { width: 1000px !important; }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
# SCREEN 1 — PRICE ESTIMATOR
# ════════════════════════════════════════════════════════════════
if st.session_state.active_tab == "estimator":

    st.markdown('<div class="page-wrapper">', unsafe_allow_html=True)

    # ── Hero ──
    st.markdown("""
    <div>
      <p class="hero-label">Next-Gen Intelligence</p>
      <h1 class="hero-title">Get Your Car's <em>True</em> Market Value</h1>
      <p class="hero-sub">
        Powered by 3,500+ real Pakistani listings and a hybrid AI + Rule Engine —
        the only tool that reads your seller's description, not just the specs.
      </p>
    </div>
    """, unsafe_allow_html=True)

    if price_model is None:
        st.error("🚨 Models not found. Please run notebooks 1→2→3→4 to generate model files.")
        st.stop()

    # ── Form ──
    st.markdown('<div class="form-card">', unsafe_allow_html=True)
    st.markdown('<p class="form-section-title">Vehicle Specifications</p>', unsafe_allow_html=True)

    # Row 1: Car model + Transmission + Fuel
    r1c1, r1c2, r1c3 = st.columns([3, 1.2, 1.2])
    with r1c1:
        default_index = 0
        for i, title in enumerate(car_options):
            if "Civic Oriel 1.8 i-VTEC CVT 2020" in title:
                default_index = i
                break
        car_name = st.selectbox("Select Car Model", options=car_options, index=default_index)
    with r1c2:
        trans = st.selectbox("Transmission", ["Automatic", "Manual"])
    with r1c3:
        fuel  = st.selectbox("Fuel Type", ["Petrol", "Diesel", "Hybrid"])

    # Row 2: Year slider
    year = st.slider("Model Year", 1985, 2025, 2020)

    # Row 3: Mileage + Engine
    r3c1, r3c2 = st.columns(2)
    with r3c1:
        mileage = st.number_input("Mileage (km)", 0, 300000, 50000, step=1000)
    with r3c2:
        engine  = st.number_input("Engine Capacity (cc)", 600, 6000, 1800, step=100)

    st.markdown('</div>', unsafe_allow_html=True)

    # ── Description ──
    st.markdown('<div class="form-card">', unsafe_allow_html=True)
    st.markdown('<p class="form-section-title">Paste Seller Description Here</p>', unsafe_allow_html=True)
    user_desc = st.text_area(
        "Description",
        height=110,
        value="Total genuine bumper to bumper",
        label_visibility="collapsed",
        placeholder="e.g. First owner, bumper to bumper original, new tires installed, recently serviced…",
    )
    st.markdown(
        '<p style="font-size:0.72rem;color:#2D3748;margin-top:0.4rem;">'
        'ⓘ Detailed descriptions help our AI detect hidden premium features or deductions.'
        '</p>',
        unsafe_allow_html=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # ── CTA ──
    calc_clicked = st.button("Calculate Market Value", type="primary", use_container_width=True)

    # ── Results ──
    if calc_clicked:
        with st.spinner("AI is analyzing your car…"):
            time.sleep(0.6)   # small pause

            # 1. Tag
            tagged_desc, tags, notes = smart_condition_tagger(user_desc)
            
            # 2. Inspector model → condition score
            X_text = vectorizer.transform([tagged_desc])
            predicted_score = inspector_model.predict(X_text)[0]
            final_score = round(predicted_score, 1)

            # 3. Rule engine
            legal_structural_penalty, status_msg, badge_class, final_score = apply_penalty(tagged_desc, tags, notes, final_score)

            # 4. Price model
            input_data = pd.DataFrame({
                "title_version":   [car_name],
                "model_year":      [year],
                "mileage":         [mileage],
                "engine":          [engine],
                "transmission":    [trans],
                "fuel":            [fuel],
                "inspection_score":[final_score],
            })
            ai_base_price = price_model.predict(input_data)[0]
            final_price   = ai_base_price * legal_structural_penalty

            base_input = input_data.copy()
            base_input["inspection_score"] = [9.0]
            standard_price = price_model.predict(base_input)[0]
            diff = final_price - standard_price

            # Score colour
            if final_score >= 9.0:
                score_color = "#48BB78"
            elif final_score >= 7.5:
                score_color = "#00D4FF"
            elif final_score >= 6.0:
                score_color = "#F6E05E"
            else:
                score_color = "#FF4B4B"

            # Badge for price diff
            if diff > 0:
                diff_badge  = f'<span class="price-badge premium">▲ PKR {int(diff):,} Premium</span>'
            elif diff < 0:
                diff_badge  = f'<span class="price-badge discount">▼ PKR {int(abs(diff)):,} Deduction</span>'
            else:
                diff_badge  = '<span class="price-badge neutral">— Standard Market</span>'

        # ── Display Results ──
        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        st.markdown('<div class="result-section">', unsafe_allow_html=True)

        res_left, res_right = st.columns([1.6, 1])

        with res_left:
            st.markdown(f"""
            <div class="price-hero-card">
              <p class="price-label">Smart Valuated Price</p>
              <p class="price-value">PKR {int(final_price):,}</p>
              <p class="price-standard">Standard Market: <span>PKR {int(standard_price):,}</span></p>
              {diff_badge}
              <div class="info-grid">
                <div class="info-card">
                  <p class="info-card-label">Condition</p>
                  <p class="info-card-value">{status_msg}</p>
                </div>
                <div class="info-card">
                  <p class="info-card-label">Data Basis</p>
                  <p class="info-card-value">3,500+ Listings</p>
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)

        with res_right:
            st.markdown(f"""
            <div class="score-card">
              <p class="score-label">AI Condition Score</p>
              <p class="score-number" style="color:{score_color};">{final_score}</p>
              <p style="font-size:0.72rem;color:#4A5568;margin-bottom:0.75rem;">out of 10</p>
              <span class="condition-badge {badge_class}">{status_msg}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

        # ── AI Upgrade Suggestions ──
        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        st.markdown("""
        <p class="upgrade-section-title">
          ✦ AI Upgrade Suggestions
        </p>
        """, unsafe_allow_html=True)

        if chat_available:
            with st.spinner("Generating upgrade analysis…"):
                suggestions = get_ai_recommendations(car_name, user_desc)

            if suggestions:
                for item in suggestions[:3]:
                    st.markdown(f"""
                    <div class="upgrade-card">
                      <div class="upgrade-info">
                        <h4>{item['mod']}</h4>
                        <p>{item['reason']}</p>
                      </div>
                      <div class="lift-badge">
                        <span class="lift-label">Lift Score</span>
                        +{item['lift']}/10
                      </div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            # Fallback upgrade card
            st.markdown("""
            <div class="upgrade-card">
            <div class="upgrade-info">
                <h4>Connect OpenRouter API</h4>
                <p>Add OPENROUTER_API_KEY to .env to unlock personalized upgrade suggestions</p>
            </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)  # page-wrapper


# ════════════════════════════════════════════════════════════════
# SCREEN 2 — AI MECHANIC CHAT
# ════════════════════════════════════════════════════════════════
elif st.session_state.active_tab == "chat":

    st.markdown('<div class="page-wrapper">', unsafe_allow_html=True)

    # Chat hero
    st.markdown("""
    <div class="chat-hero">
    <h1>AI Mechanic</h1>
    <p>Powered by GPT-4o Mini · Context-aware · Pakistani market specialist</p>
    </div>
    """, unsafe_allow_html=True)

    # Quick-action chips (rendered as buttons in a row)
    chip_cols = st.columns([1.4, 1.5, 1.2, 1.2, 4])
    quick_prompt = None
    with chip_cols[0]:
        if st.button("📈 Market Trends", key="chip1"):
            quick_prompt = "What are the current market trends for used cars in Pakistan?"
    with chip_cols[1]:
        if st.button("🛠️ Civic Maintenance", key="chip2"):
            quick_prompt = "What is the maintenance schedule and cost for a Honda Civic X in Pakistan?"
    with chip_cols[2]:
        if st.button("🚙 Alto vs Mira", key="chip3"):
            quick_prompt = "Compare Suzuki Alto 660cc vs Daihatsu Mira. Which is better for resale?"
    with chip_cols[3]:
        if st.button("⛽ Fuel Economy", key="chip4"):
            quick_prompt = "Which cars have the best fuel economy in Pakistan under 30 lakhs?"

    # Style chips
    st.markdown("""
    <style>
    /* Quick chip buttons */
    div[data-testid="column"]:not(:last-child) button:not([kind="primary"]) {
        background: rgba(255,255,255,0.04) !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        border-radius: 20px !important;
        color: #A0AEC0 !important;
        font-size: 0.78rem !important;
        padding: 0.3rem 0.7rem !important;
        height: auto !important;
        white-space: nowrap !important;
    }
    div[data-testid="column"]:not(:last-child) button:not([kind="primary"]):hover {
        background: rgba(255,75,75,0.1) !important;
        border-color: rgba(255,75,75,0.3) !important;
        color: #FF4B4B !important;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    if not chat_available:
        # Fallback upgrade card
        st.markdown("""
        <div class="upgrade-card">
        <div class="upgrade-info">
            <h4>Connect OpenRouter API</h4>
            <p>Add OPENROUTER_API_KEY to .env to unlock personalized upgrade suggestions</p>
        </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Init messages
        if "messages" not in st.session_state:
            st.session_state.messages = [{
                "role": "assistant",
                "content": (
                    "Hello! I'm your AI Mechanic. I specialise in the **Pakistani used car market**.\n\n"
                    "Ask me about maintenance schedules, inspection tips, resale values, or "
                    "comparisons between models. What's on your mind?"
                ),
            }]

        # Display history
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Input
        chat_input_value = st.chat_input("Ask your mechanic anything…")

        # Combine typed input + chip click
        user_prompt = chat_input_value or quick_prompt

        if user_prompt:
            st.session_state.messages.append({"role": "user", "content": user_prompt})
            with st.chat_message("user"):
                st.markdown(user_prompt)

            with st.chat_message("assistant"):
                placeholder = st.empty()
                full_response = ""
                try:
                    history_text = ""
                    for msg in st.session_state.messages[-6:]:
                        role_label = "User" if msg["role"] == "user" else "AI"
                        history_text += f"{role_label}: {msg['content']}\n"

                    response = client.chat.completions.create(
                        model="openai/gpt-4o-mini",
                        messages=[
                            {
                                "role": "system",
                                "content": (
                                    "You are an expert car mechanic and market "
                                    "analyst for the Pakistani Used Car Market.\n"
                                    "1. Keep answers concise (200-250 words max).\n"
                                    "2. Use bullet points for lists.\n"
                                    "3. Use Pakistani currency (Lakhs, Crores) "
                                    "and local terminology (Touching, Genuine, "
                                    "Shahnawaz Import, On-money, NCP).\n"
                                    "4. Use conversation history for context."
                                )
                            },
                            {
                                "role": "user",
                                "content": (
                                    f"CONVERSATION HISTORY:\n{history_text}"
                                    f"\n\nNEW QUESTION:\n{user_prompt}"
                                )
                            }
                        ],
                        temperature=0.7,
                        max_tokens=400,
                    )
                    ai_text = response.choices[0].message.content

                    for chunk in ai_text.split():
                        full_response += chunk + " "
                        time.sleep(0.04)
                        placeholder.markdown(full_response + "▌")
                    placeholder.markdown(full_response)
                    st.session_state.messages.append({"role": "assistant", "content": full_response})

                except Exception as e:
                    st.error("⚠️ Network busy or quota exceeded. Please try again in 10 seconds.")

    st.markdown('</div>', unsafe_allow_html=True)  # page-wrapper