import time
from datetime import datetime

import joblib
import pandas as pd
import streamlit as st


from src.preprocess import preprocess


st.set_page_config(
    page_title="LaptiQ — Laptop Price Predictor",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="collapsed",  
)


# cache_resource keeps the model loaded across reruns — without this 'Streamlit' reloads the .pkl on every single user interaction
@st.cache_resource
def load_model():
    return joblib.load("model/LaptiQ.pkl")

model = load_model()


STORAGE_MAP = {
    "256 GB": 256,
    "512 GB": 512,
    "1 TB":  1024,
    "2 TB":  2048,
    "4 TB":  4096,
    "8 TB":  8192,
}


# initialise session state on first load
if "page" not in st.session_state:
    st.session_state["page"] = "home"
if "history" not in st.session_state:
    st.session_state["history"] = []


# Indian number format: 124990 -> "1,24,990"
def indian_format(price: int) -> str:
    s = str(price)
    if len(s) <= 3:
        return s
    last3  = s[-3:]
    rest   = s[:-3]
    groups = []
    while rest:
        groups.append(rest[-2:])
        rest = rest[:-2]
    groups = [g for g in reversed(groups) if g]
    return ",".join(groups) + "," + last3


# =============================================================================
#  CSS
# =============================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

.stApp { background: #0f0f1a; }

.block-container {
    padding-top: 1.5rem !important;
    padding-left: 1.2rem !important;
    padding-right: 1.2rem !important;
    max-width: 1100px;
}

/* animated gradient title */
.laptiq-title {
    font-size: 3.2rem;
    font-weight: 900;
    text-align: center;
    background: linear-gradient(135deg, #23d5ab 0%, #a463f2 50%, #f2639a 100%);
    background-size: 200% 200%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: grad 6s ease infinite;
    margin-bottom: 0;
    letter-spacing: -1px;
    line-height: 1.1;
}

@keyframes grad {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

.laptiq-sub {
    text-align: center;
    color: #7c7c90;
    font-size: 1.05rem;
    margin-top: 4px;
    margin-bottom: 2rem;
    letter-spacing: 0.3px;
}

/* input section cards */
.card {
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.055);
    border-left: 3px solid;
    border-radius: 12px;
    padding: 1.25rem 1.3rem 0.8rem;
    margin-bottom: 1.2rem;
}

.card.c-basic   { border-left-color: #23d5ab; }
.card.c-cpu     { border-left-color: #a463f2; }
.card.c-mem     { border-left-color: #f2a623; }
.card.c-display { border-left-color: #2389d5; }
.card.c-gpu     { border-left-color: #f2639a; }
.card.c-build   { border-left-color: #56d423; }

.card-head {
    font-size: 0.78rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    margin-bottom: 0.85rem;
}

.card-head.c-basic   { color: #23d5ab; }
.card-head.c-cpu     { color: #a463f2; }
.card-head.c-mem     { color: #f2a623; }
.card-head.c-display { color: #2389d5; }
.card-head.c-gpu     { color: #f2639a; }
.card-head.c-build   { color: #56d423; }

/* global button style */
div.stButton > button {
    width: 100%;
    padding: 0.9rem 2rem;
    font-size: 1.1rem;
    font-weight: 700;
    letter-spacing: 0.5px;
    color: #0f0f1a;
    background: linear-gradient(135deg, #23d5ab, #1cc49e);
    border: none;
    border-radius: 12px;
    cursor: pointer;
    transition: transform 0.15s ease, box-shadow 0.2s ease;
    text-transform: uppercase;
    box-shadow: 0 2px 12px rgba(35,213,171,0.25);
}

div.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 24px rgba(35,213,171,0.4);
}

div.stButton > button:active {
    transform: translateY(0);
    box-shadow: 0 2px 8px rgba(35,213,171,0.2);
}

/* sidebar nav buttons */
section[data-testid="stSidebar"] div.stButton > button {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-left: 3px solid transparent !important;
    border-radius: 10px !important;
    color: #8a8aa0 !important;
    font-size: 0.9rem !important;
    font-weight: 500 !important;
    text-transform: none !important;
    letter-spacing: 0 !important;
    box-shadow: none !important;
    padding: 0.7rem 0.9rem !important;
    min-height: 44px !important;   /* touch target for mobile */
    margin-bottom: 0.4rem !important;
    text-align: left !important;
    justify-content: flex-start !important;
}

section[data-testid="stSidebar"] div.stButton > button:hover {
    background: rgba(255,255,255,0.055) !important;
    color: #c0c0d0 !important;
    transform: none !important;
    box-shadow: none !important;
}

/* active nav button */
section[data-testid="stSidebar"] div.stButton > button.nav-active {
    background: rgba(35,213,171,0.08) !important;
    border-left-color: #23d5ab !important;
    color: #e0e0f0 !important;
    font-weight: 600 !important;
}

/* price result card */
.result-wrap {
    animation: fadeUp 0.5s ease forwards;
    opacity: 0;
}

@keyframes fadeUp {
    from { opacity: 0; transform: translateY(18px); }
    to   { opacity: 1; transform: translateY(0); }
}

.result-box {
    background: linear-gradient(135deg, rgba(35,213,171,0.1), rgba(164,99,242,0.08));
    border: 1px solid rgba(35,213,171,0.2);
    border-radius: 16px;
    padding: 1.8rem 1.2rem;
    text-align: center;
    margin: 1.5rem auto;
    max-width: 520px;
    backdrop-filter: blur(10px);
}

.result-label {
    font-size: 0.85rem;
    color: #7c7c90;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 1.8px;
    margin-bottom: 0.4rem;
}

.result-price {
    font-size: 2.6rem;
    font-weight: 900;
    background: linear-gradient(135deg, #23d5ab, #a463f2);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.result-subprice {
    font-size: 0.95rem;
    font-weight: 500;
    color: #6a6a82;
    margin-top: 0.5rem;
    letter-spacing: 0.3px;
}

/* loading spinner */
.spin-dot {
    width: 40px;
    height: 40px;
    border: 3px solid rgba(35,213,171,0.12);
    border-top: 3px solid #23d5ab;
    border-radius: 50%;
    animation: spin 0.7s linear infinite;
    margin-bottom: 0.8rem;
}

@keyframes spin { to { transform: rotate(360deg); } }

/* sidebar */
section[data-testid="stSidebar"] {
    background: #111120 !important;
    border-right: 1px solid rgba(255,255,255,0.04);
}

.nav-label {
    font-size: 0.68rem;
    font-weight: 700;
    color: #4a4a5e;
    text-transform: uppercase;
    letter-spacing: 1.4px;
    margin-bottom: 0.7rem;
    padding-left: 0.2rem;
}

/* history cards */
.h-card {
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.055);
    border-left: 3px solid #23d5ab;
    border-radius: 12px;
    padding: 1.1rem 1.25rem 0.9rem;
    margin-bottom: 0.9rem;
}

.h-brand   { font-size: 1rem; font-weight: 700; color: #e0e0f0; }
.h-type    { font-size: 0.78rem; color: #7c7c90; margin-left: 0.4rem; }
.h-price   {
    font-size: 1.45rem; font-weight: 900;
    background: linear-gradient(135deg, #23d5ab, #a463f2);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    white-space: nowrap;
}
.h-specs   { font-size: 0.78rem; color: #8a8aa0; margin: 0.4rem 0; line-height: 1.5; }
.h-specs span { color: #a0a0b8; }
.h-os      { font-size: 0.75rem; color: #7c7c90; }
.h-time    { font-size: 0.7rem; color: #505068; font-style: italic; }

.h-empty {
    text-align: center;
    color: #5a5a70;
    font-size: 0.95rem;
    padding: 3.5rem 1rem;
    line-height: 1.7;
}

/* "Clear All" button */
@keyframes redPulse {
    0%   { box-shadow: 0 0 4px rgba(220,38,38,0.3), 0 0 12px rgba(220,38,38,0.1); }
    50%  { box-shadow: 0 0 8px rgba(220,38,38,0.5), 0 0 20px rgba(220,38,38,0.2); }
    100% { box-shadow: 0 0 4px rgba(220,38,38,0.3), 0 0 12px rgba(220,38,38,0.1); }
}

button.btn-clear-all,
[data-testid="stElementContainer"]:has(button[kind="secondary"]) button[kind="secondary"].btn-clear-all {
    background: transparent !important;
    border: 1.5px solid rgba(220,38,38,0.6) !important;
    color: #dc2626 !important;
    font-size: 0.82rem !important;
    font-weight: 700 !important;
    padding: 0.4rem 1rem !important;
    border-radius: 8px !important;
    text-transform: none !important;
    letter-spacing: 0.3px !important;
    width: auto !important;
    min-height: unset !important;
    animation: redPulse 2.5s ease-in-out infinite !important;
}

button.btn-clear-all:hover {
    background: rgba(220,38,38,0.08) !important;
    border-color: rgba(220,38,38,0.85) !important;
    transform: none !important;
    animation: redPulse 1.5s ease-in-out infinite !important;
}

/* delete buttons */
button.btn-del,
[data-testid="stElementContainer"]:has(button[kind="secondary"]) button[kind="secondary"].btn-del {
    background: #dc2626 !important;
    border: none !important;
    color: #fff !important;
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    padding: 0.22rem 0.65rem !important;
    border-radius: 6px !important;
    text-transform: none !important;
    box-shadow: none !important;
    letter-spacing: 0 !important;
    width: auto !important;
    min-height: unset !important;
}

button.btn-del:hover {
    background: #b91c1c !important;
    transform: none !important;
    box-shadow: none !important;
}

/* model info page */
.mi-desc {
    color: #8a8aa0;
    font-size: 0.95rem;
    max-width: 620px;
    margin: 0.5rem auto 2rem;
    line-height: 1.7;
    text-align: center;
}

.mi-section {
    font-size: 0.78rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.4px;
    color: #23d5ab;
    margin: 2rem 0 0.9rem;
}

.mi-step {
    display: flex;
    align-items: flex-start;
    gap: 0.85rem;
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.055);
    border-radius: 10px;
    padding: 0.85rem 1.1rem;
    margin-bottom: 0.6rem;
}

.mi-num {
    font-size: 0.7rem;
    font-weight: 800;
    color: #0f0f1a;
    background: linear-gradient(135deg, #23d5ab, #1cc49e);
    border-radius: 50%;
    width: 24px; height: 24px;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    margin-top: 1px;
}

.mi-text { font-size: 0.88rem; color: #c0c0d0; line-height: 1.4; }


.sb-stat {
    background: rgba(255,255,255,0.035);
    border: 1px solid rgba(255,255,255,0.05);
    border-radius: 10px;
    padding: 0.9rem 1rem;
    margin-bottom: 0.7rem;
}

.sb-stat-label {
    font-size: 0.7rem;
    color: #7c7c90;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-bottom: 0.25rem;
}

.sb-stat-val        { font-size: 1.35rem; font-weight: 800; }
.sb-stat-val.green  { color: #23d5ab; }
.sb-stat-val.orange { color: #f2a623; }
.sb-stat-val.purple { color: #a463f2; }

.footer-note {
    text-align: center;
    color: #4a4a5e;
    font-size: 0.78rem;
    margin-top: 2.5rem;
    padding: 0.8rem 0;
    border-top: 1px solid rgba(255,255,255,0.04);
    font-style: italic;
}

/* input widgets */
.stSelectbox > div > div,
.stTextInput > div > div > input,
.stNumberInput > div > div > input {
    background: rgba(255,255,255,0.035) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 8px !important;
    color: #e0e0f0 !important;
}

.stSlider > div > div > div { color: #23d5ab !important; }

/* widget labels */
.stSelectbox label,
.stTextInput label,
.stNumberInput label,
.stSlider label,
.stRadio label,
.stCheckbox label,
.stMultiSelect label,
[data-testid="stWidgetLabel"],
[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] label {
    color: #c8c8dc !important;
    font-weight: 500 !important;
    opacity: 1 !important;
}

/* selectbox value + dropdown */
.stSelectbox [data-testid="stMarkdownContainer"],
.stSelectbox [data-baseweb="select"] span,
.stSelectbox [data-baseweb="select"] .css-1dimb5e-singleValue,
.stSelectbox [data-baseweb="select"] > div > div > div,
.stSelectbox div[data-baseweb="select"] > div {
    color: #e0e0f0 !important;
}

[data-baseweb="popover"],
[data-baseweb="popover"] ul,
[data-baseweb="menu"],
[role="listbox"] {
    background: #1a1a2e !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
}

/* dropdown options */
[data-baseweb="popover"] li,
[data-baseweb="menu"] li,
[role="option"],
[data-baseweb="popover"] ul li,
ul[role="listbox"] li {
    color: #d0d0e4 !important;
    background: transparent !important;
}

[data-baseweb="popover"] li:hover,
[data-baseweb="menu"] li:hover,
[role="option"]:hover,
li[aria-selected="true"],
[data-baseweb="popover"] li[aria-selected="true"] {
    background: rgba(35,213,171,0.12) !important;
    color: #ffffff !important;
}

[data-baseweb="menu"] li[data-highlighted="true"],
[role="option"][data-highlighted] {
    background: rgba(35,213,171,0.15) !important;
    color: #ffffff !important;
}

/* input text + placeholders */
.stTextInput input,
.stNumberInput input {
    color: #e0e0f0 !important;
}

.stTextInput input::placeholder,
.stNumberInput input::placeholder {
    color: #6a6a82 !important;
    opacity: 1 !important;
}

.stSelectbox svg,
.stSelectbox [data-baseweb="select"] svg {
    fill: #8a8aa0 !important;
    color: #8a8aa0 !important;
}

/* tooltip icons */
.stSelectbox [data-testid="stTooltipIcon"],
.stTextInput [data-testid="stTooltipIcon"],
.stNumberInput [data-testid="stTooltipIcon"] {
    color: #6a6a82 !important;
}

/* lock dark background across all containers */
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stHeader"],
header[data-testid="stHeader"] {
    background: #0f0f1a !important;
}

[data-testid="stToolbar"],
[data-testid="stStatusWidget"] {
    color: #7c7c90 !important;
}

.streamlit-expanderHeader { color: #c8c8dc !important; }
.streamlit-expanderContent { color: #a0a0b8 !important; }

#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }


/* tablets */
@media (max-width: 768px) {
    .block-container {
        padding-left: 0.7rem !important;
        padding-right: 0.7rem !important;
        padding-top: 1rem !important;
    }

    .laptiq-title  { font-size: 2.2rem; letter-spacing: -0.5px; }
    .laptiq-sub    { font-size: 0.88rem; margin-bottom: 1.4rem; }
    .card          { padding: 1rem 1rem 0.5rem; margin-bottom: 0.9rem; border-radius: 10px; }
    .card-head     { font-size: 0.72rem; margin-bottom: 0.65rem; }

    /* stack st.columns vertically on mobile */
    [data-testid="stHorizontalBlock"] {
        flex-direction: column !important;
        gap: 0 !important;
    }

    [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
        width: 100% !important;
        flex: 1 1 100% !important;
        min-width: 100% !important;
    }

    div.stButton > button  { padding: 0.85rem 1.5rem; font-size: 1rem; border-radius: 10px; }
    .result-box            { padding: 1.3rem 1rem; margin: 1rem auto; border-radius: 12px; }
    .result-price          { font-size: 1.7rem; }
    .result-subprice       { font-size: 0.82rem; }
    .result-label          { font-size: 0.75rem; letter-spacing: 1.2px; }
    .footer-note           { font-size: 0.7rem; margin-top: 1.5rem; }
    .h-price               { font-size: 1.2rem; }
}

/* small phones */
@media (max-width: 480px) {
    .laptiq-title  { font-size: 1.8rem; }
    .laptiq-sub    { font-size: 0.82rem; margin-bottom: 1.2rem; }
    .card          { padding: 0.85rem 0.85rem 0.4rem; }
    .result-price  { font-size: 1.35rem; }
    .result-subprice { font-size: 0.75rem; }
    .h-price       { font-size: 1.1rem; }

    div.stButton > button { font-size: 0.92rem; padding: 0.75rem 1rem; }
}

/* iPads */
@media (min-width: 769px) and (max-width: 1024px) {
    .laptiq-title { font-size: 2.8rem; }
    .block-container { padding-left: 1rem !important; padding-right: 1rem !important; }
    .result-price { font-size: 2rem; }
    .result-subprice { font-size: 0.88rem; }
}
</style>
""", unsafe_allow_html=True)


# JS to apply CSS classes to specific buttons
st.html("""
<script>
function styleLaptiQButtons() {
    document.querySelectorAll('button[kind="secondary"]').forEach(btn => {
        const text = btn.textContent.trim();
        if (text.includes('Clear All')) {
            btn.classList.add('btn-clear-all');
        }
        if (text.length <= 2 && !text.includes('Clear')) {
            btn.classList.add('btn-del');
        }
    });
}
// Run with retries and observe DOM changes
styleLaptiQButtons();
setTimeout(styleLaptiQButtons, 200);
setTimeout(styleLaptiQButtons, 600);
setTimeout(styleLaptiQButtons, 1200);
setTimeout(styleLaptiQButtons, 2500);
const _lqObs = new MutationObserver(() => {
    requestAnimationFrame(styleLaptiQButtons);
});
_lqObs.observe(document.body, {childList: true, subtree: true});
</script>
""", unsafe_allow_javascript=True)


# --- sidebar navigation ------------------------------
with st.sidebar:
    st.markdown('<div class="nav-label">Navigation</div>', unsafe_allow_html=True)

    active = st.session_state["page"]

    nav_items = [
        ("home",       "🏠  Home"),
        ("history",    "🕒  History"),
        ("model_info", "📊  Model Info"),
    ]

    for key, label in nav_items:
        if st.button(label, key=f"nav_{key}", use_container_width=True):
            st.session_state["page"] = key
            st.rerun()

        if active == key:
            st.markdown(
                '<div style="height:2px;background:linear-gradient(90deg,#23d5ab,transparent);'
                'border-radius:2px;margin:-0.3rem 0 0.3rem 0.9rem;width:55%;"></div>',
                unsafe_allow_html=True
            )

    st.markdown("---")
    st.markdown(
        '<p style="color:#4a4a5e;font-size:0.7rem;text-align:center;">LaptiQ v1.0 · XGBoost</p>',
        unsafe_allow_html=True,
    )


# =============================================================================
#  PAGE: HOME
# =============================================================================
if st.session_state["page"] == "home":

    st.markdown('<h1 class="laptiq-title">LaptiQ</h1>', unsafe_allow_html=True)
    st.markdown('<p class="laptiq-sub">Know what your laptop is worth</p>', unsafe_allow_html=True)

    st.markdown('<div class="card c-basic"><div class="card-head c-basic">🏷️ Basic Info</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        brand = st.selectbox("Brand", [
            "Lenovo", "Asus", "HP", "MSI", "Apple", "Acer", "Dell", "Samsung", "Other"
        ])
    with c2:
        laptop_type = st.selectbox("Laptop Type", [
            "Gaming", "Notebook", "Business", "Ultrabook",
            "2-in-1 Convertible", "Workstation", "Creator"
        ])
    with c3:
        os = st.selectbox("Operating System", ["Windows", "macOS", "ChromeOS", "Other"])
    with c4:
        launch_year = st.selectbox("Launch Year", [2026, 2025, 2024, 2023])
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card c-cpu"><div class="card-head c-cpu">⚡ Processor</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        cpu_brand = st.selectbox("CPU Brand", ["Intel", "AMD", "Apple", "Qualcomm"])
    with c2:
        cpu_model = st.text_input(
            "CPU Model",
            placeholder="e.g. Intel Core i7-13700H",
            help="Full model name — e.g. AMD Ryzen 5 7530U, Apple M3 Pro, Snapdragon X Elite"
        )
    with c3:
        cpu_cores = st.number_input("CPU Cores", min_value=2, max_value=24, value=8, step=1)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card c-mem"><div class="card-head c-mem">💾 Memory & Storage</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        ram = st.selectbox("RAM", [8, 16, 24, 32, 48, 64, 96, 128], format_func=lambda x: f"{x} GB")
    with c2:
        storage_label = st.selectbox("Storage", list(STORAGE_MAP.keys()))
    with c3:
        storage_type = st.selectbox("Storage Type", ["SSD", "HDD"])
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card c-display"><div class="card-head c-display">🖥️ Display</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        screen_size = st.selectbox(
            "Screen Size",
            [11.6, 13.3, 13.6, 14.0, 15.6, 16.0, 17.3],
            format_func=lambda x: f'{x}"'
        )
    with c2:
        resolution = st.selectbox("Resolution", [
            "1920x1080", "1920x1200", "2560x1600", "2560x1664",
            "2880x1800", "2880x1864", "3024x1964", "3456x2160",
            "3456x2234", "3840x2160"
        ])
    with c3:
        refresh_rate = st.selectbox(
            "Refresh Rate",
            [60, 90, 120, 144, 165, 240, 360],
            format_func=lambda x: f"{x} Hz"
        )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card c-gpu"><div class="card-head c-gpu">🎮 Graphics</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        gpu_type = st.selectbox("GPU Type", ["Dedicated", "Integrated"])
    with c2:
        gpu_model = st.text_input(
            "GPU Model",
            placeholder="e.g. NVIDIA RTX 4060",
            help="e.g. NVIDIA RTX 4060, Intel Iris Xe, AMD Radeon 780M, Apple M3 GPU"
        )
    with c3:
        gpu_vram_label = st.selectbox(
            "GPU VRAM",
            ["Shared", "4GB", "6GB", "8GB", "12GB", "16GB", "24GB"],
            help="Pick 'Shared' for integrated graphics with no dedicated VRAM"
        )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card c-build"><div class="card-head c-build">⚖️ Build</div>', unsafe_allow_html=True)
    weight = st.number_input("Weight (kg)", min_value=0.8, max_value=4.5, value=1.8, step=0.1, format="%.1f")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    predict_clicked = st.button("⚡  Predict Price", use_container_width=True)

    if predict_clicked:
        if not cpu_model.strip():
            st.warning("Please enter a CPU Model — e.g. Intel Core i7-13700H")
        elif not gpu_model.strip():
            st.warning("Please enter a GPU Model — e.g. NVIDIA RTX 4060, Intel Iris Xe")
        else:
            placeholder = st.empty()
            placeholder.markdown("""
            <div style="display:flex;flex-direction:column;align-items:center;padding:1.5rem;">
                <div class="spin-dot"></div>
                <div style="color:#7c7c90;font-size:0.9rem;font-weight:500;">
                    Crunching your specs…
                </div>
            </div>
            """, unsafe_allow_html=True)

            time.sleep(0.8)  

            gpu_vram_raw = "Shared" if gpu_vram_label == "Shared" else gpu_vram_label.replace("GB", "").strip()

            input_data = {
                "Brand":        brand,
                "Laptop_Type":  laptop_type,
                "Launch_Year":  int(launch_year),
                "CPU_Brand":    cpu_brand,
                "CPU_Model":    cpu_model.strip(),
                "CPU_Cores":    int(cpu_cores),
                "GPU_Type":     gpu_type,
                "GPU_VRAM":     gpu_vram_raw,
                "GPU_Model":    gpu_model.strip(),
                "Screen_Size":  screen_size,
                "Resolution":   resolution,
                "Refresh_Rate": refresh_rate,
                "RAM":          ram,
                "Storage":      STORAGE_MAP[storage_label],
                "Storage_Type": storage_type,
                "Weight":       float(weight),
                "OS":           os,
            }

            try:
                input_df   = pd.DataFrame([input_data])
                processed  = preprocess(input_df)
                prediction = model.predict(processed)
                price      = int(round(prediction[0]))

                margin     = price * 0.12

                lower_price = int(round((price - margin) / 500) * 500)
                upper_price = int(round((price + margin) / 500) * 500)
                
                formatted       = indian_format(price)
                formatted_lower = indian_format(lower_price)
                formatted_upper = indian_format(upper_price)

                # save to history — max 20 entries, oldest dropped first
                entry = {
                    "timestamp":   datetime.now().strftime("%d %b %Y, %I:%M %p"),
                    "brand":       brand,
                    "laptop_type": laptop_type,
                    "cpu_model":   cpu_model.strip(),
                    "ram":         f"{ram} GB",
                    "storage":     storage_label,
                    "gpu_model":   gpu_model.strip(),
                    "os":          os,
                    "price":       formatted,
                }

                hist = st.session_state["history"]
                if len(hist) >= 20:
                    hist.pop(0)
                hist.append(entry)

                placeholder.empty()
                st.markdown(f"""
                <div class="result-wrap">
                    <div class="result-box">
                        <div class="result-label">Estimated Price Range</div>
                        <div class="result-price">₹{formatted_lower} – ₹{formatted_upper}</div>
                        <div class="result-subprice">(Most likely around ₹{formatted})</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            except Exception as e:
                placeholder.empty()
                st.error(f"Prediction failed: {e}")

    st.markdown("""
    <div class="footer-note">
        Predictions are based on training data and current market trends. Actual prices may vary.
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
#  PAGE: HISTORY
# =============================================================================
elif st.session_state["page"] == "history":

    hist = st.session_state["history"]

    top_l, top_r = st.columns([5, 1])
    with top_l:
        st.markdown('<h1 class="laptiq-title" style="text-align:left;font-size:2rem;">Prediction History</h1>', unsafe_allow_html=True)
    with top_r:
        if hist:
            if st.button("🗑 Clear All", key="clear_all"):
                st.session_state["history"] = []
                st.rerun()

    if not hist:
        st.markdown("""
        <div class="h-empty">
            📋<br><br>
            No predictions yet.<br>
            Head to Home and predict your first laptop!
        </div>
        """, unsafe_allow_html=True)
    else:
        for idx, entry in enumerate(reversed(hist)):
            real_idx = len(hist) - 1 - idx  

            row_l, row_r = st.columns([11, 1])
            with row_l:
                st.markdown(f"""
                <div class="h-card">
                    <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:0.45rem;">
                        <div>
                            <span class="h-brand">{entry['brand']}</span>
                            <span class="h-type">· {entry['laptop_type']}</span>
                        </div>
                        <div class="h-price">₹{entry['price']}</div>
                    </div>
                    <div class="h-specs">
                        <span>{entry['cpu_model']}</span> &nbsp;|&nbsp;
                        <span>{entry['ram']}</span> &nbsp;|&nbsp;
                        <span>{entry['storage']}</span> &nbsp;|&nbsp;
                        <span>{entry['gpu_model']}</span>
                    </div>
                    <div style="display:flex;justify-content:space-between;align-items:center;">
                        <div class="h-os">{entry['os']}</div>
                        <div class="h-time">{entry['timestamp']}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            with row_r:
                if st.button("🗑", key=f"del_{real_idx}"):
                    st.session_state["history"].pop(real_idx)
                    st.rerun()


# =============================================================================
#  PAGE: MODEL INFO
# =============================================================================
elif st.session_state["page"] == "model_info":

    st.markdown('<h1 class="laptiq-title">About LaptiQ</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div class="mi-desc">
        LaptiQ estimates the current market value of any laptop based on its specs.
        Enter the processor, GPU, RAM, display, and build details — and LaptiQ returns
        a price prediction in Indian Rupees, powered by a tuned XGBoost pipeline.
    </div>
    """, unsafe_allow_html=True)

    # --- Model Stats ------------------------------

    st.markdown('<div class="mi-section">📊 Model Performance</div>', unsafe_allow_html=True)
    s1, s2, s3 = st.columns(3)
    with s1:
        st.markdown("""
        <div class="sb-stat">
            <div class="sb-stat-label">R² Accuracy</div>
            <div class="sb-stat-val green">93.46%</div>
        </div>
        """, unsafe_allow_html=True)
    with s2:
        st.markdown("""
        <div class="sb-stat">
            <div class="sb-stat-label">Avg Prediction Error</div>
            <div class="sb-stat-val orange">11.68%</div>
        </div>
        """, unsafe_allow_html=True)
    with s3:
        st.markdown("""
        <div class="sb-stat">
            <div class="sb-stat-label">Laptops Trained On</div>
            <div class="sb-stat-val purple">935</div>
        </div>
        """, unsafe_allow_html=True)

    # --- How It Works ------------------------------

    st.markdown('<div class="mi-section">⚙️ How It Works</div>', unsafe_allow_html=True)
    steps = [
        "Enter your laptop specs",
        "LaptiQ engineers your input",
        "A tuned XGBoost pipeline predicts the log-transformed price",
        "The result is converted back and displayed in Indian rupee format",
    ]
    for i, text in enumerate(steps, 1):
        st.markdown(f"""
        <div class="mi-step">
            <div class="mi-num">{i}</div>
            <div class="mi-text">{text}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="footer-note">
        Built with XGBoost · sklearn Pipeline · Streamlit
    </div>
    """, unsafe_allow_html=True)