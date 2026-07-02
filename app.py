import time
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import streamlit as st

import shap


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

@st.cache_resource
def load_explainer():
    xgb_model = model.regressor_.named_steps['laptiQ']
    return shap.TreeExplainer(xgb_model)

explainer = load_explainer()


STORAGE_MAP = {
    "256 GB": 256,
    "512 GB": 512,
    "1 TB":  1024,
    "2 TB":  2048,
    "4 TB":  4096,
    "8 TB":  8192,
}


if "page" not in st.session_state:
    st.session_state["page"] = "home"
if "history" not in st.session_state:
    st.session_state["history"] = []
if "compare_active_tab" not in st.session_state:
    st.session_state["compare_active_tab"] = 0


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


FEATURE_GROUPS = {
    "oridinal__GPU_Tier":               "Graphic-Card",
    "oridinal__CPU_Series":             "CPU",
    "oridinal__CPU_Segment":            "CPU",
    "oridinal__CPU_Generation":         "CPU",
    "one_hot__Brand_Apple":             "Brand",
    "one_hot__Brand_Asus":              "Brand",
    "one_hot__Brand_Dell":              "Brand",
    "one_hot__Brand_HP":                "Brand",
    "one_hot__Brand_Lenovo":            "Brand",
    "one_hot__Brand_MSI":               "Brand",
    "one_hot__Brand_Samsung":           "Brand",
    "one_hot__Laptop_Type_Business":    "Laptop Type",
    "one_hot__Laptop_Type_Creator":     "Laptop Type",
    "one_hot__Laptop_Type_Gaming":      "Laptop Type",
    "one_hot__Laptop_Type_Notebook":    "Laptop Type",
    "one_hot__Laptop_Type_Ultrabook":   "Laptop Type",
    "one_hot__Laptop_Type_Workstation": "Laptop Type",
    "one_hot__CPU_Brand_Apple":         "CPU Brand",
    "one_hot__CPU_Brand_Intel":         "CPU Brand",
    "one_hot__CPU_Brand_Qualcomm":      "CPU Brand",
    "one_hot__GPU_Type_Integrated":     "GPU Type",
    "one_hot__OS_Windows":              "Operating System",
    "one_hot__OS_macOS":                "Operating System",
    "log__RAM":                         "RAM",
    "log__Storage":                     "Storage Capacity",
    "yeo_johnson__Weight":              "Weight",
    "yeo_johnson__Pixel_Per_Inch":      "Display Quality ",
    "robust_scaler__CPU_Cores":         "CPU Cores",
    "robust_scaler__GPU_VRAM":          "Graphics Memory (VRAM)",
    "robust_scaler__Laptop_Age":        "Launch Year",
    "remainder__Refresh_Rate":          "Refresh Rate",
}


def get_top_factors(explainer, encoded_array, feature_names, top_n=5):
    shap_values = explainer.shap_values(encoded_array)[0]
    baseline_log = explainer.expected_value
    base_price = np.exp(baseline_log)

    col_impacts = {}
    for name, value in zip(feature_names, shap_values):
        col_impacts[name] = np.exp(baseline_log + value) - base_price

    group_impacts = {}
    for col_name, rupee_shift in col_impacts.items():
        group = FEATURE_GROUPS.get(col_name, col_name)
        group_impacts[group] = group_impacts.get(group, 0) + rupee_shift

    factors = []
    for label, impact in group_impacts.items():
        factors.append({
            "label":     label,
            "impact":    int(round(impact)),
            "direction": "up" if impact >= 0 else "down",
        })

    factors.sort(key=lambda f: abs(f["impact"]), reverse=True)
    return factors[:top_n]


def render_shap_html(factors, compact=False):
    if not factors:
        return ""
    max_imp = max(abs(f["impact"]) for f in factors) or 1
    rows = ""
    for i, f in enumerate(factors):
        d = f["direction"]
        sign = "+" if d == "up" else "&minus;"
        pct = min(abs(f["impact"]) / max_imp * 100, 100)
        delay = 0.3 + i * 0.07
        rows += (
            f'<div class="shap-row" style="animation-delay:{delay:.2f}s">'
            f'<div class="shap-dot {d}"></div>'
            f'<span class="shap-name">{f["label"]}</span>'
            f'<span class="shap-val {d}">{sign}\u20b9{indian_format(abs(f["impact"]))}</span>'
            f'<div class="shap-bar-bg"><div class="shap-bar-fg {d}" style="width:{pct:.0f}%"></div></div>'
            f'</div>'
        )
    cls = "shap-wrap compact" if compact else "shap-wrap"
    return (
        f'<div class="{cls}">'
        f'<div class="shap-title">What Drives This Price?</div>'
        f'{rows}'
        f'</div>'
    )


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

/* sidebar nav buttons — YouTube style */
section[data-testid="stSidebar"] div.stButton > button {
    background: transparent !important;
    border: none !important;
    border-radius: 10px !important;
    color: #a0a0b8 !important;
    font-size: 0.88rem !important;
    font-weight: 500 !important;
    text-transform: none !important;
    letter-spacing: 0.2px !important;
    box-shadow: none !important;
    padding: 0.6rem 0.75rem !important;
    min-height: 40px !important;
    margin-bottom: 0.15rem !important;
    text-align: left !important;
    justify-content: flex-start !important;
    transition: background 0.15s ease, color 0.15s ease !important;
}

section[data-testid="stSidebar"] div.stButton > button:hover {
    background: rgba(255,255,255,0.07) !important;
    color: #e0e0f0 !important;
    transform: none !important;
    box-shadow: none !important;
}

/* active nav button — YouTube filled background */
section[data-testid="stSidebar"] div.stButton > button.nav-active {
    background: rgba(255,255,255,0.1) !important;
    color: #ffffff !important;
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

/* sidebar — YouTube style */
section[data-testid="stSidebar"] {
    background: #0f0f1a !important;
    border-right: 1px solid rgba(255,255,255,0.06);
}

section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
    padding-top: 1rem !important;
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

.sb-divider {
    height: 1px;
    background: rgba(255,255,255,0.06);
    margin: 0.6rem 0.5rem;
}

/* history entries */
.h-card {
    position: relative;
    background: transparent;
    border: none;
    border-radius: 0;
    padding: 1.1rem 0;
    margin-bottom: 0;
}

.h-separator {
    height: 1px;
    background: rgba(255,255,255,0.08);
    margin: 0;
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
.section-title-wrap {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    margin: 2rem 0 1.2rem;
}
.section-icon {
    display: flex;
    align-items: center;
    color: #4e8cff;
}
.section-title {
    font-size: 1.15rem;
    font-weight: 600;
    color: #e0e0f0;
}
.section-divider {
    flex: 1;
    height: 1px;
    background: rgba(255,255,255,0.06);
}

.sb-stat-new {
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.05);
    border-radius: 12px;
    padding: 1.25rem;
    margin-bottom: 0.7rem;
    display: flex;
    flex-direction: column;
}

/* Form Sections */
.fs-wrap {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    margin: 1.5rem 0 1rem;
}
.fs-icon {
    width: 32px;
    height: 32px;
    border-radius: 8px;
    background: rgba(120, 120, 250, 0.12);
    color: #8c8df0;
    display: flex;
    align-items: center;
    justify-content: center;
}
.fs-title {
    font-size: 0.95rem;
    font-weight: 700;
    letter-spacing: 0.5px;
    color: #ffffff;
}
.fs-divider {
    height: 1px;
    background: rgba(255,255,255,0.06);
    margin: 1rem 0 1.5rem;
    width: 100%;
}
div.stButton > button[kind="primary"] {
    background: #93C5FD !important;
    color: #0f0f1a !important;
    font-weight: 700 !important;
    border: none !important;
}
div.stButton > button[kind="primary"]:hover {
    background: #bfdbfe !important;
    color: #0f0f1a !important;
}
.sb-stat-new-extra {
    border-radius: 12px;
    padding: 1.25rem;
    margin-bottom: 0.7rem;
    display: flex;
    flex-direction: column;
}
.sb-stat-top {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 1.2rem;
}
.sb-stat-title {
    font-size: 0.9rem;
    color: #c0c0d0;
    font-weight: 500;
}
.sb-stat-icon {
    width: 34px;
    height: 34px;
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
}
.sb-stat-icon.blue { background: rgba(78, 140, 255, 0.12); color: #4e8cff; }
.sb-stat-icon.orange { background: rgba(234, 179, 8, 0.12); color: #eab308; }
.sb-stat-icon.purple { background: rgba(180, 108, 248, 0.12); color: #b46cf8; }

.sb-stat-val-new { font-size: 1.8rem; font-weight: 700; margin-bottom: 0.4rem; }
.sb-stat-val-new.blue { color: #93C5FD; }
.sb-stat-val-new.orange { color: #FCD34D; }
.sb-stat-val-new.purple { color: #C4B5FD; }

.sb-stat-sub { font-size: 0.75rem; color: #7c7c90; }

.mi-step-card {
    display: flex;
    align-items: center;
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.05);
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 0.8rem;
    transition: background 0.2s ease;
}
.mi-step-card:hover {
    background: rgba(255,255,255,0.04);
}
.mi-step-num {
    width: 34px;
    height: 34px;
    border-radius: 50%;
    background: #3b5bdb;
    color: white;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 700;
    font-size: 0.9rem;
    flex-shrink: 0;
    margin-right: 1.2rem;
}
.mi-step-content {
    flex: 1;
}
.mi-step-title {
    font-size: 0.95rem;
    color: #e0e0f0;
    font-weight: 500;
    margin-bottom: 0.25rem;
}
.mi-step-desc {
    font-size: 0.82rem;
    color: #8a8aa0;
}
.mi-step-chevron {
    flex-shrink: 0;
    margin-left: 1rem;
    display: flex;
    align-items: center;
}

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

/* compare page */
.compare-col-header {
    font-size: 1.05rem;
    font-weight: 700;
    color: #e0e0f0;
    text-align: center;
    margin-bottom: 1rem;
    padding: 0.6rem 0;
    border-bottom: 2px solid rgba(35,213,171,0.15);
    letter-spacing: 0.3px;
}

div.stButton.cmp-tab-btn > button {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    color: #8a8aa0 !important;
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    padding: 0.5rem 0.8rem !important;
    border-radius: 8px !important;
    text-transform: none !important;
    box-shadow: none !important;
    letter-spacing: 0 !important;
}

div.stButton.cmp-tab-btn > button:hover {
    background: rgba(255,255,255,0.055) !important;
    color: #c0c0d0 !important;
    transform: none !important;
}

div.stButton.cmp-tab-btn > button.cmp-tab-active {
    background: rgba(35,213,171,0.1) !important;
    border-color: rgba(35,213,171,0.35) !important;
    color: #23d5ab !important;
    font-weight: 700 !important;
}

.compare-result-box {
    background: linear-gradient(135deg, rgba(35,213,171,0.1), rgba(164,99,242,0.08));
    border: 1px solid rgba(35,213,171,0.2);
    border-radius: 16px;
    padding: 1.4rem 1rem;
    text-align: center;
    backdrop-filter: blur(10px);
    margin-bottom: 1rem;
    transition: border-color 0.3s ease, box-shadow 0.3s ease;
}

.compare-result-box.best-value {
    border-color: rgba(35,213,171,0.5);
    box-shadow: 0 0 20px rgba(35,213,171,0.15), 0 0 40px rgba(35,213,171,0.05);
}

.compare-result-header {
    font-size: 0.85rem;
    font-weight: 600;
    color: #c0c0d0;
    margin-bottom: 0.15rem;
}

.compare-result-sub {
    font-size: 0.72rem;
    color: #7c7c90;
    margin-bottom: 0.6rem;
}

.best-value-badge {
    display: inline-block;
    font-size: 0.75rem;
    font-weight: 700;
    color: #0f0f1a;
    background: linear-gradient(135deg, #23d5ab, #1cc49e);
    padding: 0.25rem 0.75rem;
    border-radius: 20px;
    margin-top: 0.7rem;
    letter-spacing: 0.3px;
    box-shadow: 0 0 12px rgba(35,213,171,0.35);
}


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
    .result-price          { font-size: 1.7rem !important; }
    .result-subprice       { font-size: 0.82rem; }
    .result-label          { font-size: 0.75rem; letter-spacing: 1.2px; }
    .footer-note           { font-size: 0.7rem; margin-top: 1.5rem; }
    .h-price               { font-size: 1.2rem; }

    /* compare mobile */
    .compare-col-header { font-size: 0.92rem; padding: 0.4rem 0; }
    .compare-result-box { padding: 1.1rem 0.8rem; border-radius: 12px; }

    /* keep compare tab pills in a row on mobile */
    [data-testid="stHorizontalBlock"]:has(.cmp-tab-marker) {
        flex-direction: row !important;
        gap: 0.4rem !important;
    }
    [data-testid="stHorizontalBlock"]:has(.cmp-tab-marker) > [data-testid="stColumn"] {
        min-width: 0 !important;
        flex: 1 1 0% !important;
        width: auto !important;
    }
}

/* small phones */
@media (max-width: 480px) {
    .laptiq-title  { font-size: 1.8rem; }
    .laptiq-sub    { font-size: 0.82rem; margin-bottom: 1.2rem; }
    .card          { padding: 0.85rem 0.85rem 0.4rem; }
    .result-price  { font-size: 1.35rem !important; }
    .result-subprice { font-size: 0.75rem; }
    .h-price       { font-size: 1.1rem; }

    div.stButton > button { font-size: 0.92rem; padding: 0.75rem 1rem; }
    div.stButton.cmp-tab-btn > button { font-size: 0.75rem !important; padding: 0.4rem 0.5rem !important; }
}

/* iPads */
@media (min-width: 769px) and (max-width: 1024px) {
    .laptiq-title { font-size: 2.8rem; }
    .block-container { padding-left: 1rem !important; padding-right: 1rem !important; }
    .result-price { font-size: 2rem !important; }
    .result-subprice { font-size: 0.88rem; }
    .compare-col-header { font-size: 0.95rem; }
}
/* SHAP price factors */
.shap-wrap {
    max-width: 520px;
    margin: 1.5rem auto 0;
    animation: fadeUp 0.5s ease forwards;
    animation-delay: 0.15s;
    opacity: 0;
}

.shap-wrap.compact { max-width: 100%; margin-top: 1rem; }

.shap-title {
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: #C4B5FD;
    text-align: center;
    margin-bottom: 0.65rem;
}

.shap-row {
    display: flex;
    align-items: center;
    gap: 0.55rem;
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.05);
    border-radius: 10px;
    padding: 0.55rem 0.9rem;
    margin-bottom: 0.35rem;
    transition: background 0.2s ease, transform 0.15s ease;
    animation: fadeUp 0.4s ease forwards;
    opacity: 0;
}

.shap-row:hover {
    background: rgba(255,255,255,0.045);
    transform: translateX(3px);
}

.shap-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    flex-shrink: 0;
}

.shap-dot.up {
    background: #23d5ab;
    box-shadow: 0 0 6px rgba(35,213,171,0.4);
}

.shap-dot.down {
    background: #f2639a;
    box-shadow: 0 0 6px rgba(242,99,154,0.4);
}

.shap-name {
    flex: 1;
    font-size: 0.82rem;
    color: #b0b0c8;
    font-weight: 500;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    min-width: 0;
}

.shap-val {
    font-size: 0.82rem;
    font-weight: 700;
    white-space: nowrap;
}

.shap-val.up   { color: #23d5ab; }
.shap-val.down { color: #f2639a; }

.shap-bar-bg {
    width: 48px;
    height: 4px;
    background: rgba(255,255,255,0.06);
    border-radius: 2px;
    overflow: hidden;
    flex-shrink: 0;
}

.shap-bar-fg {
    height: 100%;
    border-radius: 2px;
}

.shap-bar-fg.up   { background: linear-gradient(90deg, #1cc49e, #23d5ab); }
.shap-bar-fg.down { background: linear-gradient(90deg, #e84393, #f2639a); }

@media (max-width: 768px) {
    .shap-wrap { margin-top: 1.2rem; }
    .shap-bar-bg { display: none; }
    .shap-name { font-size: 0.78rem; }
    .shap-val  { font-size: 0.78rem; }
    .shap-row  { padding: 0.5rem 0.75rem; gap: 0.45rem; }
}

@media (max-width: 480px) {
    .shap-row   { padding: 0.45rem 0.65rem; border-radius: 8px; }
    .shap-name  { font-size: 0.75rem; }
    .shap-val   { font-size: 0.75rem; }
    .shap-title { font-size: 0.7rem; letter-spacing: 1.2px; }
}

/* ── Hero Section ───────────────────────────────────────── */
.hero-section {
    position: relative;
    min-height: 340px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    padding: 2.5rem 1.2rem 2rem;
    background: #0f0f1a;
    background-image: radial-gradient(ellipse 70% 60% at 50% 50%, rgba(35,213,171,0.06) 0%, transparent 70%);
    overflow: hidden;
}

.hero-section.hero-compact {
    min-height: auto;
    padding: 1.8rem 1.2rem 1.4rem;
}

.hero-label {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 2.5px;
    color: #23d5ab;
    text-transform: uppercase;
    margin-bottom: 1.2rem;
    opacity: 0;
    animation: fadeUp 0.6s ease forwards;
    animation-delay: 0s;
}

.hero-heading {
    opacity: 1;
}

.hero-line-1 {
    font-size: clamp(2.2rem, 6vw, 4.5rem);
    font-weight: 300;
    color: #8a8aa0;
    line-height: 1;
    margin: 0;
    opacity: 0;
    animation: fadeUp 0.6s ease forwards;
    animation-delay: 0.1s;
}

.hero-line-2 {
    font-size: clamp(2.2rem, 6vw, 4.5rem);
    font-weight: 900;
    color: #ffffff;
    line-height: 1;
    margin: 0.5rem 0 0 0;
    opacity: 0;
    animation: fadeUp 0.6s ease forwards;
    animation-delay: 0.2s;
}

.hero-subtitle {
    font-size: clamp(0.85rem, 2vw, 1.05rem);
    color: #7c7c90;
    margin-top: 0.5rem;
    max-width: 480px;
    text-align: center;
    line-height: 1.6;
    opacity: 0;
    animation: fadeUp 0.6s ease forwards;
    animation-delay: 0.3s;
}

.hero-pills {
    display: inline-flex;
    align-items: center;
    gap: 1rem;
    margin-top: 1.5rem;
    opacity: 0;
    animation: fadeUp 0.6s ease forwards;
    animation-delay: 0.4s;
}

.hero-pill {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 50px;
    padding: 0.35rem 1rem;
    font-size: 0.78rem;
    color: #a0a0b8;
    display: inline-flex;
    align-items: center;
    white-space: nowrap;
}

.hero-accent {
    width: 120px;
    height: 2px;
    margin: 1.8rem auto 0;
    border-radius: 2px;
    background: linear-gradient(90deg, transparent, #23d5ab, #a463f2, transparent);
    background-size: 200% 100%;
    animation: fadeUp 0.6s ease forwards, shimmer 3s ease infinite;
    animation-delay: 0.5s, 0.5s;
    opacity: 0;
}

@keyframes shimmer {
    0%   { background-position: -200% 0; }
    100% { background-position: 200% 0; }
}

@media (max-width: 768px) {
    .hero-section {
        min-height: 240px;
        padding: 2rem 1rem;
    }
    .hero-section.hero-compact {
        padding: 1.5rem 1rem;
    }
    .hero-line-1,
    .hero-line-2 {
        font-size: clamp(1.8rem, 8vw, 2.8rem);
    }
    .hero-subtitle {
        font-size: 0.82rem;
    }
    .hero-pills {
        flex-direction: column;
        align-items: center;
        gap: 0.5rem;
    }
    .hero-label {
        letter-spacing: 1.5px;
    }
}
</style>
""", unsafe_allow_html=True)


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
function handleCompareLayout() {
    const activeEl = document.getElementById('cmp-active-tab');
    if (!activeEl) return;
    const activeTab = parseInt(activeEl.dataset.tab);
    const isMobile = window.matchMedia('(max-width: 768px)').matches;

    const tabMarkers = document.querySelectorAll('.cmp-tab-marker');
    if (tabMarkers.length > 0) {
        const tabRow = tabMarkers[0].closest('[data-testid="stHorizontalBlock"]');
        if (tabRow) tabRow.style.display = isMobile ? '' : 'none';
    }
    tabMarkers.forEach(marker => {
        const idx = parseInt(marker.dataset.idx);
        const col = marker.closest('[data-testid="stColumn"]');
        if (!col) return;
        const btnDiv = col.querySelector('div.stButton');
        if (btnDiv) {
            btnDiv.classList.add('cmp-tab-btn');
            const btn = btnDiv.querySelector('button');
            if (btn) {
                if (idx === activeTab) btn.classList.add('cmp-tab-active');
                else btn.classList.remove('cmp-tab-active');
            }
        }
    });

    document.querySelectorAll('.cmp-col-marker').forEach(marker => {
        const laptopIdx = parseInt(marker.dataset.laptop);
        const col = marker.closest('[data-testid="stColumn"]');
        if (!col) return;
        col.style.display = (isMobile && laptopIdx !== activeTab) ? 'none' : '';
    });
}

handleCompareLayout();
setTimeout(handleCompareLayout, 300);
setTimeout(handleCompareLayout, 800);
window.addEventListener('resize', handleCompareLayout);

const _lqObs = new MutationObserver(() => {
    requestAnimationFrame(styleLaptiQButtons);
    requestAnimationFrame(handleCompareLayout);
});
_lqObs.observe(document.body, {childList: true, subtree: true});
</script>
""", unsafe_allow_javascript=True)


with st.sidebar:
    active = st.session_state["page"]

    nav_items = [
        ("home",       "Home"),
        ("history",    "History"),
        ("model_info", "Model Info"),
        ("compare",    "Compare"),
    ]

    for key, label in nav_items:
        if st.button(label, key=f"nav_{key}", use_container_width=True):
            st.session_state["page"] = key
            st.rerun()

    st.markdown('<div class="sb-divider">&nbsp;</div>', unsafe_allow_html=True)
    st.markdown(
        '<p style="color:#4a4a5e;font-size:0.7rem;text-align:center;margin-top:0.5rem;">LaptiQ v1.0 · XGBoost</p>',
        unsafe_allow_html=True,
    )


def render_laptop_form(i):

    st.markdown(
        """<div class="fs-wrap"><div class="fs-icon"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path><circle cx="12" cy="7" r="4"></circle></svg></div><div class="fs-title">LAPTOP PROFILE</div></div>""",
        unsafe_allow_html=True,
    )
    brand = st.selectbox("Brand", [
        "Lenovo", "Asus", "HP", "MSI", "Apple", "Acer", "Dell", "Samsung", "Other"
    ], key=f"brand_{i}")
    laptop_type = st.selectbox("Laptop Type", [
        "Gaming", "Notebook", "Business", "Ultrabook",
        "2-in-1 Convertible", "Workstation", "Creator"
    ], key=f"laptop_type_{i}")
    os_val = st.selectbox(
        "Operating System", ["Windows", "macOS", "ChromeOS", "Other"], key=f"os_{i}"
    )
    launch_year = st.selectbox(
        "Launch Year", [2026, 2025, 2024, 2023], key=f"launch_year_{i}"
    )

    st.markdown(
        """<div class="fs-divider"></div><div class="fs-wrap"><div class="fs-icon"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="4" y="4" width="16" height="16" rx="2" ry="2"></rect><rect x="9" y="9" width="6" height="6"></rect><line x1="9" y1="1" x2="9" y2="4"></line><line x1="15" y1="1" x2="15" y2="4"></line><line x1="9" y1="20" x2="9" y2="23"></line><line x1="15" y1="20" x2="15" y2="23"></line><line x1="20" y1="9" x2="23" y2="9"></line><line x1="20" y1="14" x2="23" y2="14"></line><line x1="1" y1="9" x2="4" y2="9"></line><line x1="1" y1="14" x2="4" y2="14"></line></svg></div><div class="fs-title">PROCESSOR</div></div>""",
        unsafe_allow_html=True,
    )
    cpu_brand = st.selectbox(
        "CPU Brand", ["Intel", "AMD", "Apple", "Qualcomm"], key=f"cpu_brand_{i}"
    )
    cpu_model = st.text_input(
        "CPU Model",
        placeholder="e.g. Intel Core i7-13700H",
        help="Full model name — e.g. AMD Ryzen 5 7530U, Apple M3 Pro, Snapdragon X Elite",
        key=f"cpu_model_{i}",
    )
    cpu_cores = st.number_input(
        "CPU Cores", min_value=2, max_value=24, value=8, step=1, key=f"cpu_cores_{i}"
    )

    st.markdown(
        """<div class="fs-divider"></div><div class="fs-wrap"><div class="fs-icon"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="2" y="7" width="20" height="10" rx="2"></rect><path d="M6 10v4"></path><path d="M10 10v4"></path><path d="M14 10v4"></path><path d="M18 10v4"></path></svg></div><div class="fs-title">MEMORY</div></div>""",
        unsafe_allow_html=True,
    )
    ram = st.selectbox(
        "RAM", [8, 16, 24, 32, 48, 64, 96, 128],
        format_func=lambda x: f"{x} GB", key=f"ram_{i}",
    )
    storage_label = st.selectbox(
        "Storage", list(STORAGE_MAP.keys()), key=f"storage_{i}"
    )
    storage_type = st.selectbox(
        "Storage Type", ["SSD", "HDD"], key=f"storage_type_{i}"
    )

    st.markdown(
        """<div class="fs-divider"></div><div class="fs-wrap"><div class="fs-icon"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="2" y="3" width="20" height="14" rx="2" ry="2"></rect><line x1="8" y1="21" x2="16" y2="21"></line><line x1="12" y1="17" x2="12" y2="21"></line></svg></div><div class="fs-title">DISPLAY</div></div>""",
        unsafe_allow_html=True,
    )
    screen_size = st.selectbox(
        "Screen Size",
        [11.6, 13.3, 13.6, 14.0, 15.6, 16.0, 17.3],
        format_func=lambda x: f'{x}"',
        key=f"screen_size_{i}",
    )
    resolution = st.selectbox("Resolution", [
        "1920x1080", "1920x1200", "2560x1600", "2560x1664",
        "2880x1800", "2880x1864", "3024x1964", "3456x2160",
        "3456x2234", "3840x2160"
    ], key=f"resolution_{i}")
    refresh_rate = st.selectbox(
        "Refresh Rate",
        [60, 90, 120, 144, 165, 240, 360],
        format_func=lambda x: f"{x} Hz",
        key=f"refresh_rate_{i}",
    )

    st.markdown(
        """<div class="fs-divider"></div><div class="fs-wrap"><div class="fs-icon"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="2" y="6" width="20" height="12" rx="2"></rect><circle cx="8" cy="12" r="2"></circle><circle cx="16" cy="12" r="2"></circle></svg></div><div class="fs-title">GRAPHICS</div></div>""",
        unsafe_allow_html=True,
    )
    gpu_type = st.selectbox(
        "GPU Type", ["Dedicated", "Integrated"], key=f"gpu_type_{i}"
    )
    gpu_model = st.text_input(
        "GPU Model",
        placeholder="e.g. NVIDIA RTX 4060",
        help="e.g. NVIDIA RTX 4060, Intel Iris Xe, AMD Radeon 780M, Apple M3 GPU",
        key=f"gpu_model_{i}",
    )
    gpu_vram_label = st.selectbox(
        "GPU VRAM",
        ["Shared", "4GB", "6GB", "8GB", "12GB", "16GB", "24GB"],
        help="Pick 'Shared' for integrated graphics with no dedicated VRAM",
        key=f"gpu_vram_{i}",
    )

    st.markdown(
        """<div class="fs-divider"></div><div class="fs-wrap"><div class="fs-icon"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"></path><polyline points="3.27 6.96 12 12.01 20.73 6.96"></polyline><line x1="12" y1="22.08" x2="12" y2="12"></line></svg></div><div class="fs-title">BUILD</div></div>""",
        unsafe_allow_html=True,
    )
    weight = st.number_input(
        "Weight (kg)", min_value=0.8, max_value=4.5, value=1.8,
        step=0.1, format="%.1f", key=f"weight_{i}",
    )

    return {
        "brand": brand,
        "laptop_type": laptop_type,
        "os": os_val,
        "launch_year": launch_year,
        "cpu_brand": cpu_brand,
        "cpu_model": cpu_model,
        "cpu_cores": cpu_cores,
        "ram": ram,
        "storage_label": storage_label,
        "storage_type": storage_type,
        "screen_size": screen_size,
        "resolution": resolution,
        "refresh_rate": refresh_rate,
        "gpu_type": gpu_type,
        "gpu_model": gpu_model,
        "gpu_vram_label": gpu_vram_label,
        "weight": weight,
    }


def predict_laptop(form_data):

    gpu_vram_raw = (
        "Shared"
        if form_data["gpu_vram_label"] == "Shared"
        else form_data["gpu_vram_label"].replace("GB", "").strip()
    )

    input_data = {
        "Brand":        form_data["brand"],
        "Laptop_Type":  form_data["laptop_type"],
        "Launch_Year":  int(form_data["launch_year"]),
        "CPU_Brand":    form_data["cpu_brand"],
        "CPU_Model":    form_data["cpu_model"].strip(),
        "CPU_Cores":    int(form_data["cpu_cores"]),
        "GPU_Type":     form_data["gpu_type"],
        "GPU_VRAM":     gpu_vram_raw,
        "GPU_Model":    form_data["gpu_model"].strip(),
        "Screen_Size":  form_data["screen_size"],
        "Resolution":   form_data["resolution"],
        "Refresh_Rate": form_data["refresh_rate"],
        "RAM":          form_data["ram"],
        "Storage":      STORAGE_MAP[form_data["storage_label"]],
        "Storage_Type": form_data["storage_type"],
        "Weight":       float(form_data["weight"]),
        "OS":           form_data["os"],
    }

    input_df   = pd.DataFrame([input_data])
    processed  = preprocess(input_df)
    encoded_array = model.regressor_.named_steps['pipeline'].transform(processed)
    feature_names = model.regressor_.named_steps['pipeline'].get_feature_names_out()
    top_factors = get_top_factors(explainer, encoded_array, feature_names, top_n=5)


    prediction = model.predict(processed)
    price      = int(round(prediction[0]))

    margin      = price * 0.12
    lower_price = int(round((price - margin) / 500) * 500)
    upper_price = int(round((price + margin) / 500) * 500)

    return (
        price, lower_price, upper_price,
        indian_format(price),
        indian_format(lower_price),
        indian_format(upper_price),
        top_factors,
    )


if st.session_state["page"] == "home":

    st.markdown("""
    <div class="hero-section">
        <div class="hero-label">LaptiQ</div>
        <div class="hero-heading">
            <div class="hero-line-1">Know what your</div>
            <div class="hero-line-2">Laptop is Worth</div>
        </div>
        <div class="hero-subtitle">
            Enter your specs. Get a market price range. Understand why.
        </div>
        <div class="hero-accent">&nbsp;</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""<div class="fs-wrap"><div class="fs-icon"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path><circle cx="12" cy="7" r="4"></circle></svg></div><div class="fs-title">LAPTOP PROFILE</div></div>""", unsafe_allow_html=True)
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

    st.markdown("""<div class="fs-divider"></div><div class="fs-wrap"><div class="fs-icon"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="4" y="4" width="16" height="16" rx="2" ry="2"></rect><rect x="9" y="9" width="6" height="6"></rect><line x1="9" y1="1" x2="9" y2="4"></line><line x1="15" y1="1" x2="15" y2="4"></line><line x1="9" y1="20" x2="9" y2="23"></line><line x1="15" y1="20" x2="15" y2="23"></line><line x1="20" y1="9" x2="23" y2="9"></line><line x1="20" y1="14" x2="23" y2="14"></line><line x1="1" y1="9" x2="4" y2="9"></line><line x1="1" y1="14" x2="4" y2="14"></line></svg></div><div class="fs-title">PROCESSOR</div></div>""", unsafe_allow_html=True)
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

    st.markdown("""<div class="fs-divider"></div><div class="fs-wrap"><div class="fs-icon"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="2" y="7" width="20" height="10" rx="2"></rect><path d="M6 10v4"></path><path d="M10 10v4"></path><path d="M14 10v4"></path><path d="M18 10v4"></path></svg></div><div class="fs-title">MEMORY</div></div>""", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        ram = st.selectbox("RAM", [8, 16, 24, 32, 48, 64, 96, 128], format_func=lambda x: f"{x} GB")
    with c2:
        storage_label = st.selectbox("Storage", list(STORAGE_MAP.keys()))
    with c3:
        storage_type = st.selectbox("Storage Type", ["SSD", "HDD"])

    st.markdown("""<div class="fs-divider"></div><div class="fs-wrap"><div class="fs-icon"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="2" y="3" width="20" height="14" rx="2" ry="2"></rect><line x1="8" y1="21" x2="16" y2="21"></line><line x1="12" y1="17" x2="12" y2="21"></line></svg></div><div class="fs-title">DISPLAY</div></div>""", unsafe_allow_html=True)
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

    st.markdown("""<div class="fs-divider"></div><div class="fs-wrap"><div class="fs-icon"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="2" y="6" width="20" height="12" rx="2"></rect><circle cx="8" cy="12" r="2"></circle><circle cx="16" cy="12" r="2"></circle></svg></div><div class="fs-title">GRAPHICS</div></div>""", unsafe_allow_html=True)
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

    st.markdown("""<div class="fs-divider"></div><div class="fs-wrap"><div class="fs-icon"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"></path><polyline points="3.27 6.96 12 12.01 20.73 6.96"></polyline><line x1="12" y1="22.08" x2="12" y2="12"></line></svg></div><div class="fs-title">BUILD</div></div>""", unsafe_allow_html=True)
    weight = st.number_input("Weight (kg)", min_value=0.8, max_value=4.5, value=1.8, step=0.1, format="%.1f")

    st.markdown("<br>", unsafe_allow_html=True)
    predict_clicked = st.button("Price", type="primary", use_container_width=True)

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

                # SHAP explainability
                encoded_array = model.regressor_.named_steps['pipeline'].transform(processed)
                feature_names = model.regressor_.named_steps['pipeline'].get_feature_names_out()
                top_factors = get_top_factors(explainer, encoded_array, feature_names, top_n=5)
                st.markdown(render_shap_html(top_factors), unsafe_allow_html=True)

            except Exception as e:
                placeholder.empty()
                st.error(f"Prediction failed: {e}")

    st.markdown("""
    <div class="footer-note">
        Predictions are based on training data and current market trends. Actual prices may vary.
    </div>
    """, unsafe_allow_html=True)


elif st.session_state["page"] == "history":

    hist = st.session_state["history"]

    st.markdown("""
    <div class="hero-section hero-compact">
        <div class="hero-heading">
            <div class="hero-line-1">Prediction</div>
            <div class="hero-line-2">History</div>
        </div>
        <div class="hero-subtitle">
            Your recent laptop price predictions — all in one place.
        </div>
    </div>
    """, unsafe_allow_html=True)

    if hist:
        btn_col, _ = st.columns([1, 5])
        with btn_col:
            if st.button("Clear All", key="clear_all"):
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

            st.markdown('<div class="h-separator">&nbsp;</div>', unsafe_allow_html=True)


elif st.session_state["page"] == "model_info":

    st.markdown("""
    <div class="hero-section">
        <div class="hero-heading">
            <div class="hero-line-1">About the</div>
            <div class="hero-line-2">Model Behind It</div>
        </div>
        <div class="hero-subtitle">
            LaptiQ estimates market value from specs — powered by a tuned XGBoost pipeline
            trained on 935 laptops from the Indian market.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="section-title-wrap">
        <div class="section-icon">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="20" x2="18" y2="10"></line><line x1="12" y1="20" x2="12" y2="4"></line><line x1="6" y1="20" x2="6" y2="14"></line></svg>
        </div>
        <div class="section-title">Model Performance</div>
        <div class="section-divider"></div>
    </div>
    """, unsafe_allow_html=True)

    s1, s2, s3 = st.columns(3)
    with s1:
        st.markdown("""
        <div class="sb-stat-new">
            <div class="sb-stat-top">
                <div class="sb-stat-title">R² Accuracy</div>
                <div class="sb-stat-icon blue">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="23 6 13.5 15.5 8.5 10.5 1 18"></polyline><polyline points="17 6 23 6 23 12"></polyline></svg>
                </div>
            </div>
            <div class="sb-stat-val-new blue">93.46%</div>
            <div class="sb-stat-sub">Higher is better</div>
        </div>
        """, unsafe_allow_html=True)
    with s2:
        st.markdown("""
        <div class="sb-stat-new">
            <div class="sb-stat-top">
                <div class="sb-stat-title">Avg Prediction Error</div>
                <div class="sb-stat-icon orange">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><circle cx="12" cy="12" r="6"></circle><circle cx="12" cy="12" r="2"></circle></svg>
                </div>
            </div>
            <div class="sb-stat-val-new orange">11.68%</div>
            <div class="sb-stat-sub">Lower is better</div>
        </div>
        """, unsafe_allow_html=True)
    with s3:
        st.markdown("""
        <div class="sb-stat-new">
            <div class="sb-stat-top">
                <div class="sb-stat-title">Laptops Trained On</div>
                <div class="sb-stat-icon purple">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><ellipse cx="12" cy="5" rx="9" ry="3"></ellipse><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"></path><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"></path></svg>
                </div>
            </div>
            <div class="sb-stat-val-new purple">935</div>
            <div class="sb-stat-sub">Indian market data</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
    <div class="section-title-wrap">
        <div class="section-icon">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"></circle><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"></path></svg>
        </div>
        <div class="section-title">How It Works</div>
        <div class="section-divider"></div>
    </div>
    """, unsafe_allow_html=True)

    steps = [
        ("Enter your laptop specs", "Provide configuration details like CPU, RAM, GPU, storage, and more."),
        ("LaptiQ engineers your input", "We clean, process, and transform your specs for optimal prediction."),
        ("Get accurate market value", "Our tuned XGBoost model predicts the fair market price instantly."),
    ]
    for i, (title, desc) in enumerate(steps, 1):
        st.markdown(f"""
        <div class="mi-step-card">
            <div class="mi-step-num">{i}</div>
            <div class="mi-step-content">
                <div class="mi-step-title">{title}</div>
                <div class="mi-step-desc">{desc}</div>
            </div>
            <div class="mi-step-chevron">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#7c7c90" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="9 18 15 12 9 6"></polyline></svg>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="footer-note">
        Built with XGBoost · sklearn Pipeline · Streamlit
    </div>
    """, unsafe_allow_html=True)


elif st.session_state["page"] == "compare":

    st.markdown("""
    <div class="hero-section hero-compact">
        <div class="hero-heading">
            <div class="hero-line-2">Compare</div>
        </div>
        <div class="hero-subtitle">
            Fill in the specs for each laptop and see which one gives you the best value.
        </div>
        <div class="hero-accent">&nbsp;</div>
    </div>
    """, unsafe_allow_html=True)

    active_tab = st.session_state.get("compare_active_tab", 0)


    st.markdown(
        f'<div id="cmp-active-tab" data-tab="{active_tab}" style="display:none"></div>',
        unsafe_allow_html=True,
    )

    tab_cols = st.columns(3)
    for idx, tc in enumerate(tab_cols):
        with tc:
            st.markdown(
                f'<div class="cmp-tab-marker" data-idx="{idx}" style="display:none"></div>',
                unsafe_allow_html=True,
            )
            if st.button(
                f"Laptop {idx + 1}", key=f"cmp_tab_{idx}", use_container_width=True
            ):
                st.session_state["compare_active_tab"] = idx
                st.rerun()

        cols = st.columns(3)
    forms = []
    for idx in range(3):
        with cols[idx]:
            st.markdown(
                f'<div class="cmp-col-marker" data-laptop="{idx}" style="display:none"></div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="compare-col-header">Laptop {idx + 1}</div>',
                unsafe_allow_html=True,
            )
            form_data = render_laptop_form(idx + 1)
            forms.append(form_data)

        st.markdown("<br>", unsafe_allow_html=True)
    compare_clicked = st.button("Compare Prices  →", type="primary", use_container_width=True)

    if compare_clicked:
        filled = []
        for idx, fd in enumerate(forms):
            if fd["cpu_model"].strip() and fd["gpu_model"].strip():
                filled.append((idx, fd))

        if not filled:
            st.warning(
                "Please fill in CPU Model and GPU Model for at least one laptop."
            )
        else:
            placeholder = st.empty()
            placeholder.markdown("""
            <div style="display:flex;flex-direction:column;align-items:center;padding:1.5rem;">
                <div class="spin-dot"></div>
                <div style="color:#7c7c90;font-size:0.9rem;font-weight:500;">
                    Comparing specs…
                </div>
            </div>
            """, unsafe_allow_html=True)

            time.sleep(0.8)

            results = []
            error_occurred = False
            for laptop_idx, fd in filled:
                try:
                    result = predict_laptop(fd)
                    results.append((laptop_idx, fd, result))
                except Exception as e:
                    placeholder.empty()
                    st.error(f"Prediction failed for Laptop {laptop_idx + 1}: {e}")
                    error_occurred = True
                    break

            if not error_occurred and results:
                placeholder.empty()

                # Identify the laptop with the lowest predicted price
                best_laptop_idx = min(results, key=lambda x: x[2][0])[0]

                result_cols = st.columns(3)
                for (
                    laptop_idx, fd,
                    (price, lower_price, upper_price,
                     formatted, formatted_lower, formatted_upper, top_factors),
                ) in results:
                    with result_cols[laptop_idx]:
                        is_best = (
                            laptop_idx == best_laptop_idx and len(results) > 1
                        )
                        best_class = " best-value" if is_best else ""
                        badge_html = (
                            '<div class="best-value-badge">💰 Best Value</div>'
                            if is_best
                            else ""
                        )

                        st.markdown(f"""<div class="result-wrap"><div class="compare-result-box{best_class}">
<div class="compare-result-header">Laptop {laptop_idx + 1}</div>
<div class="compare-result-sub">{fd['brand']} · {fd['laptop_type']}</div>
<div class="result-label">Estimated Price Range</div>
<div class="result-price">₹{formatted_lower} – ₹{formatted_upper}</div>
<div class="result-subprice">(Most likely around ₹{formatted})</div>
{badge_html}
</div></div>""", unsafe_allow_html=True)
                        st.markdown(render_shap_html(top_factors, compact=True), unsafe_allow_html=True)

    st.markdown("""
    <div class="footer-note">
        Predictions are based on training data and current market trends. Actual prices may vary.
    </div>
    """, unsafe_allow_html=True)