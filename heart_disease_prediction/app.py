from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Heart Check", page_icon="🫀", layout="centered")

CP_OPTIONS = ["Typical angina", "Atypical angina", "Non-anginal pain", "No symptoms"]

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;600;700&display=swap');
html, body, [class*="css"], .stApp { font-family: 'IBM Plex Sans', sans-serif; }
.stApp { background: #F3F6F8; color: #12263A; }
.stApp label, .stApp label p, .stApp [data-testid="stWidgetLabel"] p { color: #12263A !important; }
.stApp div[data-baseweb="select"] > div { background: #fff; color: #12263A; }
.stApp div[role="radiogroup"] p { color: #12263A !important; }
#MainMenu, footer, header { visibility: hidden; }
.hero { background: #12263A; color: #fff; border-radius: 14px; padding: 28px 28px 8px; margin-bottom: 22px; }
.hero h1 { margin: 0 0 6px; font-size: 2rem; font-weight: 700; color: #fff; }
.hero p { margin: 0; color: #B8C7D6; max-width: 46ch; line-height: 1.5; }
.hero svg { width: 100%; height: 70px; display: block; margin-top: 10px; }
.trace { stroke: #3DD6C6; stroke-width: 2.5; fill: none; stroke-linecap: round; stroke-linejoin: round;
         stroke-dasharray: 900; stroke-dashoffset: 900; animation: draw 2.4s ease-out forwards; }
@keyframes draw { to { stroke-dashoffset: 0; } }
@media (prefers-reduced-motion: reduce) { .trace { animation: none; stroke-dashoffset: 0; } }
.result { border-radius: 14px; padding: 22px 24px; margin-top: 18px; border-left: 8px solid; background: #fff; }
.result.high { border-color: #C0392B; } .result.low { border-color: #2E7D5B; }
.result h2 { margin: 0 0 4px; font-size: 1.5rem; }
.result.high h2 { color: #C0392B; } .result.low h2 { color: #2E7D5B; }
.result p { margin: 0; color: #4A5D70; }
.note { font-size: .85rem; color: #6B7C8D; margin-top: 18px; line-height: 1.5; }
div.stButton > button { background: #0F8B8D; color: #fff; border: 0; border-radius: 10px;
    padding: 10px 22px; font-weight: 600; width: 100%; }
div.stButton > button:hover { background: #0B6E70; color: #fff; }
div.stButton > button:focus-visible { outline: 3px solid #12263A; outline-offset: 2px; }
</style>
<div class="hero">
  <h1>Heart disease risk check</h1>
  <p>Enter four details from a stress test and see what the model predicts.</p>
  <svg viewBox="0 0 600 70" preserveAspectRatio="none" aria-hidden="true">
    <path class="trace" d="M0 40 H140 l10 -6 10 6 H230 l8 6 12 -38 14 54 10 -22 H360 l12 -10 14 10 H600"/>
  </svg>
</div>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    return joblib.load(Path(__file__).parent / "heart_model.pkl")


model = load_model()

c1, c2 = st.columns(2)
with c1:
    cp = st.selectbox("Chest pain type", CP_OPTIONS)
    exang = st.radio("Chest pain during exercise?", ["No", "Yes"], horizontal=True)
with c2:
    thalach = st.slider("Max heart rate reached (bpm)", 70, 210, 150)
    oldpeak = st.slider("ST depression (oldpeak)", 0.0, 6.5, 1.0, 0.1,
                        help="How far the ECG ST segment dropped during exercise vs. rest.")

if st.button("Check my result"):
    X = pd.DataFrame([{"cp": cp, "thalach": thalach, "exang": exang, "oldpeak": oldpeak}])
    pred = int(model.predict(X)[0])
    prob = float(model.predict_proba(X)[0][1]) * 100
    if pred == 1:
        st.markdown(f'<div class="result high"><h2>Higher risk</h2>'
                    f'<p>The model estimates a {prob:.0f}% chance of heart disease.</p></div>',
                    unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="result low"><h2>Lower risk</h2>'
                    f'<p>The model estimates a {prob:.0f}% chance of heart disease.</p></div>',
                    unsafe_allow_html=True)

st.markdown('<p class="note">Built for a university machine learning assignment using the UCI Heart '
            'Disease dataset. It is not a medical tool. See a doctor for real health concerns.</p>',
            unsafe_allow_html=True)