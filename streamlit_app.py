"""
streamlit_app.py  -  PHASE 9 (web app)

Run from the project folder:   streamlit run app/streamlit_app.py

Streamlit re-runs this whole file every time you click something.
Things we want to remember between clicks are kept in  st.session_state.
"""
import hashlib
import sys
import uuid
from datetime import datetime
from pathlib import Path

# let Python find our own folders (config, utils, database ...)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd          # noqa: E402
import streamlit as st       # noqa: E402

import config                                                       # noqa: E402
from database import db                                             # noqa: E402
from preprocessing.image_validation import validate_image           # noqa: E402
from preprocessing.preprocess import preprocess_for_model           # noqa: E402
from recommendation_engine.engine import build_recommendation       # noqa: E402
from reports.pdf_report import build_pdf                            # noqa: E402
from utils.inference import (dl_model_ready, predict_condition,     # noqa: E402
                             predict_risk, risk_model_info)

st.set_page_config(page_title="Dental Screening AI", page_icon="🦷", layout="wide")

PAGES = ["1. Home Dashboard", "2. Patient Information", "3. Upload Dental Image",
         "4. Image Processing Preview", "5. Dental Condition Prediction", "6. Risk Assessment",
         "7. Treatment Recommendation", "8. Home-Care Guidance", "9. Patient History",
         "10. Download Screening Report"]

DEFAULT_SYMPTOMS = {"pain_level": 0, "sensitivity": 0, "gum_bleeding": 0, "swelling": 0,
                    "symptom_days": 0, "fever": False, "breathing_trouble": False,
                    "age_group": "18-40", "patient_name": ""}

st.markdown("""
<style>
:root { --accent:#0F766E; --soft:#E6F4F1; --line:#D9E2E0; }
.block-container { padding-top: 2rem; max-width: 1100px; }
h1, h2, h3 { color: #0F766E; }
.card { background:#fff; border:1px solid var(--line); border-radius:10px; padding:1rem 1.2rem; margin-bottom:1rem; }
.big { font-size:1.7rem; font-weight:700; color:var(--accent); }
.label { font-size:.72rem; letter-spacing:.12em; text-transform:uppercase; color:#6b7b78; }
.pill { display:inline-block; padding:.2rem .7rem; border-radius:99px; font-weight:600; font-size:.85rem; }
.pill-routine { background:#E6F4F1; color:#0F766E; }
.pill-priority { background:#FFF4DB; color:#8A5A00; }
.pill-urgent { background:#FDE4E4; color:#9B1C1C; }
.disclaimer { font-size:.8rem; color:#6b7b78; border-top:1px solid var(--line); padding-top:.6rem; margin-top:1.2rem; }
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Session helpers
# ----------------------------------------------------------------------------
def init_state():
    st.session_state.setdefault("page", PAGES[0])
    st.session_state.setdefault("symptoms", DEFAULT_SYMPTOMS.copy())
    st.session_state.setdefault("upload", None)
    st.session_state.setdefault("prediction", None)
    st.session_state.setdefault("saved_ref", None)


def go(page):
    st.session_state.page = page


def next_button(label, page):
    st.button(label, on_click=go, args=(page,), type="primary")


def disclaimer():
    from recommendation_engine.engine import DISCLAIMER
    st.markdown(f'<div class="disclaimer">{DISCLAIMER}</div>', unsafe_allow_html=True)


def has_valid_upload():
    up = st.session_state.upload
    return bool(up and up["result"]["ok"])


def get_prediction():
    """Runs the deep-learning model once per uploaded image and remembers the answer."""
    if not has_valid_upload():
        return None
    up = st.session_state.upload
    cached = st.session_state.prediction
    if cached and cached["hash"] == up["hash"]:
        return cached
    with st.spinner("Analysing the image..."):
        pred = predict_condition(up["result"]["image"])
    pred["hash"] = up["hash"]
    st.session_state.prediction = pred
    return pred


def get_results():
    """Returns (prediction, risk, recommendation) or None when something is missing."""
    if not dl_model_ready():
        st.error("No trained image model found yet. Train it first (see the guide, Step 4):  "
                 "`python -m training.train_dl`")
        return None
    pred = get_prediction()
    if pred is None:
        st.info("Please upload a valid image first (page 3).")
        return None
    s = st.session_state.symptoms
    risk = predict_risk(pred["probabilities"], s)
    rec = build_recommendation(pred["class_name"], pred["confidence"], risk["category"], s,
                               uncertain=pred["uncertain"])
    return pred, risk, rec


def urgency_pill(rec):
    return f'<span class="pill pill-{rec["urgency_level"]}">{rec["urgency_title"]}</span>'


# ----------------------------------------------------------------------------
# Pages
# ----------------------------------------------------------------------------
def page_home():
    st.title("🦷 AI Dental Screening and Decision Support")
    st.write("Upload a clear photo of a tooth. The system checks the image, finds a possible dental "
             "condition with **Deep Learning**, estimates a screening risk with **Machine Learning**, "
             "and gives a report with safe next steps.")
    st.warning("This is a **screening prototype**, not a doctor. It cannot confirm a diagnosis or "
               "decide that surgery is needed.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Image model (Deep Learning)", "Ready" if dl_model_ready() else "Not trained yet")
    info = risk_model_info()
    c2.metric("Risk model (Machine Learning)", info["best_model"].replace("_", " ").title() if info
              else "Starter rules only")
    c3.metric("Screenings saved", "off (public demo)" if config.PUBLIC_DEMO else len(db.get_history()))
    st.subheader("How to use")
    st.markdown("1. Fill **Patient Information**  \n2. **Upload** a JPG or PNG photo  \n3. Look at the "
                "**Prediction**, **Risk** and **Treatment** pages  \n4. **Download** the PDF report")
    next_button("Start  →", PAGES[1])
    disclaimer()


def page_patient():
    st.header("Patient Information")
    s = st.session_state.symptoms
    c1, c2 = st.columns(2)
    with c1:
        s["patient_name"] = st.text_input("Name (optional - only printed on the PDF, never saved)",
                                          value=s["patient_name"])
        ages = ["Under 18", "18-40", "41-60", "Over 60"]
        s["age_group"] = st.selectbox("Age group", ages, index=ages.index(s["age_group"]))
        s["pain_level"] = st.slider("Pain level (0 = none, 10 = worst)", 0, 10, s["pain_level"])
        s["symptom_days"] = st.number_input("For how many days have you had symptoms?", 0, 3650,
                                            int(s["symptom_days"]))
    with c2:
        s["sensitivity"] = int(st.checkbox("Tooth sensitivity (hot / cold / sweet)", bool(s["sensitivity"])))
        s["gum_bleeding"] = int(st.checkbox("Gums bleed", bool(s["gum_bleeding"])))
        s["swelling"] = int(st.checkbox("Swelling of the face or gum", bool(s["swelling"])))
        s["fever"] = st.checkbox("Fever", s["fever"])
        s["breathing_trouble"] = st.checkbox("Difficulty swallowing or breathing", s["breathing_trouble"])
    if s["swelling"] or s["fever"] or s["breathing_trouble"]:
        st.error("⚠️ Swelling, fever or trouble swallowing/breathing can be a sign of a serious infection. "
                 "Contact a dentist or emergency service TODAY. Do not wait for this app.")
    next_button("Next: upload image  →", PAGES[2])


def page_upload():
    st.header("Upload Dental Image")
    st.write("Use a **clear, well-lit JPG, JPEG or PNG** photo of the tooth area. Avoid blur and shadows.")
    file = st.file_uploader("Choose an image (JPG, JPEG or PNG)", type=["jpg", "jpeg", "png"])
    if file is not None:
        data = file.getvalue()
        digest = hashlib.md5(data).hexdigest()
        old = st.session_state.upload
        if not old or old["hash"] != digest:           # new file -> check it
            st.session_state.upload = {"hash": digest, "name": file.name,
                                       "result": validate_image(data, file.name),
                                       "ref_id": f"DS-{datetime.now():%Y%m%d}-{uuid.uuid4().hex[:4].upper()}"}
            st.session_state.prediction = None
    up = st.session_state.upload
    if up:
        res = up["result"]
        if res["ok"]:
            st.success(f"Image accepted. Reference ID: **{up['ref_id']}**")
            st.image(res["image"], caption=up["name"], width=360)
            next_button("Next: see processing  →", PAGES[3])
        else:
            for r in res["reasons"]:
                st.error(r)
            if res["image"] is not None:
                st.image(res["image"], caption="Rejected image", width=240)


def page_preview():
    st.header("Image Processing Preview")
    if not has_valid_upload():
        st.info("Please upload a valid image first (page 3).")
        return
    res = st.session_state.upload["result"]
    processed = preprocess_for_model(res["image"])
    c1, c2 = st.columns(2)
    c1.image(res["image"], caption="Original", width="stretch")
    c2.image(processed, caption="After resize to 224x224 + light denoising (this goes to the AI)",
             width="stretch")
    m = res["metrics"]
    st.table(pd.DataFrame({"Check": ["Resolution", "Sharpness score (higher = sharper)",
                                     "Brightness (0-255)", "File size"],
                           "Value": [f"{m['width']} x {m['height']} px", str(m["blur_score"]),
                                     str(m["brightness"]), f"{m['size_mb']} MB"],
                           "Status": ["OK", "OK", "OK", "OK"]}).set_index("Check"))
    next_button("Next: prediction  →", PAGES[4])


def page_prediction():
    st.header("Dental Condition Prediction")
    out = get_results()
    if not out:
        return
    pred, risk, rec = out
    if pred["uncertain"]:
        st.warning("The model is **not confident** about this image "
                   f"(top probability {pred['confidence']:.0%}). Do not rely on this result.")
    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown(f'<div class="card"><div class="label">Possible condition</div>'
                    f'<div class="big">{rec["condition_name"]}</div>'
                    f'<div>Model confidence: <b>{pred["confidence"]:.0%}</b> '
                    f'<span class="label">({pred["model_name"]})</span></div></div>',
                    unsafe_allow_html=True)
        probs = pd.DataFrame({"Probability (%)": {config.display_name(k): round(v * 100, 1)
                                                  for k, v in pred["probabilities"].items()}})
        st.bar_chart(probs)
    with c2:
        st.image(pred["gradcam_overlay"], width="stretch",
                 caption="Grad-CAM: red/yellow = areas that influenced the model most. "
                         "This is an explanation aid, not proof of the disease location.")
    next_button("Next: risk assessment  →", PAGES[5])
    disclaimer()


def page_risk():
    st.header("Risk Assessment")
    out = get_results()
    if not out:
        return
    pred, risk, rec = out
    st.markdown(f'<div class="card"><div class="label">Screening risk category</div>'
                f'<div class="big">{risk["category"].title()}</div>'
                f'<div class="label">Method: {risk["method"]}</div></div>', unsafe_allow_html=True)
    if risk["warning"]:
        st.warning(risk["warning"])
    st.markdown(f'Consultation advice: {urgency_pill(rec)}', unsafe_allow_html=True)
    st.caption("This is a screening category from the image result and the symptoms. "
               "It is not a confirmed severity.")
    next_button("Next: treatment  →", PAGES[6])
    disclaimer()


def page_treatment():
    st.header("Treatment Recommendation")
    out = get_results()
    if not out:
        return
    pred, risk, rec = out
    if rec["urgent_message"]:
        st.error("🚨 " + rec["urgent_message"])
    st.markdown(urgency_pill(rec) + f" &nbsp; {rec['urgency_text']}", unsafe_allow_html=True)
    st.subheader("What might be happening?")
    st.write(rec["explanation"])
    st.subheader("Possible treatment pathways (for a dentist to evaluate)")
    for t in rec["pathways"]:
        st.markdown(f"- {t}")
    st.info("The image alone cannot show whether any treatment is needed. A dentist must examine you "
            "and may take X-rays.")
    st.subheader("Get urgent help if you notice")
    for t in rec["warning_signs"]:
        st.markdown(f"- {t}")
    next_button("Next: home care  →", PAGES[7])
    disclaimer()


def page_homecare():
    st.header("Home-Care Guidance")
    out = get_results()
    if not out:
        return
    _, _, rec = out
    st.write("Safe supportive care. It does **not** replace a visit to the dentist.")
    for t in rec["home_care"]:
        st.markdown(f"- ✅ {t}")
    st.caption("Do not put home remedies or chemicals directly on a tooth or gum.")
    next_button("Next: report  →", PAGES[9])
    disclaimer()


def page_history():
    st.header("Patient History")
    if config.PUBLIC_DEMO:
        st.info("History is switched off in the public demo, so visitors cannot see each other's results.")
        return
    hist = db.get_history()
    if hist.empty:
        st.info("No saved screenings yet. Save one from the last page.")
    else:
        st.dataframe(hist, width="stretch", hide_index=True)
        st.download_button("Download history (CSV)", hist.to_csv(index=False), "screening_history.csv")
    st.caption("Only reference ids, dates and results are stored. No names. No photos.")


def page_report():
    st.header("Download Screening Report")
    out = get_results()
    if not out:
        return
    pred, risk, rec = out
    up = st.session_state.upload
    s = st.session_state.symptoms
    st.write(f"Reference ID: **{up['ref_id']}**  |  Condition: **{rec['condition_name']}**  |  "
             f"Risk: **{risk['category'].title()}**")

    pdf = build_pdf({"ref_id": up["ref_id"], "date": f"{datetime.now():%Y-%m-%d %H:%M}",
                     "patient_name": s["patient_name"], "quality": up["result"]["metrics"],
                     "rec": rec,
                     "risk_method": risk["method"] + (f" | {risk['warning']}" if risk["warning"] else "")},
                    pred["processed_image"], pred["gradcam_overlay"])
    st.download_button("⬇️  Download PDF report", pdf, f"{up['ref_id']}.pdf", "application/pdf",
                       type="primary")

    if config.PUBLIC_DEMO:
        st.caption("Public demo: results are not saved anywhere. Download the PDF if you want to keep them.")
    elif st.session_state.saved_ref == up["ref_id"]:
        st.success("Saved to history.")
    elif st.button("Save this screening to history"):
        m = up["result"]["metrics"]
        db.save_screening({"ref_id": up["ref_id"], "age_group": s["age_group"],
                           "condition": pred["class_name"], "confidence": round(pred["confidence"], 3),
                           "risk": risk["category"], "urgency": rec["urgency_level"],
                           "blur_score": m["blur_score"], "brightness": m["brightness"]})
        st.session_state.saved_ref = up["ref_id"]
        st.rerun()
    disclaimer()


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
init_state()
st.sidebar.title("🦷 Dental Screening")
st.sidebar.radio("Navigate", PAGES, key="page")
if has_valid_upload():
    st.sidebar.success(f"Image: {st.session_state.upload['ref_id']}")
st.sidebar.caption("Screening prototype. Not a medical device.")

{PAGES[0]: page_home, PAGES[1]: page_patient, PAGES[2]: page_upload, PAGES[3]: page_preview,
 PAGES[4]: page_prediction, PAGES[5]: page_risk, PAGES[6]: page_treatment, PAGES[7]: page_homecare,
 PAGES[8]: page_history, PAGES[9]: page_report}[st.session_state.page]()
