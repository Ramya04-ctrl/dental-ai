"""
engine.py  -  PHASE 8  (rule-based recommendation engine)

This is NOT machine learning. It is a lookup table of safe, general guidance for
each condition, plus a few safety rules.

*** IMPORTANT ***  The texts below are a DRAFT written for a student prototype.
Before you present or use the project, ask a qualified dentist to review and
correct every sentence in KNOWLEDGE.

The engine never says "you need surgery". It only lists treatments that a
dentist may EVALUATE after examining the patient.
"""
import config

DISCLAIMER = ("This is an AI-generated screening result, not a confirmed diagnosis or prescription. "
              "Treatment decisions must be made by a qualified dentist after a clinical examination.")

KNOWLEDGE = {
    "caries": {
        "explanation": "The image contains an area the model associates with possible tooth decay. "
                       "Only a dentist can confirm whether decay is present and how deep it goes.",
        "pathways": [
            "Early-stage decay: the dentist may consider fluoride or other preventive care.",
            "Cavity needing repair: a dental filling may be considered.",
            "Deep decay reaching the nerve: root canal treatment may be evaluated.",
            "Severely damaged tooth: extraction may be considered only if the dentist decides the tooth cannot be restored.",
        ],
        "home_care": [
            "Brush twice a day with fluoride toothpaste.",
            "Clean between your teeth once a day.",
            "Reduce sugary snacks and drinks between meals.",
            "Avoid chewing hard food on a painful tooth.",
            "Book a dental check-up.",
        ],
        "warning_signs": ["Severe or throbbing tooth pain", "Swelling of the face or gum",
                          "Fever", "Pus or a bad taste in the mouth"],
        "base_urgency": "priority",
    },
    "gingivitis": {
        "explanation": "The image shows features the model associates with inflamed gums (gingivitis). "
                       "A dentist must check the gums to confirm this.",
        "pathways": [
            "Professional cleaning (scaling and polishing) may be considered.",
            "Gum health assessment (gum pocket measurement).",
            "If the gum problem is more advanced, the dentist may evaluate periodontal treatment.",
        ],
        "home_care": [
            "Brush gently twice a day with a soft toothbrush, including along the gum line.",
            "Clean between your teeth once a day.",
            "Do not stop brushing because gums bleed; gentle regular cleaning is needed.",
            "Book a dental check-up and cleaning.",
        ],
        "warning_signs": ["Heavy bleeding that does not stop", "Loose teeth",
                          "Swelling or pus from the gums", "Fever"],
        "base_urgency": "routine",
    },
    "plaque_calculus": {
        "explanation": "The image shows deposits the model associates with plaque or hardened plaque (calculus). "
                       "These collect bacteria and can lead to gum disease and decay.",
        "pathways": ["Professional scaling (cleaning) may be considered.",
                     "Gum health assessment, because calculus often goes with gum problems."],
        "home_care": [
            "Brush twice a day with fluoride toothpaste.",
            "Clean between your teeth once a day.",
            "Do not scrape hard deposits at home with sharp tools; this can hurt gums and teeth.",
            "Book a dental cleaning.",
        ],
        "warning_signs": ["Bleeding or swollen gums that get worse", "Bad breath that does not go away",
                          "Loose teeth"],
        "base_urgency": "routine",
    },
    "fracture": {
        "explanation": "The image shows an area the model associates with a crack or broken tooth. "
                       "Cracks can be hard to see, so a dental examination (and often an X-ray) is needed.",
        "pathways": [
            "Clinical examination to find how deep the damage goes.",
            "Depending on the findings, the dentist may evaluate bonding, a filling, a crown, or other treatment.",
            "If the nerve is affected, root canal assessment may be needed.",
        ],
        "home_care": [
            "Avoid chewing on the affected tooth.",
            "Keep any broken piece and bring it to the dentist.",
            "Avoid very hot, very cold and very hard food.",
            "Book a dental appointment soon.",
        ],
        "warning_signs": ["Severe pain", "Bleeding that does not stop", "Swelling",
                          "Injury to the head or face"],
        "base_urgency": "priority",
    },
    "discoloration": {
        "explanation": "The image shows tooth colour changes. Causes range from surface stains (food, drinks, tobacco) "
                       "to changes inside the tooth, so the cause must be found by a dentist.",
        "pathways": [
            "Examination to find the cause of the colour change.",
            "Surface stains: professional cleaning may be considered.",
            "Whitening or other cosmetic options may be discussed after examination.",
            "A single tooth that is darkening may need further assessment (for example an X-ray).",
        ],
        "home_care": [
            "Brush twice a day and clean between your teeth.",
            "Reduce staining drinks and avoid tobacco.",
            "Do not use harsh scrubbing or unverified whitening products.",
            "Book a dental check-up.",
        ],
        "warning_signs": ["One tooth turning dark after an injury", "Pain with the colour change", "Swelling"],
        "base_urgency": "routine",
    },
    "healthy": {
        "explanation": "The model did not find a visible abnormality in this photo. "
                       "A normal-looking photo cannot rule out hidden problems such as decay between teeth, "
                       "root infection or bone loss.",
        "pathways": ["No specific treatment suggested from this image.",
                     "Regular dental check-ups (and X-rays when the dentist advises)."],
        "home_care": [
            "Brush twice a day with fluoride toothpaste.",
            "Clean between your teeth once a day.",
            "Limit sugary snacks and drinks.",
            "Keep regular dental check-ups.",
        ],
        "warning_signs": ["Tooth pain", "Bleeding or swollen gums", "Swelling of the face"],
        "base_urgency": "routine",
    },
}

GENERIC = {
    "explanation": "The model suggests a possible dental problem. A dentist must examine the patient.",
    "pathways": ["Clinical examination and, if needed, dental X-rays."],
    "home_care": ["Brush twice a day with fluoride toothpaste.", "Clean between your teeth once a day.",
                  "Book a dental check-up."],
    "warning_signs": ["Severe pain", "Facial swelling", "Fever"],
    "base_urgency": "priority",
}

URGENCY_TEXT = {
    "routine": ("Routine dental consultation", "Book a normal dental appointment in the coming weeks."),
    "priority": ("Priority dental consultation", "Try to see a dentist within the next few days."),
    "urgent": ("URGENT - seek care today", "Contact a dentist or emergency service today."),
}
ORDER = ["routine", "priority", "urgent"]


def _raise(level, to):
    return to if ORDER.index(to) > ORDER.index(level) else level


def build_recommendation(condition, confidence, risk_category, symptoms, uncertain=False):
    """
    condition      : class name, e.g. "caries"
    confidence     : 0-1
    risk_category  : "low" / "moderate" / "high"
    symptoms       : dict from the patient form (see app) - needs the red-flag keys below
    uncertain      : True when the image model was not confident (low probability)
    """
    info = KNOWLEDGE.get(condition, GENERIC)
    level = info["base_urgency"]

    # a higher risk category, strong pain, or an unsure model can raise the urgency
    if risk_category in ("moderate", "high"):
        level = _raise(level, "priority")
    if symptoms.get("pain_level", 0) >= 7:
        level = _raise(level, "priority")
    explanation = info["explanation"]
    if uncertain:
        level = _raise(level, "priority")
        explanation = ("NOTE: the model is NOT confident about this image, so this result is unreliable. "
                       "Please see a dentist. " + explanation)

    # SAFETY RULE: red flags always mean urgent care, whatever the model says
    red_flags = []
    if symptoms.get("swelling"):
        red_flags.append("facial or gum swelling")
    if symptoms.get("fever"):
        red_flags.append("fever")
    if symptoms.get("breathing_trouble"):
        red_flags.append("difficulty swallowing or breathing")
    urgent_message = None
    if red_flags:
        level = "urgent"
        urgent_message = ("You reported " + ", ".join(red_flags) + ". This can be a sign of a serious "
                          "infection. Please contact a dentist or emergency service today. "
                          "If you have trouble breathing or swallowing, go to an emergency department now.")

    title, text = URGENCY_TEXT[level]
    return {
        "condition": condition,
        "condition_name": config.display_name(condition),
        "confidence": confidence,
        "risk_category": risk_category,
        "explanation": explanation,
        "pathways": info["pathways"],
        "home_care": info["home_care"],
        "warning_signs": info["warning_signs"],
        "urgency_level": level,
        "urgency_title": title,
        "urgency_text": text,
        "urgent_message": urgent_message,
        "disclaimer": DISCLAIMER,
    }
