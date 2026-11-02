"""
make_demo_risk_data.py  -  *** FOR TESTING THE PIPELINE ONLY ***

The ML risk model needs a table of REAL patients: image-model probabilities +
symptoms + a risk label given by a dentist. You probably do not have it yet.

This script makes FAKE (synthetic) rows, so you can see how the risk model
trains and how the app uses it. The labels come from a simple rule that I
wrote, NOT from dentists. A model trained on this file learns my rule, not
dentistry.  NEVER present its results as real medical evidence.
The file is named ..._DEMO.csv and the app shows a warning when it is used.

Run:   python -m training.make_demo_risk_data
"""
import json

import numpy as np
import pandas as pd

import config


def get_class_names():
    if config.MODEL_INFO_PATH.exists():
        return json.loads(config.MODEL_INFO_PATH.read_text())["class_names"]
    return config.DEFAULT_CLASSES


def main(n_rows=600):
    config.ensure_dirs()
    rng = np.random.default_rng(config.SEED)
    classes = get_class_names()
    probs = rng.dirichlet(np.ones(len(classes)) * 0.4, size=n_rows)
    df = pd.DataFrame(probs, columns=[f"prob_{c}" for c in classes])
    df["pain_level"] = rng.integers(0, 11, n_rows)
    df["sensitivity"] = rng.integers(0, 2, n_rows)
    df["gum_bleeding"] = rng.integers(0, 2, n_rows)
    df["swelling"] = (rng.random(n_rows) < 0.15).astype(int)
    df["symptom_days"] = rng.integers(0, 60, n_rows)

    healthy_prob = df["prob_healthy"] if "prob_healthy" in df else 0
    disease_prob = 1 - healthy_prob
    score = (df["pain_level"] / 10 * 3 + df["sensitivity"] + df["gum_bleeding"]
             + df["swelling"] * 3 + (df["symptom_days"] > 14) * 1 + disease_prob * 2
             + rng.normal(0, 0.5, n_rows))
    df[config.RISK_TARGET_COLUMN] = pd.cut(score, [-99, 3, 6, 99], labels=config.RISK_LABELS).astype(str)

    out = config.DATA_DIR / "clinical_data_DEMO.csv"
    df.to_csv(out, index=False)
    print(f"Saved FAKE demo data -> {out}")
    print(df[config.RISK_TARGET_COLUMN].value_counts().to_string())
    print("\nREMINDER: this is synthetic data for testing only. Replace it with real,")
    print("dentist-labelled data (same column names) for your final project.")


if __name__ == "__main__":
    main()
