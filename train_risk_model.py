"""
train_risk_model.py  -  PHASE 7  (Machine Learning part)

Trains a model on a TABLE (csv) to predict the screening risk: low / moderate / high.
We compare Random Forest (baseline) and XGBoost and keep the better one.

The csv needs these columns:
    prob_<class>  one column per dental class (the image model's probabilities)
    pain_level (0-10), sensitivity (0/1), gum_bleeding (0/1), swelling (0/1), symptom_days
    risk_label    low / moderate / high   <- must come from a dentist

Run:   python -m training.train_risk_model --csv data/clinical_data.csv
Test:  python -m training.train_risk_model --csv data/clinical_data_DEMO.csv
"""
import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

import config


def main(csv_path):
    config.ensure_dirs()
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise SystemExit(f"File not found: {csv_path}\n"
                         "You need a real, dentist-labelled csv. To only TEST the code, run:\n"
                         "  python -m training.make_demo_risk_data")
    df = pd.read_csv(csv_path)
    prob_cols = [c for c in df.columns if c.startswith("prob_")]
    feature_cols = prob_cols + config.SYMPTOM_FEATURES
    missing = [c for c in feature_cols + [config.RISK_TARGET_COLUMN] if c not in df.columns]
    if not prob_cols or missing:
        raise SystemExit(f"The csv is missing columns: {missing or 'prob_<class> columns'}")

    X = df[feature_cols]
    encoder = LabelEncoder().fit(df[config.RISK_TARGET_COLUMN])
    y = encoder.transform(df[config.RISK_TARGET_COLUMN])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=config.SEED)

    candidates = {
        "random_forest": RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                                random_state=config.SEED),
        "xgboost": XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                                 eval_metric="mlogloss", random_state=config.SEED),
    }
    scores = {}
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        scores[name] = {"macro_f1": float(f1_score(y_test, pred, average="macro")),
                        "macro_recall": float(recall_score(y_test, pred, average="macro"))}
        print(f"\n{name}: macro-F1 = {scores[name]['macro_f1']:.3f}")
        print(classification_report(y_test, pred, target_names=encoder.classes_, zero_division=0))

    best = max(scores, key=lambda n: scores[n]["macro_f1"])
    final = candidates[best].fit(X, y)          # refit the winner on all the data
    joblib.dump({"model": final, "label_encoder": encoder, "feature_columns": feature_cols},
                config.RISK_MODEL_PATH)

    is_demo = "DEMO" in csv_path.name.upper()
    info = {"best_model": best, "scores_on_held_out_20_percent": scores,
            "classes": [c[len("prob_"):] for c in prob_cols],
            "data_source": "SYNTHETIC DEMO DATA - not real clinical evidence" if is_demo
            else f"user-provided file {csv_path.name}",
            "is_demo": is_demo, "n_rows": int(len(df))}
    config.RISK_INFO_PATH.write_text(json.dumps(info, indent=2))
    print(f"\nBest: {best}. Saved models/risk_model.pkl")
    if is_demo:
        print("WARNING: trained on SYNTHETIC data. Use only to test the app.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(config.DATA_DIR / "clinical_data.csv"))
    main(ap.parse_args().csv)
