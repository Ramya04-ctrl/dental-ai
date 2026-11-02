"""
evaluate.py  -  PHASE 6

Tests the best model on the TEST set (images the model has never seen).
Creates:
  reports/classification_report.txt   precision, recall, F1 for every class
  reports/confusion_matrix.png        where the model makes mistakes
  reports/test_summary.json           overall numbers

For a medical screening tool, RECALL matters a lot: recall = "of all the real
caries images, how many did the model find?". Low recall = missed patients.

Run:   python -m evaluation.evaluate
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score)
from sklearn.preprocessing import label_binarize
from tensorflow import keras

import config
from training.train_dl import load_split, predict_dataset


def main():
    if not config.BEST_MODEL_PATH.exists():
        raise SystemExit("No trained model found. Run:  python -m training.train_dl")
    info = json.loads(config.MODEL_INFO_PATH.read_text())
    class_names = info["class_names"]
    model = keras.models.load_model(config.BEST_MODEL_PATH)
    test_ds = load_split("test", class_names, shuffle=False)

    y_true, y_pred, probs = predict_dataset(model, test_ds)
    labels = list(range(len(class_names)))

    report = classification_report(y_true, y_pred, labels=labels, target_names=class_names,
                                   zero_division=0)
    print(report)
    (config.REPORTS_DIR / "classification_report.txt").write_text(report)

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(cm, display_labels=class_names).plot(ax=ax, xticks_rotation=45, colorbar=False)
    ax.set_title(f"Confusion matrix - {info['best_model']} (test set)")
    fig.tight_layout()
    fig.savefig(config.REPORTS_DIR / "confusion_matrix.png", dpi=130)
    plt.close(fig)

    auc = None
    try:
        auc = float(roc_auc_score(label_binarize(y_true, classes=labels), probs,
                                  multi_class="ovr", average="macro"))
    except ValueError:
        print("ROC-AUC could not be computed (a class is missing in the test set).")

    summary = {"model": info["best_model"], "test_images": int(len(y_true)),
               "accuracy": float(accuracy_score(y_true, y_pred)), "roc_auc_macro": auc}
    (config.REPORTS_DIR / "test_summary.json").write_text(json.dumps(summary, indent=2))
    print("Summary:", summary)

    recalls = np.diag(cm) / np.maximum(cm.sum(axis=1), 1)
    for name, r in zip(class_names, recalls):
        if r < 0.7:
            print(f"WARNING: recall for '{name}' is only {r:.0%}. The model misses many real cases.")
    print("\nSaved reports in:", config.REPORTS_DIR)
    print("Remember: do not claim the system is reliable from accuracy alone.")


if __name__ == "__main__":
    main()
