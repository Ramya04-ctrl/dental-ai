"""
train_dl.py  -  PHASES 4 and 5

Trains the 3 deep-learning models, compares them on the VALIDATION set,
and saves the best one as  models/best_model.keras

Training has two stages for the pretrained models:
  Stage 1: pretrained part is frozen, only the new final layer learns.
  Stage 2 (fine-tuning): the top layers of the pretrained part also learn, slowly.

Run everything:        python -m training.train_dl
Run one model only:    python -m training.train_dl --models mobilenet
Quick test:            python -m training.train_dl --epochs 2 --fine-tune-epochs 1
"""
import argparse
import json
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras

import config
from training.models import MODEL_NAMES, build_model, unfreeze_top_layers


def get_class_names():
    train_dir = config.DATASET_DIR / "train"
    if not train_dir.exists():
        raise SystemExit("dataset/train not found. Run:  python -m preprocessing.prepare_dataset")
    return sorted(d.name for d in train_dir.iterdir() if d.is_dir())


def load_split(split, class_names, shuffle):
    ds = keras.utils.image_dataset_from_directory(
        config.DATASET_DIR / split,
        labels="inferred", label_mode="int", class_names=class_names,
        image_size=config.IMG_SIZE, batch_size=config.BATCH_SIZE,
        shuffle=shuffle, seed=config.SEED)
    return ds.cache().prefetch(tf.data.AUTOTUNE)


def predict_dataset(model, ds):
    """Returns true labels, predicted labels and probabilities for a dataset (not shuffled)."""
    y_true = np.concatenate([y.numpy() for _, y in ds])
    probs = model.predict(ds, verbose=0)
    return y_true, probs.argmax(axis=1), probs


def class_weights_for(class_names):
    """Classes with fewer images get a bigger weight, so the model does not ignore them."""
    labels = []
    for i, c in enumerate(class_names):
        labels += [i] * len(list((config.DATASET_DIR / "train" / c).glob("*.jpg")))
    present = np.unique(labels)
    weights = compute_class_weight("balanced", classes=present, y=labels)
    cw = {int(c): float(w) for c, w in zip(present, weights)}
    return {i: cw.get(i, 1.0) for i in range(len(class_names))}


def make_callbacks():
    return [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=0),
    ]


def compile_model(model, lr):
    model.compile(optimizer=keras.optimizers.Adam(lr),
                  loss=keras.losses.SparseCategoricalCrossentropy(),
                  metrics=["accuracy"])


def plot_history(history, name):
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    ax[0].plot(history["accuracy"], label="train"); ax[0].plot(history["val_accuracy"], label="validation")
    ax[0].set_title(f"{name} - accuracy"); ax[0].set_xlabel("epoch"); ax[0].legend()
    ax[1].plot(history["loss"], label="train"); ax[1].plot(history["val_loss"], label="validation")
    ax[1].set_title(f"{name} - loss"); ax[1].set_xlabel("epoch"); ax[1].legend()
    fig.tight_layout()
    fig.savefig(config.REPORTS_DIR / f"training_curves_{name}.png", dpi=120)
    plt.close(fig)


def train_one(name, class_names, train_ds, val_ds, cw, epochs, fine_tune_epochs, pretrained):
    print(f"\n{'=' * 60}\nTraining: {name}   (pretrained={pretrained and name != 'custom_cnn'})\n{'=' * 60}")
    model, base = build_model(name, len(class_names), pretrained)
    history = {"accuracy": [], "val_accuracy": [], "loss": [], "val_loss": []}

    def add(h):
        for k in history:
            history[k] += h.history[k]

    use_transfer = base is not None and pretrained
    if use_transfer:
        base.trainable = False                       # Stage 1: freeze the pretrained part
    compile_model(model, 1e-3)
    add(model.fit(train_ds, validation_data=val_ds, epochs=epochs,
                  class_weight=cw, callbacks=make_callbacks(), verbose=2))

    if use_transfer and fine_tune_epochs > 0:        # Stage 2: fine-tune
        print("\nFine-tuning the top layers...")
        unfreeze_top_layers(base, config.FINE_TUNE_LAYERS)
        compile_model(model, 1e-5)
        add(model.fit(train_ds, validation_data=val_ds, epochs=fine_tune_epochs,
                      class_weight=cw, callbacks=make_callbacks(), verbose=2))

    y_true, y_pred, _ = predict_dataset(model, val_ds)
    result = {
        "model": name,
        "val_accuracy": accuracy_score(y_true, y_pred),
        "val_macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "val_macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
    }
    model.save(config.MODELS_DIR / f"{name}.keras")
    plot_history(history, name)
    print(f"Saved models/{name}.keras   val macro-F1 = {result['val_macro_f1']:.3f}")
    return result


def main(models=None, epochs=None, fine_tune_epochs=None, pretrained=None):
    config.ensure_dirs()
    models = models or MODEL_NAMES
    epochs = config.EPOCHS if epochs is None else epochs
    fine_tune_epochs = config.FINE_TUNE_EPOCHS if fine_tune_epochs is None else fine_tune_epochs
    pretrained = config.USE_PRETRAINED if pretrained is None else pretrained

    keras.utils.set_random_seed(config.SEED)
    class_names = get_class_names()
    print("Classes:", class_names)
    train_ds = load_split("train", class_names, shuffle=True)
    val_ds = load_split("val", class_names, shuffle=False)
    cw = class_weights_for(class_names)
    print("Class weights:", {class_names[i]: round(w, 2) for i, w in cw.items()})

    results = [train_one(m, class_names, train_ds, val_ds, cw, epochs, fine_tune_epochs, pretrained)
               for m in models]

    table = pd.DataFrame(results).sort_values("val_macro_f1", ascending=False)
    table.to_csv(config.REPORTS_DIR / "model_comparison.csv", index=False)
    print("\nMODEL COMPARISON (validation set)")
    print(table.round(3).to_string(index=False))

    best = table.iloc[0]["model"]
    shutil.copy(config.MODELS_DIR / f"{best}.keras", config.BEST_MODEL_PATH)
    info = {"best_model": best, "class_names": class_names, "img_size": list(config.IMG_SIZE),
            "pretrained": bool(pretrained), "val_macro_f1": float(table.iloc[0]["val_macro_f1"])}
    config.MODEL_INFO_PATH.write_text(json.dumps(info, indent=2))
    print(f"\nBest model: {best}  ->  saved as models/best_model.keras")
    print("The TEST set has not been used yet. Next run:  python -m evaluation.evaluate")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", choices=MODEL_NAMES)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--fine-tune-epochs", type=int)
    a = ap.parse_args()
    main(a.models, a.epochs, a.fine_tune_epochs)
