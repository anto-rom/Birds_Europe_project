# src/train_classifier_lr.py
# ------------------------------------------------------------
# FINAL TRAINING SCRIPT (Fast baseline for deadline)
# - Reads YAMNet .npz embeddings (emb_mean + emb_std + n_frames)
# - Builds X (N, 2049) + y labels
# - Caches X/y to .npy to avoid reloading 92k files every run
# - Trains Logistic Regression (saga, multinomial) for fast training
# - Saves model + LabelEncoder
# - Reports Macro/Weighted metrics + Top-5 accuracy
# - Provides helper to predict Top-5 from a single .npz
# ------------------------------------------------------------

from __future__ import annotations

import sys
print("PYTHON USED:", sys.executable)
import argparse
import os
from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, top_k_accuracy_score
from sklearn.linear_model import LogisticRegression

import joblib


# -----------------------------
# CONFIG DEFAULTS
# -----------------------------
DEFAULT_EMB_DIR = r"C:\Projects\Xeno_Canto_Project\artifacts\embeddings_yamnet"
DEFAULT_ARTIFACTS_DIR = r"C:\Projects\Xeno_Canto_Project\artifacts\model_artifacts"

DEFAULT_X_CACHE = "X_2049.npy"
DEFAULT_Y_CACHE = "y.npy"
DEFAULT_LABELS_CACHE = "label_encoder.joblib"
DEFAULT_MODEL_CACHE = "lr_multinomial.joblib"

RANDOM_SEED = 42


# -----------------------------
# FEATURE LOADING
# -----------------------------
def npz_to_feature(npz_path: Path) -> Tuple[np.ndarray, str]:
    """
    Loads one .npz and returns (x_features[2049], species_label)
    Expected keys: emb_mean, emb_std, embeddings, species
    """
    d = np.load(npz_path, allow_pickle=True)

    emb_mean = d["emb_mean"].astype(np.float32)      # (1024,)
    emb_std = d["emb_std"].astype(np.float32)        # (1024,)
    n_frames = np.array([d["embeddings"].shape[0]], dtype=np.float32)  # (1,)

    x = np.concatenate([emb_mean, emb_std, n_frames])  # (2049,)
    y = str(d["species"])
    return x, y


def build_dataset_from_npz(emb_dir: Path, limit: int | None = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reads all .npz under emb_dir and returns X (N, 2049), y (N,)
    """
    paths = list(emb_dir.rglob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"No .npz files found under: {emb_dir}")

    if limit is not None:
        paths = paths[:limit]

    X_list: List[np.ndarray] = []
    y_list: List[str] = []

    for i, p in enumerate(paths, 1):
        x, y = npz_to_feature(p)
        X_list.append(x)
        y_list.append(y)

        # Small progress log every 10k
        if i % 10000 == 0:
            print(f"Loaded {i}/{len(paths)} npz...")

    X = np.vstack(X_list)  # (N, 2049)
    y = np.array(y_list, dtype=object)

    return X, y


def load_or_build_cached_dataset(
    emb_dir: Path,
    artifacts_dir: Path,
    force_rebuild: bool = False,
    limit: int | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads cached X/y if present; otherwise builds and caches them.
    Uses mmap for X to keep RAM stable.
    """
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    x_path = artifacts_dir / DEFAULT_X_CACHE
    y_path = artifacts_dir / DEFAULT_Y_CACHE

    if (not force_rebuild) and x_path.exists() and y_path.exists():
        print("✅ Loading cached dataset...")
        X = np.load(x_path, mmap_mode="r")
        y = np.load(y_path, allow_pickle=True)
        print("X shape:", X.shape, "| y len:", len(y))
        return X, y

    print("🔄 Building dataset from NPZ (first time, may take a while)...")
    X, y = build_dataset_from_npz(emb_dir, limit=limit)

    print("💾 Caching dataset to:", artifacts_dir)
    np.save(x_path, X.astype(np.float32))
    np.save(y_path, y)

    # Reload X as mmap for stability
    X = np.load(x_path, mmap_mode="r")
    y = np.load(y_path, allow_pickle=True)
    print("✅ Cached. X shape:", X.shape, "| y len:", len(y))
    return X, y


# -----------------------------
# TRAINING
# -----------------------------
def train_lr_multinomial(
    X: np.ndarray,
    y: np.ndarray,
    artifacts_dir: Path,
    test_size: float = 0.2,
    val_split: float = 0.5,  # split of the temp set -> half val, half test
) -> None:
    """
    Trains Logistic Regressionand saves artifacts. 
    """
    # Encode labels
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # Split train / (val+test)
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y_enc, test_size=test_size, random_state=RANDOM_SEED, stratify=y_enc
    )

    # Split temp into val/test
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=val_split, random_state=RANDOM_SEED, stratify=y_tmp
    )

    print("Train size:", X_train.shape, "Val size:", X_val.shape, "Test size:", X_test.shape)
    print("Num classes:", len(le.classes_))

    # Model (fast + probabilistic)
    clf = LogisticRegression(
        max_iter=200,
        n_jobs=-1,
        solver="saga",
        verbose=0
    )

    print("🚀 Training LogisticRegression (saga, multinomial)...")
    clf.fit(X_train, y_train)

    print("📊 Evaluating...")
    proba = clf.predict_proba(X_test)
    pred = proba.argmax(axis=1)

    report = classification_report(y_test, pred, target_names=le.classes_, digits=3)
    top5 = top_k_accuracy_score(y_test, proba, k=5)

    print(report)
    print("Top-5 accuracy:", float(top5))

    # Save artifacts
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifacts_dir / DEFAULT_MODEL_CACHE
    labels_path = artifacts_dir / DEFAULT_LABELS_CACHE

    joblib.dump(clf, model_path)
    joblib.dump(le, labels_path)

    # Save metrics as txt
    metrics_path = artifacts_dir / "metrics.txt"
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write(report)
        f.write("\n")
        f.write(f"Top-5 accuracy: {float(top5)}\n")

    print("✅ Saved model:", model_path)
    print("✅ Saved labels:", labels_path)
    print("✅ Saved metrics:", metrics_path)


# -----------------------------
# INFERENCE (DEMO)
# -----------------------------
def load_model_and_labels(artifacts_dir: Path):
    model_path = artifacts_dir / DEFAULT_MODEL_CACHE
    labels_path = artifacts_dir / DEFAULT_LABELS_CACHE

    if not model_path.exists() or not labels_path.exists():
        raise FileNotFoundError(
            f"Missing model artifacts. Expected:\n- {model_path}\n- {labels_path}\nRun train first."
        )

    clf = joblib.load(model_path)
    le = joblib.load(labels_path)
    return clf, le


def predict_top5_from_npz(npz_path: Path, artifacts_dir: Path):
    clf, le = load_model_and_labels(artifacts_dir)
    x, _ = npz_to_feature(npz_path)
    proba = clf.predict_proba(x.reshape(1, -1))[0]
    top5_idx = proba.argsort()[-5:][::-1]
    return [(le.classes_[i], float(proba[i])) for i in top5_idx]


# -----------------------------
# CLI
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Train fast bird classifier on YAMNet embeddings (LogReg).")
    p.add_argument("--emb_dir", type=str, default=DEFAULT_EMB_DIR, help="Directory with .npz embeddings.")
    p.add_argument("--artifacts_dir", type=str, default=DEFAULT_ARTIFACTS_DIR, help="Where to save caches and model.")
    p.add_argument("--force_rebuild", action="store_true", help="Force rebuild X/y cache from NPZ.")
    p.add_argument("--limit", type=int, default=None, help="Limit number of NPZ files (debug).")
    p.add_argument("--mode", choices=["train", "predict"], default="train", help="train or predict")
    p.add_argument("--npz_path", type=str, default=None, help="For predict mode: path to one .npz file.")
    return p.parse_args()


def main():
    args = parse_args()

    emb_dir = Path(args.emb_dir)
    artifacts_dir = Path(args.artifacts_dir)

    if args.mode == "train":
        X, y = load_or_build_cached_dataset(
            emb_dir=emb_dir,
            artifacts_dir=artifacts_dir,
            force_rebuild=args.force_rebuild,
            limit=args.limit
        )
        train_lr_multinomial(X, y, artifacts_dir=artifacts_dir)

    elif args.mode == "predict":
        if not args.npz_path:
            raise ValueError("--npz_path is required in predict mode.")
        preds = predict_top5_from_npz(Path(args.npz_path), artifacts_dir)
        print("Top-5 predictions:")
        for label, score in preds:
            print(f"{label:40s}  {score:.4f}")


if __name__ == "__main__":
    main()
