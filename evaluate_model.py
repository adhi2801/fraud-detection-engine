"""Score the committed ONNX model (models/fraud_model.onnx) on the exact test split train_model.py holds out.

    pip install pandas numpy scikit-learn onnxruntime
    python evaluate_model.py --csv data/creditcard.csv

Why: only 0.17 % of transactions are fraud, so ROC-AUC alone looks excellent for almost any model. This adds the
precision-recall view (PR-AUC against the 0.0017 base rate), precision / recall / F1 at the 0.5 threshold, the
confusion matrix, and bootstrap 95 % confidence intervals. Writes results/model_eval.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as rt
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).parent


def at_threshold(y: np.ndarray, p: np.ndarray, thr: float) -> dict[str, float]:
    pred = p >= thr
    tp, fp = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum())
    fn, tn = int((~pred & (y == 1)).sum()), int((~pred & (y == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn)
    return {"precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "false_alarms_per_10k_legit": 1e4 * fp / (fp + tn), "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def bootstrap(y: np.ndarray, p: np.ndarray, iters: int = 1000, seed: int = 0) -> dict[str, list[float]]:
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    out = {"roc_auc": [], "pr_auc": [], "precision": [], "recall": []}
    for _ in range(iters):  # stratified resample keeps the fraud count fixed
        i = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        out["roc_auc"].append(roc_auc_score(y[i], p[i]))
        out["pr_auc"].append(average_precision_score(y[i], p[i]))
        m = at_threshold(y[i], p[i], 0.5)
        out["precision"].append(m["precision"])
        out["recall"].append(m["recall"])
    return {k: [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)] for k, v in out.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/creditcard.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    X = df[[f"V{i}" for i in range(1, 29)] + ["Amount", "Time"]].astype(np.float32).to_numpy()
    y = df["Class"].to_numpy()
    # identical to train_model.py: same columns, test_size, seed and stratification
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    sess = rt.InferenceSession(str(ROOT / "models" / "fraud_model.onnx"))
    probs = sess.run(None, {sess.get_inputs()[0].name: X_test})[1]
    p = np.array([row[1] for row in probs]) if isinstance(probs, list) else np.asarray(probs)[:, 1]

    result = {
        "test_split": {"n": int(len(y_test)), "frauds": int(y_test.sum()), "fraud_rate": round(float(y_test.mean()), 5),
                       "how": "train_test_split(test_size=0.2, random_state=42, stratify=y), as in train_model.py"},
        "roc_auc": round(float(roc_auc_score(y_test, p)), 4),
        "pr_auc": round(float(average_precision_score(y_test, p)), 4),
        "at_threshold_0.5": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in at_threshold(y_test, p, 0.5).items()},
        "ci95": bootstrap(y_test, p),
    }
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "model_eval.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
