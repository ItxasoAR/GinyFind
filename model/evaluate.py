#!/usr/bin/env python3

import argparse
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
)


def load_model(model_path):
    with open(model_path, "rb") as f:
        data = pickle.load(f)
    return data["model"], data.get("threshold", 0.5)


def load_dataset(path):
    data = np.load(path)
    return data["X"], data["y"]


def compute_metrics(y_true, y_proba, threshold):
    y_pred = (y_proba >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    sensitivity = tp / (tp + fn + 1e-8)   # recall
    specificity = tn / (tn + fp + 1e-8)
    precision   = tp / (tp + fp + 1e-8)
    accuracy    = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    mcc         = matthews_corrcoef(y_true, y_pred)

    roc_auc = roc_auc_score(y_true, y_proba)
    pr_auc  = average_precision_score(y_true, y_proba)

    print(f"Using threshold: {threshold:.3f}")

    return {
        "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "Precision": precision,
        "Accuracy": accuracy,
        "MCC": mcc,
        "ROC_AUC": roc_auc,
        "PR_AUC": pr_auc,
    }, y_pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output_dir", default="evaluation")
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Optional decision threshold override. If omitted, use model threshold."
    )
    args = parser.parse_args()

    X, y = load_dataset(args.dataset)
    model, model_threshold = load_model(args.model)
    threshold = args.threshold if args.threshold is not None else model_threshold

    proba = model.predict_proba(X)[:, 1]
    metrics, y_pred = compute_metrics(y, proba, threshold)



    # ---- Print metrics ----
    print("\n=== TEST RESULTS ===")
    for k, v in metrics.items():
        print(f"{k:15s}: {v:.4f}" if isinstance(v, float) else f"{k:15s}: {v}")

    # ---- Save per-residue predictions (IMPORTANT for R) ----
    df = pd.DataFrame({
        "y_true": y,
        "y_pred": y_pred,
        "y_proba": proba,
    })

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    df.to_csv(f"{args.output_dir}/predictions.csv", index=False)

    # ---- Save metrics ----
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(f"{args.output_dir}/metrics.csv", index=False)


if __name__ == "__main__":
    main()