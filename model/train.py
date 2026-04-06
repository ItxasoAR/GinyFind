"""
model/train.py
---------------
Trains a Random Forest or Gradient Boosting classifier on the pre-built
dataset and serialises the fitted model + scaler to disk.

Handles class imbalance (typically ~5-10% positive residues) with
class_weight='balanced' and outputs a full evaluation report.

Usage
-----
python -m model.train --dataset data/dataset.npz --model RF --output model/trained_model.pkl
python -m model.train --dataset data/dataset.npz --model GBT --output model/trained_model.pkl
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
)
from sklearn.metrics import precision_recall_curve
from features import all_feature_names
from sklearn.utils.class_weight import compute_sample_weight 

def find_best_threshold(y_true, y_proba):
    """
    Find threshold that maximizes F1 score using precision-recall curve.
    """
    prec, rec, thr = precision_recall_curve(y_true, y_proba)
    f1 = 2 * (prec * rec) / (prec + rec + 1e-8)
    best = np.argmax(f1)
    return thr[best]



# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

def build_model(model_type: str = "RF") -> Pipeline:
    """
    Return a sklearn Pipeline with StandardScaler + classifier.
    
    Parameters
    ----------
    model_type : "RF" for Random Forest, "GBT" for Gradient Boosting
    """
    if model_type == "RF":
        clf = RandomForestClassifier(
            n_estimators=500,
            max_depth=None,
            min_samples_leaf=5,
            max_features="sqrt",
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        )
    elif model_type == "GBT":
        clf = GradientBoostingClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            min_samples_leaf=10,
            random_state=42,
        )
    else:
        raise ValueError(f"Unknown model type '{model_type}'. Choose RF or GBT.")

    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", clf),
    ])


# ---------------------------------------------------------------------------
# Training & evaluation
# ---------------------------------------------------------------------------

def load_dataset(npz_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load X and y from a .npz file produced by build_dataset."""
    data = np.load(npz_path, allow_pickle=True)
    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.int8)
    print(
        f"Dataset loaded: {X.shape[0]} residues × {X.shape[1]} features | "
        f"pos={y.sum()} ({100*y.mean():.1f}%)"
    )
    return X, y


def cross_validate_model(pipeline, X, y, n_splits=5):
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    metrics = {"roc_auc": [], "f1": [], "precision": [], "recall": [], "mcc": []}
    oof_proba = np.zeros(len(y))   # out-of-fold probabilities

    print(f"\nRunning {n_splits}-fold cross-validation ...")
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        clone = pickle.loads(pickle.dumps(pipeline))  # fresh copy
        clone.fit(X_tr, y_tr)
        proba = clone.predict_proba(X_val)[:, 1]
        oof_proba[val_idx] = proba

        thr = find_best_threshold(y_val, proba)
        pred = (proba >= thr).astype(int)

        from sklearn.metrics import f1_score, precision_score, recall_score
        metrics["roc_auc"].append(roc_auc_score(y_val, proba))
        metrics["f1"].append(f1_score(y_val, pred))
        metrics["precision"].append(precision_score(y_val, pred))
        metrics["recall"].append(recall_score(y_val, pred))
        metrics["mcc"].append(matthews_corrcoef(y_val, pred))

    print("\n── Cross-Validation Results ──────────────────────────")
    labels = {"roc_auc": "ROC-AUC", "f1": "F1", "precision": "Precision",
              "recall": "Recall", "mcc": "MCC"}
    for key, label in labels.items():
        vals = np.array(metrics[key])
        print(f"  {label:12s}: {vals.mean():.3f} ± {vals.std():.3f}")
    print("──────────────────────────────────────────────────────\n")

    return oof_proba  # <-- now returns OOF probabilities

def feature_importance_report(pipeline: Pipeline, top_n: int = 15) -> None:
    """Print top-N most important features."""
    clf = pipeline.named_steps["clf"]
    if not hasattr(clf, "feature_importances_"):
        return

    names = all_feature_names()
    importances = clf.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]

    print(f"\n── Top {top_n} Feature Importances ──────────────────────")
    for i in range(min(top_n, len(names))):
        idx = sorted_idx[i]
        print(f"  {i+1:2d}. {names[idx]:35s} {importances[idx]:.4f}")
    print("──────────────────────────────────────────────────────\n")


def train(
    dataset_path: str,
    model_type: str = "RF",
    output_path: str = "model/trained_model.pkl",
    run_cv: bool = True,
) -> Pipeline:
    """
    Full training pipeline:
        1. Load dataset
        2. (Optional) cross-validate
        3. Fit on full dataset
        4. Save model
    """
    X, y = load_dataset(dataset_path)
    pipeline = build_model(model_type)

    if run_cv:
        oof_proba = cross_validate_model(pipeline, X, y)
        threshold = find_best_threshold(y, oof_proba)  # honest threshold from unseen data
        print(f"Best threshold (F1-optimal, from OOF): {threshold:.3f}")
    else:
        threshold = 0.5  # fallback if no CV
        print("Skipping CV — using default threshold 0.5")

    print("Fitting model on full dataset ...")
    if model_type == "GBT":
        sample_weights = compute_sample_weight("balanced", y)
        pipeline.fit(X, y, clf__sample_weight=sample_weights)
    else:
        pipeline.fit(X, y)

    y_proba = pipeline.predict_proba(X)[:, 1]   # keep this for the report only

    # Apply threshold
    y_pred = (y_proba >= threshold).astype(int)

    print("\n── Train-set evaluation (optimistic — use CV scores) ──")
    print(classification_report(y, y_pred, target_names=["non-binding", "binding"]))
    print(f"  ROC-AUC : {roc_auc_score(y, y_proba):.3f}")
    print(f"  Avg-Prec: {average_precision_score(y, y_proba):.3f}")
    print(f"  MCC     : {matthews_corrcoef(y, y_pred):.3f}\n")


    feature_importance_report(pipeline)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump({"model": pipeline, "threshold": threshold}, f)
    print(f"✓ Model saved to {output_path}")

    return pipeline


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train ligand binding site predictor.")
    parser.add_argument("--dataset",  required=True,  help="Path to dataset.npz")
    parser.add_argument("--model",    default="RF",    help="RF or GBT (default: RF)")
    parser.add_argument("--output",   default="model/trained_model.pkl",
                        help="Output pickle path")
    parser.add_argument("--no_cv",    action="store_true",
                        help="Skip cross-validation (faster)")
    args = parser.parse_args()

    train(
        dataset_path=args.dataset,
        model_type=args.model,
        output_path=args.output,
        run_cv=not args.no_cv,
    )


if __name__ == "__main__":
    main()
