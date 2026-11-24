"""Train heart-disease models, track every run in MLflow, save the best one.

Examples
    python train.py --n_estimators 100 --max_depth 6          # one random-forest run
    python train.py --model logreg --C 1.0                    # one logistic-regression run
    python train.py --sweep                                   # 9 runs, 3 algorithms
    mlflow ui --backend-store-uri sqlite:///mlflow.db

The best model is chosen by 5-fold cross-validated F1 on the TRAINING split, not by
the test split, so the 61 test rows are only used for the final report.
"""
import argparse
import json
import warnings
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from preprocessing import FEATURE_BOUNDS, FEATURES, TARGET, build_pipeline

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "model.joblib"
META_PATH = BASE_DIR / "model" / "meta.json"
TRACKING_URI = f"sqlite:///{BASE_DIR / 'mlflow.db'}"
EXPERIMENT = "heart-disease-risk"

SWEEP = [
    {"model": "logreg", "C": 0.1},
    {"model": "logreg", "C": 1.0},
    {"model": "logreg", "C": 10.0},
    {"model": "rf", "n_estimators": 100, "max_depth": 4},
    {"model": "rf", "n_estimators": 300, "max_depth": 6},
    {"model": "rf", "n_estimators": 300, "max_depth": 0},
    {"model": "gb", "n_estimators": 100, "learning_rate": 0.1, "max_depth": 2},
    {"model": "gb", "n_estimators": 200, "learning_rate": 0.05, "max_depth": 3},
    {"model": "gb", "n_estimators": 100, "learning_rate": 0.1, "max_depth": 3},
]


DEFAULTS = {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.1, "C": 1.0}
HP_KEYS = {"logreg": ["C"], "rf": ["n_estimators", "max_depth"],
           "gb": ["n_estimators", "learning_rate", "max_depth"]}


def hp_string(cfg):
    return ", ".join(f"{k}={cfg[k]}" for k in HP_KEYS[cfg["model"]])


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=str(BASE_DIR / "data" / "heart.csv"))
    p.add_argument("--model", choices=["logreg", "rf", "gb"], default="rf")
    p.add_argument("--n_estimators", type=int, default=100)
    p.add_argument("--max_depth", type=int, default=6, help="0 means unlimited (random forest)")
    p.add_argument("--learning_rate", type=float, default=0.1)
    p.add_argument("--C", type=float, default=1.0, help="inverse regularisation (logreg)")
    p.add_argument("--k_features", type=int, default=15)
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--disease_class", type=int, choices=[0, 1], default=None,
                   help="which target value means 'has heart disease' (auto-detected if omitted)")
    p.add_argument("--sweep", action="store_true", help="run the predefined 9-run comparison")
    p.add_argument("--no_save", action="store_true", help="track in MLflow but do not touch model/")
    p.add_argument("--force_save", action="store_true", help="overwrite model/ even if not better")
    return p.parse_args()


def load_data(path):
    df = pd.read_csv(path)
    n_raw = len(df)
    # This Kaggle file repeats the same patients many times. Duplicates would land in both the
    # train and the test split and inflate every score, so each patient is kept once.
    df = df.drop_duplicates().reset_index(drop=True)
    if len(df) < n_raw:
        print(f"Dropped {n_raw - len(df)} duplicate rows ({n_raw} -> {len(df)} unique patients).")
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise SystemExit(f"{path} is missing columns: {missing}")
    if df[FEATURES + [TARGET]].isnull().any().any():
        raise SystemExit("Dataset has missing values; impute after the split or clean the CSV first.")
    # The API rejects anything outside FEATURE_BOUNDS, so the data must live inside them.
    for col, (lo, hi) in FEATURE_BOUNDS.items():
        bad = df[(df[col] < lo) | (df[col] > hi)]
        if len(bad):
            raise SystemExit(f"{len(bad)} rows have '{col}' outside the contract range [{lo}, {hi}]. "
                             "Widen FEATURE_BOUNDS in preprocessing.py and the schema in app.py.")
    return df


def detect_disease_class(df):
    """Which target value means 'disease'? Some copies of this dataset have it flipped.

    Higher oldpeak, exercise angina and more blocked vessels (ca) go with disease; a
    higher maximum heart rate goes with a healthy heart. The sign of that combined
    correlation tells us which label is the disease label.
    """
    t = df[TARGET]
    score = (t.corr(df["oldpeak"]) + t.corr(df["exang"]) + t.corr(df["ca"]) - t.corr(df["thalach"]))
    return 1 if score > 0 else 0


def make_estimator(cfg, seed):
    if cfg["model"] == "logreg":
        return LogisticRegression(C=cfg["C"], solver="liblinear", max_iter=1000, random_state=seed)
    if cfg["model"] == "rf":
        depth = cfg["max_depth"] or None
        return RandomForestClassifier(n_estimators=cfg["n_estimators"], max_depth=depth,
                                      random_state=seed, n_jobs=-1)
    return GradientBoostingClassifier(n_estimators=cfg["n_estimators"], learning_rate=cfg["learning_rate"],
                                      max_depth=cfg["max_depth"], random_state=seed)


def config_from_args(args):
    return {"model": args.model, **{k: getattr(args, k) for k in HP_KEYS[args.model]}}


def run_one(cfg, args, splits, disease_class):
    X_train, X_test, y_train, y_test = splits
    cfg = {"model": cfg["model"], **{k: cfg.get(k, DEFAULTS[k]) for k in HP_KEYS[cfg["model"]]}}
    name = f"{cfg['model']} {hp_string(cfg)}"

    pipe = build_pipeline(make_estimator(cfg, args.seed), k_features=args.k_features)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    cv_f1 = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1")

    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)[:, list(pipe.classes_).index(1)]
    metrics = {
        "cv_f1_mean": float(cv_f1.mean()), "cv_f1_std": float(cv_f1.std()),
        "accuracy": accuracy_score(y_test, pred), "precision": precision_score(y_test, pred),
        "recall": recall_score(y_test, pred), "f1": f1_score(y_test, pred),
        "roc_auc": roc_auc_score(y_test, proba),
    }
    with mlflow.start_run(run_name=name):
        mlflow.log_params({**cfg, "k_features": args.k_features, "seed": args.seed, "disease_class": disease_class, "n_train": len(X_train), "n_test": len(X_test)})
        mlflow.log_metrics(metrics)
    print(f"[{name}] cv_f1={metrics['cv_f1_mean']:.3f}  test acc={metrics['accuracy']:.3f}  f1={metrics['f1']:.3f}")
    return {"name": name, "cfg": cfg, "metrics": metrics, "pipe": pipe}


def maybe_save(result, args, disease_class):
    if args.no_save:
        return
    best_cv = -1.0
    if META_PATH.exists() and not args.force_save:
        best_cv = json.loads(META_PATH.read_text()).get("cv_f1_mean", -1.0)
    if result["metrics"]["cv_f1_mean"] > best_cv or args.force_save:
        MODEL_PATH.parent.mkdir(exist_ok=True)
        joblib.dump(result["pipe"], MODEL_PATH)
        META_PATH.write_text(json.dumps({
            "run": result["name"], "disease_class": disease_class, "features": FEATURES,
            **result["metrics"],
        }, indent=2))
        print(f"Saved {MODEL_PATH.name}  <- {result['name']}")
    else:
        print(f"Kept existing model (cv_f1 {best_cv:.3f} >= {result['metrics']['cv_f1_mean']:.3f})")


def main():
    args = parse_args()
    warnings.filterwarnings("ignore", category=UserWarning)
    df = load_data(args.data)
    disease_class = args.disease_class if args.disease_class is not None else detect_disease_class(df)
    print(f"{len(df)} rows. target={disease_class} is treated as 'heart disease'. "
          "CHECK this against your dataset description.")

    splits = train_test_split(df[FEATURES], df[TARGET], test_size=args.test_size,
                              random_state=args.seed, stratify=df[TARGET])
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)

    configs = SWEEP if args.sweep else [config_from_args(args)]
    results = [run_one(c, args, splits, disease_class) for c in configs]
    best = max(results, key=lambda r: r["metrics"]["cv_f1_mean"])
    maybe_save(best, args, disease_class)

    if args.sweep:  # markdown table for the README
        print("\n| Run | Algorithm | Hyperparameters | CV F1 | Accuracy | F1 | ROC-AUC |")
        print("|---|---|---|---|---|---|---|")
        for i, r in enumerate(results, 1):
            hp = hp_string(r["cfg"])
            m = r["metrics"]
            print(f"| {i} | {r['cfg']['model']} | {hp} | {m['cv_f1_mean']:.3f} | "
                  f"{m['accuracy']:.3f} | {m['f1']:.3f} | {m['roc_auc']:.3f} |")


if __name__ == "__main__":
    main()
