"""Bonus: simple drift check. Compares logged API inputs with the training data.

    python drift_check.py                       # uses data/heart.csv and logs/predictions.jsonl
    python drift_check.py --log logs/predictions.jsonl --data data/heart.csv

A feature is flagged when the mean of the logged inputs moved more than 3 standard
errors away from the training mean, or the standard deviation changed by more than 2x.
If many features are flagged, collect new data and retrain.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from preprocessing import FEATURES

BASE_DIR = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=str(BASE_DIR / "data" / "heart.csv"))
    p.add_argument("--log", default=str(BASE_DIR / "logs" / "predictions.jsonl"))
    p.add_argument("--min_rows", type=int, default=20)
    args = p.parse_args()

    train = pd.read_csv(args.data).drop_duplicates()[FEATURES]
    lines = [json.loads(line) for line in Path(args.log).read_text().splitlines() if line.strip()]
    live = pd.DataFrame([r["input"] for r in lines])[FEATURES]
    n = len(live)
    if n < args.min_rows:
        print(f"Only {n} logged predictions (need {args.min_rows}); the check would be unreliable.")
        return

    print(f"{n} logged predictions vs {len(train)} training rows\n")
    print(f"{'feature':<10}{'train mean':>12}{'live mean':>12}{'train std':>11}{'live std':>10}  status")
    flagged = 0
    for col in FEATURES:
        tm, ts = train[col].mean(), train[col].std()
        lm, ls = live[col].mean(), live[col].std()
        z = abs(lm - tm) / (ts / np.sqrt(n)) if ts > 0 else 0.0
        ratio = (ls / ts) if ts > 0 else 1.0
        drift = z > 3 or ratio > 2 or ratio < 0.5
        flagged += drift
        print(f"{col:<10}{tm:>12.2f}{lm:>12.2f}{ts:>11.2f}{ls:>10.2f}  {'DRIFT' if drift else 'ok'}")
    print(f"\n{flagged} of {len(FEATURES)} features flagged.")


if __name__ == "__main__":
    main()
