import argparse
import importlib
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score as AP
from config import (ID_COL, LABEL_COL, FEATURE_MODULES, TRAIN_DIR, VALID_DIR, TEST_DIR, PROC, OUT)
import base_table
from model import make_folds, precision_at_top
from stacking import stacked_cv, stacked_fit_predict, stage2_names
from check_submission import check

def features_for(directory, name, refresh):
    """Base table + all feature modules for ONE split (features never mix splits)."""
    key = "_".join(m.replace("features_", "") for m in FEATURE_MODULES)
    cache = PROC / f"features_{name}_{key}.parquet"
    base = base_table.load_split(directory)
    base_table.sanity(base, name)
    if cache.exists() and not refresh:
        F = pd.read_parquet(cache)
        if list(F.index) == list(base[ID_COL]):
            return base, F
    F = pd.concat([importlib.import_module(m).build(base) for m in FEATURE_MODULES], axis=1)
    PROC.mkdir(parents=True, exist_ok=True)
    F.to_parquet(cache)
    return base, F

def report(y, diff, scores, tag):
    print(f"{tag}: AP*100 = {100 * AP(y, scores):.2f}   | precision in top 1%: {precision_at_top(y, scores):.3f}")
    for lvl in ("easy", "medium", "hard"):
        m = (diff == lvl) | (diff == "normal")
        print(f"     {lvl:6s} positives vs all normals: AP*100 = {100 * AP(y[m], scores[m]):.1f}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default=str(TRAIN_DIR))
    ap.add_argument("--test", default=None)
    ap.add_argument("--no-cv", action="store_true")
    ap.add_argument("--refresh", action="store_true")
    a = ap.parse_args()
    test_dir = Path(a.test) if a.test else (TEST_DIR if (TEST_DIR / "transactions.parquet").exists() else VALID_DIR)
    if test_dir == VALID_DIR:
        print("!! No test-data/ found: using the UNLABELLED validation split as a stand-in. "
              "Do not upload this CSV.")

    base, F = features_for(a.train, "train", a.refresh)
    lab = pd.read_csv(Path(a.train) / "labels.csv").set_index(ID_COL).loc[F.index]
    y = lab[LABEL_COL].to_numpy().astype(int)
    X = F.to_numpy()
    groups = lab.scenario_id.fillna("sender_" + base.set_index(ID_COL).loc[F.index].sender).to_numpy()
    print(f"train: {X.shape[0]:,} tx x {X.shape[1]} features, {y.mean():.2%} suspicious")

    if not a.no_cv:
        print("grouped CV (nested, leak-free), about 2 minutes:")
        o1, o2 = stacked_cv(X, y, groups, base, make_folds(X, y, groups))
        report(y, lab.difficulty.to_numpy(), o1, "STAGE 1 (single model)")
        report(y, lab.difficulty.to_numpy(), o2, "STAGE 2 (with neighbour scores)  <- what we submit")
        pd.DataFrame({ID_COL: F.index, "oof": o2}).to_csv(PROC / "oof_train.csv", index=False)

    base_te, Ft = features_for(test_dir, test_dir.name, a.refresh)
    Ft = Ft[F.columns]
    scores, model = stacked_fit_predict(X, y, groups, Ft.to_numpy(), base, base_te)
    OUT.mkdir(exist_ok=True)
    pd.DataFrame({ID_COL: Ft.index, "score": scores}).to_csv(OUT / "predictions.csv", index=False)
    imp = pd.Series(model.booster_.feature_importance("gain"), index=stage2_names(F.columns)).sort_values(ascending=False)
    print("top features (gain share):\n", (imp / imp.sum()).head(10).round(3).to_string())
    check(OUT / "predictions.csv", expected_rows=len(Ft))

if __name__ == "__main__":
    main()
