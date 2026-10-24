"""Rerank the top band of the stacked model.

    python3 run_stack.py --cv-only            # first: writes data/processed/oof_stack2.csv
    python3 run_rerank.py                     # CV: does reranking the top band help?
    python3 run_stack.py --no-cv              # writes outputs/predictions.csv (the stacked scores)
    python3 run_rerank.py --predict           # -> outputs/predictions_rerank.csv  (upload this one if CV says so)

Options: --frac 0.10 (band size), --w 0.5 (blend weight; default = best in CV), --col ens (which OOF column).
"""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score as AP
from config import ID_COL, LABEL_COL, TRAIN_DIR, VALID_DIR, TEST_DIR, PROC, OUT
from run_pipeline import features_for
from model import precision_at_top
from check_submission import check
import rerank as rr_mod

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frac", type=float, default=0.10)
    ap.add_argument("--w", type=float, default=None)
    ap.add_argument("--col", default=None)
    ap.add_argument("--test", default=None)
    ap.add_argument("--predict", action="store_true")
    a = ap.parse_args()

    base, F = features_for(TRAIN_DIR, "train", False)
    lab = pd.read_csv(TRAIN_DIR / "labels.csv").set_index(ID_COL).loc[F.index]
    y = lab[LABEL_COL].to_numpy().astype(int)
    X = F.replace([np.inf, -np.inf], np.nan).to_numpy()
    groups = lab.scenario_id.fillna("sender_" + base.set_index(ID_COL).loc[F.index].sender).to_numpy()
    oof = pd.read_csv(PROC / "oof_stack2.csv").set_index(ID_COL).loc[F.index]
    col = a.col or ("ens" if "ens" in oof.columns else [c for c in oof.columns if c.startswith("pass")][-1])
    score = oof[col].to_numpy()
    print(f"main-model OOF column: {col}   AP*100 = {100 * AP(y, score):.2f}   precision@1% = {precision_at_top(y, score):.3f}")

    band = rr_mod.band_mask(score, a.frac)
    print(f"top {a.frac:.0%} band = {band.sum():,} rows; holds {y[band].sum()} of {y.sum()} positives "
          f"({y[band].mean():.1%} of the band is suspicious)")
    diff = lab.difficulty.to_numpy()
    for lvl in ("easy", "medium", "hard"):
        m = diff == lvl
        print(f"   {lvl:6s}: {band[m].mean():.0%} of these positives are inside the band")

    band, rr, finals = rr_mod.rerank_cv(X, y, groups, score, frac=a.frac)
    print(f"\nreranker CV (grouped by scenario)     AP*100   precision@1%")
    best_w, best_ap = None, -1
    for w, f in finals.items():
        ap_w = 100 * AP(y, f)
        print(f"   blend weight w = {w:4.2f}            {ap_w:6.2f}   {precision_at_top(y, f):.3f}"
              + ("   (= main model order)" if w == 0 else ""))
        if ap_w > best_ap:
            best_w, best_ap = w, ap_w
    gain = best_ap - 100 * AP(y, score)
    print(f"\nbest w = {best_w}: {gain:+.2f} AP vs main model. "
          f"Within the band alone: AP {100 * AP(y[band], score[band]):.1f} -> {100 * AP(y[band], rr):.1f}")
    if gain < 1.0:
        print("Gain under 1 point is inside the fold-to-fold noise: do not trust it unless it is "
              "positive in the per-weight list above for several w.")

    if a.predict:
        test_dir = Path(a.test) if a.test else (TEST_DIR if (TEST_DIR / "transactions.parquet").exists() else VALID_DIR)
        _, Ft = features_for(test_dir, test_dir.name, False)
        Ft = Ft[F.columns]
        s_te = pd.read_csv(OUT / "predictions.csv").set_index(ID_COL).loc[Ft.index, "score"].to_numpy()
        w = best_w if a.w is None else a.w
        final = rr_mod.rerank_fit_predict(X, y, score, Ft.replace([np.inf, -np.inf], np.nan).to_numpy(),
                                          s_te, frac=a.frac, w=w)
        out = OUT / "predictions_rerank.csv"
        pd.DataFrame({ID_COL: Ft.index, "score": rr_mod.rank01(final)}).to_csv(out, index=False)
        print(f"reranked test predictions written with w={w}")
        check(out, expected_rows=len(Ft))

if __name__ == "__main__":
    main()
