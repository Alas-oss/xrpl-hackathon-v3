"""Precision / recall / F1 at different cut-offs, plus AP and ROC-AUC, from the saved out-of-fold scores.

    python3 src/metrics_report.py                 # uses the 'ens' column of data/processed/oof_stack2.csv
    python3 src/metrics_report.py --col pass2     # any other column

Your scores are ranks, so a "threshold" is a cut-off in the ranking: review the top k% of transactions
and call those suspicious. The portal uses k = 1%. F1 depends on the cut-off; AP does not (it is the
score that counts).
"""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

FRACS = (0.005, 0.01, 0.02, 0.03, 0.05, 0.10)

def curve(y, score):
    """precision, recall, F1 after reviewing the top-k rows, for every k."""
    ys = y[np.argsort(-score, kind="stable")]
    tp = np.cumsum(ys)
    k = np.arange(1, len(y) + 1)
    p, r = tp / k, tp / max(y.sum(), 1)
    f1 = np.divide(2 * p * r, p + r, out=np.zeros(len(y)), where=(p + r) > 0)
    return p, r, f1

def report(y, score, diff=None):
    n = len(y)
    p, r, f1 = curve(y, score)
    lines = [f"rows {n:,}, suspicious {y.sum():,} ({y.mean():.2%})",
             f"AP*100 = {100 * average_precision_score(y, score):.2f}   ROC-AUC = {roc_auc_score(y, score):.3f}   "
             f"(random guessing AP = {100 * y.mean():.1f})", "",
             "review the top...   rows   precision   recall     F1"]
    for fr in FRACS:
        k = int(round(fr * n))
        lines.append(f"  {fr:>6.1%}          {k:6,d}   {p[k - 1]:8.3f}  {r[k - 1]:7.3f}  {f1[k - 1]:6.3f}"
                     + ("   <- the portal's cut-off" if fr == 0.01 else ""))
    kb = int(np.argmax(f1))
    lines += ["", f"best F1 = {f1[kb]:.3f} when reviewing the top {kb + 1:,} rows ({(kb + 1) / n:.1%}): "
                  f"precision {p[kb]:.3f}, recall {r[kb]:.3f}",
              f"ceiling at the 1% cut-off: recall can never pass {int(round(0.01 * n)) / max(y.sum(), 1):.1%}, "
              f"so F1 there cannot pass {2 * (int(round(0.01 * n)) / y.sum()) / (1 + int(round(0.01 * n)) / y.sum()):.3f}"]
    if diff is not None:
        pct = pd.Series(score).rank(pct=True).to_numpy()
        lines += ["", "share of each kind of suspicious row found within the top 1% / 3% / 10%:"]
        for lvl in ("easy", "medium", "hard"):
            m = (diff == lvl) & (y == 1)
            if m.any():
                lines.append(f"  {lvl:6s} ({m.sum():4d}):  " + "   ".join(
                    f"{(pct[m] >= 1 - f).mean():5.0%}" for f in (0.01, 0.03, 0.10)))
    return "\n".join(lines)

def main():
    from config import ID_COL, LABEL_COL, TRAIN_DIR, PROC
    ap = argparse.ArgumentParser()
    ap.add_argument("--col", default="ens")
    a = ap.parse_args()
    oof = pd.read_csv(PROC / "oof_stack2.csv").set_index(ID_COL)
    lab = pd.read_csv(TRAIN_DIR / "labels.csv").set_index(ID_COL).loc[oof.index]
    print(f"scores: column '{a.col}' of oof_stack2.csv (out-of-fold, so this is a fair estimate)\n")
    print(report(lab[LABEL_COL].to_numpy().astype(int), oof[a.col].to_numpy(), lab.difficulty.to_numpy()))

if __name__ == "__main__":
    main()
