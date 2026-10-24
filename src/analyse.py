"""Understand what the model is doing and where to improve it.

    python3 run_pipeline.py          # first (creates the cached features + out-of-fold scores)
    python3 src/analyse.py

Prints: A) where the model succeeds/fails by scenario family, B) which single features
separate suspicious from normal, C) which feature GROUPS matter (ablation), D) what the
missed MEDIUM cases look like versus normal traffic, E) what the model wrongly ranks on top.
"""
import importlib
import sys
import warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score as AP
from config import ID_COL, LABEL_COL, FEATURE_MODULES, TRAIN_DIR, PROC
sys.path.insert(0, str(Path(__file__).parents[1]))
from run_pipeline import features_for
from model import make_folds, cross_validate

pd.set_option("display.width", 200)

def main():
    base, F = features_for(TRAIN_DIR, "train", False)
    lab = pd.read_csv(TRAIN_DIR / "labels.csv").set_index(ID_COL).loc[F.index]
    y = lab[LABEL_COL].to_numpy().astype(int)
    sender = base.set_index(ID_COL).loc[F.index].sender
    groups = lab.scenario_id.fillna("sender_" + sender).to_numpy()
    folds = make_folds(F.to_numpy(), y, groups)

    oof_path = PROC / "oof_train.csv"
    if oof_path.exists():
        oof = pd.read_csv(oof_path).set_index(ID_COL).loc[F.index, "oof"].to_numpy()
    else:
        oof = cross_validate(F.to_numpy(), y, folds)
    pct = pd.Series(oof).rank(pct=True).to_numpy()          # 1.0 = ranked most suspicious
    top1 = pct >= 1 - 402 / len(y)
    print(f"\nOverall grouped-CV AP*100 = {100 * AP(y, oof):.1f}\n")

    # ---- A) by family / difficulty ---------------------------------------------------
    print("A) HOW EACH SCENARIO TYPE IS DOING (positives only)")
    pos = lab.assign(pct=pct, top1=top1)[y == 1]
    tab = pos.groupby(["difficulty", "family"]).agg(
        n=("pct", "size"), median_rank_pct=("pct", "median"), in_top1pct=("top1", "sum"),
        found_top10pct=("pct", lambda s: (s >= 0.9).mean()))
    print(tab.round(2).to_string(), "\n")

    # ---- B) single-feature power -----------------------------------------------------
    print("B) BEST SINGLE FEATURES (AP*100 if used alone as the score; baseline = %.1f)" % (100 * y.mean()))
    diff = lab.difficulty.to_numpy()
    rows = []
    med = (diff == "medium") | (diff == "normal")
    for c in F.columns:
        x = F[c].to_numpy()
        a_all = max(AP(y, x), AP(y, -x))
        a_med = max(AP(y[med], x[med]), AP(y[med], -x[med]))
        sign = "+" if AP(y, x) >= AP(y, -x) else "-"
        rows.append((c, sign, 100 * a_all, 100 * a_med))
    t = pd.DataFrame(rows, columns=["feature", "higher=", "AP_all", "AP_medium"]).set_index("feature")
    print("top 12 overall:\n", t.sort_values("AP_all", ascending=False).head(12).round(1).to_string())
    print("\ntop 12 for MEDIUM cases:\n", t.sort_values("AP_medium", ascending=False).head(12).round(1).to_string(), "\n")

    # ---- C) feature-group ablation ---------------------------------------------------
    print("C) DOES EACH FEATURE GROUP HELP? (grouped CV AP*100 with the group removed)")
    cols_by_mod = {m: list(importlib.import_module(m).build(base).columns) for m in FEATURE_MODULES}
    full = 100 * AP(y, cross_validate(F.to_numpy(), y, folds))   # single-stage, so the comparison is like-for-like
    print(f"   all features (single-stage model): {full:.1f}")
    for m, cols in cols_by_mod.items():
        keep = [c for c in F.columns if c not in cols]
        o = cross_validate(F[keep].to_numpy(), y, folds)
        print(f"   without {m:20s} ({len(cols):2d} features): {100 * AP(y, o):.1f}  (change {100 * AP(y, o) - full:+.1f})")
    print()

    # ---- D) what medium positives look like vs normal --------------------------------
    print("D) HOW MEDIUM POSITIVES DIFFER FROM NORMAL TRAFFIC (standardised mean difference)")
    m_pos = (diff == "medium")
    norm = (diff == "normal")
    smd = ((F[m_pos].mean() - F[norm].mean()) / F[norm].std().replace(0, np.nan)).dropna()
    print(smd.reindex(smd.abs().sort_values(ascending=False).index).head(12).round(2).to_string())
    print("   (|value| > 0.5 = clearly different; the model is probably not using these well enough)\n")

    # ---- E) false positives ----------------------------------------------------------
    print("E) WHAT THE MODEL RANKS IN ITS TOP 1% BUT IS NOT SUSPICIOUS")
    fp = lab.assign(top1=top1)[(y == 0) & top1]
    print(f"   {len(fp)} of the top 402 are false positives; by family:\n",
          fp.family.value_counts().to_string())

    # ---- F) the real bottleneck: look-alike decoys inside the scenario families ------
    print("\nF) CAN WE TELL SUSPICIOUS FROM THEIR LOOK-ALIKE DECOYS? (same scenario families)")
    fam = lab.family.to_numpy()
    scen = ~np.isin(fam, ["background", "aggregate_relay", "distributed_relay", "ambiguous"])
    for lvl in ("easy", "medium"):
        m = scen & ((diff == lvl) | (diff == "normal"))
        print(f"   {lvl:6s}: AP*100 = {100 * AP(y[m], oof[m]):5.1f}   (chance = {100 * y[m].mean():.1f})")
    pos_m, dec = scen & (diff == "medium"), scen & (diff == "normal")
    smd2 = ((F[pos_m].mean() - F[dec].mean()) / F[dec].std().replace(0, np.nan)).dropna()
    print("   features that differ most, MEDIUM positives vs decoys (standardised mean diff):")
    print(smd2.reindex(smd2.abs().sort_values(ascending=False).index).head(8).round(2).to_string())
    print("   AP below 'chance' = the features cannot separate them; new features must target THIS gap.")

if __name__ == "__main__":
    main()
