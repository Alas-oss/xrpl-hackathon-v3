"""Multi-pass neighbour stacking + ensemble  ->  outputs/predictions.csv

    python run_stack.py                          # nested grouped CV on train, then predict test-data/
    python run_stack.py --no-cv                  # skip CV, just predict
    python run_stack.py --nb win,link            # only some neighbour groups (win,pair,prov,link,clu)
    python run_stack.py --passes 2               # 2 = same depth as stacking.py, 3 = one extra pass
    python run_stack.py --cv-only                # report CV, do not write predictions

The CV prints AP*100 for every pass and for each member of the final ensemble, so the
CV (not the leaderboard) decides which setting to submit.
"""
import argparse
import importlib
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score as AP
from config import ID_COL, LABEL_COL, TRAIN_DIR, VALID_DIR, TEST_DIR, PROC, OUT
from run_pipeline import features_for
from model import make_folds, precision_at_top
from check_submission import check
import stacking2 as st2

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default=str(TRAIN_DIR))
    ap.add_argument("--test", default=None)
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--nb", default=",".join(st2.ALL_GROUPS))
    ap.add_argument("--no-cv", action="store_true")
    ap.add_argument("--cv-only", action="store_true")
    ap.add_argument("--no-ens", action="store_true", help="skip HistGradientBoosting / ExtraTrees")
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--extra", default="", help="extra feature modules, e.g. features_fine (not cached)")
    ap.add_argument("--drop", default="", help="feature columns to remove, e.g. hour_sin,hour_cos,is_night")
    a = ap.parse_args()
    extra = [m for m in a.extra.split(",") if m]
    drop = [c for c in a.drop.split(",") if c]
    groups = tuple(g for g in a.nb.split(",") if g)
    assert set(groups) <= set(st2.ALL_GROUPS), f"--nb must be a subset of {st2.ALL_GROUPS}"
    test_dir = Path(a.test) if a.test else (TEST_DIR if (TEST_DIR / "transactions.parquet").exists() else VALID_DIR)
    if test_dir == VALID_DIR and not a.cv_only:
        print("!! No test-data/ found: using the UNLABELLED validation split as a stand-in. Do not upload this CSV.")

    def with_extras(base, F):
        """Append --extra modules and remove --drop columns (same recipe for train and test)."""
        if extra:
            F = pd.concat([F] + [importlib.import_module(m).build(base).set_axis(F.index) for m in extra], axis=1)
        missing = [c for c in drop if c not in F.columns]
        assert not missing, f"--drop names not in the features: {missing}"
        return F.drop(columns=drop)

    base, F = features_for(a.train, "train", a.refresh)
    F = with_extras(base, F)
    lab =pd.read_csv(Path(a.train) / "labels.csv").set_index(ID_COL).loc[F.index]
    y = lab[LABEL_COL].to_numpy().astype(int)
    X = F.to_numpy()
    groups_cv = lab.scenario_id.fillna("sender_" + base.set_index(ID_COL).loc[F.index].sender).to_numpy()
    print(f"train: {X.shape[0]:,} tx x {X.shape[1]} features; neighbour groups: {groups}; passes: {a.passes}")
    t0 = time.time()
    ctx = st2.Ctx(base)
    print(f"context (links, clusters) built in {time.time() - t0:.0f}s; "
          f"{(ctx.nxt >= 0).mean():.1%} of rows have a next hop; largest cluster {ctx.comp_size.max()}")

    if not a.no_cv:
        folds = make_folds(X, y, groups_cv)
        passes = [np.zeros(len(y)) for _ in range(a.passes)]
        parts = {k: np.zeros(len(y)) for k in ("lgb", "hgb", "et", "ens")}
        for k, (tr, va) in enumerate(folds):
            fz = st2.cv_featuriser(X, ctx, tr, va, groups)
            pp, fin = st2.run_passes(X[tr], y[tr], groups_cv[tr], X[va], fz, a.passes, ensemble=not a.no_ens)
            for p, s in zip(passes, pp):
                p[va] = s
            if fin:
                for key in parts:
                    parts[key][va] = fin[key]
            print(f"  fold {k}: " + " -> ".join(f"pass{i + 1} {100 * AP(y[va], p[va]):.1f}"
                                                 for i, p in enumerate(passes))
                  + (f" | ens {100 * AP(y[va], parts['ens'][va]):.1f}" if fin else ""), flush=True)
        print("\nGROUPED-CV (nested, leak-free)   AP*100   precision@1%")
        for i, p in enumerate(passes):
            print(f"  pass {i + 1}                       {100 * AP(y, p):6.2f}   {precision_at_top(y, p):.3f}")
        if not a.no_ens:
            for key, p in parts.items():
                print(f"  final {key:4s}                   {100 * AP(y, p):6.2f}   {precision_at_top(y, p):.3f}")
        PROC.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({ID_COL: F.index, **{f"pass{i + 1}": p for i, p in enumerate(passes)},
                      **({k: v for k, v in parts.items()} if not a.no_ens else {})}
                     ).to_csv(PROC / "oof_stack2.csv", index=False)

    if a.cv_only:
        return
    base_te, Ft = features_for(test_dir, test_dir.name, a.refresh)
    Ft = with_extras(base_te, Ft)[F.columns]
    ctx_te = st2.Ctx(base_te)
    fz = st2.final_featuriser(X, ctx, Ft.to_numpy(), ctx_te, groups)
    pp, fin = st2.run_passes(X, y, groups_cv, Ft.to_numpy(), fz, a.passes, ensemble=not a.no_ens)
    score = fin["ens"] if fin else pd.Series(pp[-1]).rank(pct=True).to_numpy()
    OUT.mkdir(exist_ok=True)
    pd.DataFrame({ID_COL: Ft.index, "score": score}).to_csv(OUT / "predictions.csv", index=False)
    check(OUT / "predictions.csv", expected_rows=len(Ft))

if __name__ == "__main__":
    main()
