"""Which features separate suspicious rows from their look-alike decoys, family by family.

    python3 src/family_gap.py

For every scenario family and difficulty level this compares the suspicious rows with the
NOT-suspicious rows of the SAME family (the decoys), and prints the features that separate
them best (single-feature AUC: 0.5 = useless, 1.0 or 0.0 = perfect). Where nothing reaches
about 0.65, the current features cannot tell the two apart and NEW features are needed there.
"""
import sys
import warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parents[1]))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

SKIP_FAMILIES = {"background", "ambiguous"}

def family_gap(F, y, fam, diff, scen_notna=None, top=6, min_n=10):
    """Returns a printable report. F: feature DataFrame; y, fam, diff: aligned arrays."""
    X = F.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())
    out = []
    for level in ("easy", "medium", "hard"):
        for f in sorted(set(fam) - SKIP_FAMILIES):
            pos = (fam == f) & (y == 1) & (diff == level)
            dec = (fam == f) & (y == 0)
            if pos.sum() < min_n or dec.sum() < min_n:
                continue
            m = pos | dec
            auc = pd.Series({c: roc_auc_score(pos[m], X[c].to_numpy()[m]) for c in X.columns})
            best = auc.reindex((auc - 0.5).abs().sort_values(ascending=False).index).head(top)
            head = f"[{level:6s}] {f:18s} positives {pos.sum():4d} vs decoys {dec.sum():5d}"
            if scen_notna is not None:
                head += f"  (decoys with a scenario_id: {scen_notna[dec].mean():.0%})"
            out.append(head + f"   best single-feature |AUC-0.5|+0.5 = {abs(best.iloc[0] - 0.5) + 0.5:.2f}")
            for c, a in best.items():
                mp, md = np.median(X[c].to_numpy()[pos]), np.median(X[c].to_numpy()[dec])
                out.append(f"      {c:28s} AUC {a:.2f}  ({'higher' if a > 0.5 else 'lower '} in positives)  "
                           f"median pos {mp:10.3g}  decoy {md:10.3g}")
    return "\n".join(out)

def main():
    from config import ID_COL, LABEL_COL, TRAIN_DIR
    from run_pipeline import features_for
    base, F = features_for(TRAIN_DIR, "train", False)
    lab = pd.read_csv(TRAIN_DIR / "labels.csv").set_index(ID_COL).loc[F.index]
    print(family_gap(F, lab[LABEL_COL].to_numpy().astype(int), lab.family.to_numpy(),
                     lab.difficulty.to_numpy(), lab.scenario_id.notna().to_numpy()))

if __name__ == "__main__":
    main()
