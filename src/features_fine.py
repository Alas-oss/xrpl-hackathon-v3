"""Fine-grained timing + link-shape features.

Why: in the training data the planted scenarios move money within seconds to minutes (positives
forward after ~1 minute, look-alike decoys after 8 minutes to an hour) but the existing windows
are 10 min / 1 h / 24 h. These add second-scale windows, the exact gap and amount ratio to the
next / previous hop of a payment trail, and the size / time span / account count of the trail.

Use via run_stack.py --extra features_fine (no config change, no cache problem).
"""
import numpy as np
import pandas as pd
from config import ID_COL
from windows import EventIndex
from stacking2 import Ctx

FINE = (30, 60, 120, 300)
CAP = 7200

def build(base):
    c = Ctx(base)
    s, r, t, v = c.S, c.R, c.T, c.V
    sends, recvs = EventIndex(s, t, v), EventIndex(r, t, v)
    f = pd.DataFrame(index=base[ID_COL])
    for w in FINE:
        f[f"fine_s_sends_pm{w}s"] = sends.count(s, t - w, t + w) - 1        # self excluded
        f[f"fine_r_recvs_pm{w}s"] = recvs.count(r, t - w, t + w) - 1
        f[f"fine_r_fwd_n_{w}s"] = sends.count(r, t, t + w)                   # receiver sends on
        f[f"fine_s_got_n_{w}s"] = recvs.count(s, t - w, t)                   # sender just received
    f["fine_r_fwd_similar_120s"] = sends.similar(r, t, t + 120, v, 0.10).astype(float)
    f["fine_s_got_similar_120s"] = recvs.similar(s, t - 120, t, v, 0.10).astype(float)

    # exact trail links (same hop definition as features_chain / stacking2)
    nxt, prv = c.nxt, c.prv
    f["fine_gap_next_hop"] = np.where(nxt >= 0, t[np.maximum(nxt, 0)] - t, CAP)
    f["fine_gap_prev_hop"] = np.where(prv >= 0, t - t[np.maximum(prv, 0)], CAP)
    f["fine_ratio_next_hop"] = np.where(nxt >= 0, v[np.maximum(nxt, 0)] / np.maximum(v, 1e-9), np.nan)
    f["fine_ratio_prev_hop"] = np.where(prv >= 0, v / np.maximum(v[np.maximum(prv, 0)], 1e-9), np.nan)

    # the whole trail (connected component of hops): size, time span, distinct accounts
    comp = c.comp
    tmin = pd.Series(t).groupby(comp).transform("min").to_numpy()
    tmax = pd.Series(t).groupby(comp).transform("max").to_numpy()
    acc = pd.Series(np.concatenate([s, r])).groupby(np.concatenate([comp, comp])).nunique()
    f["fine_trail_size"] = c.comp_size.astype(float)
    f["fine_trail_span_s"] = tmax - tmin
    f["fine_trail_accounts"] = pd.Series(comp).map(acc).to_numpy().astype(float)
    return f
