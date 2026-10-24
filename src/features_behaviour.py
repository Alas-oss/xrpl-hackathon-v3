"""Account behaviour + timing + flow features (computed WITHIN one split's week).

Ideas: relays/mules forward what they receive quickly; fan-in/out bursts; layering
(similar amount in then out); aggregate flows over longer windows for the hard cases.
"""
import numpy as np
import pandas as pd
from config import ID_COL
from windows import EventIndex

M10, H1, H6, H24 = 600, 3600, 6 * 3600, 86400

def build(base):
    s_all = pd.concat([base.sender, base.receiver])
    codes, _ = pd.factorize(s_all)
    n = len(base)
    s, r = codes[:n], codes[n:]
    t = base.ts.to_numpy()
    v = base.dst_usd.to_numpy()
    sends = EventIndex(s, t, v)
    recvs = EventIndex(r, t, v)
    f = pd.DataFrame(index=base[ID_COL])

    def per_acct(acct, col, how):
        ser = pd.Series(v if col == "v" else np.ones(n)).groupby(acct).agg(how)
        return ser

    # --- whole-week activity of sender / receiver -------------------------------------
    out_n = pd.Series(s).value_counts(); in_n = pd.Series(r).value_counts()
    out_usd = pd.Series(v).groupby(s).sum(); in_usd = pd.Series(v).groupby(r).sum()
    out_uniq = pd.Series(r).groupby(s).nunique(); in_uniq = pd.Series(s).groupby(r).nunique()
    out_med = pd.Series(v).groupby(s).median(); in_med = pd.Series(v).groupby(r).median()
    for side, a in (("s", s), ("r", r)):
        f[f"{side}_out_n"] = pd.Series(a).map(out_n).fillna(0).to_numpy()
        f[f"{side}_in_n"] = pd.Series(a).map(in_n).fillna(0).to_numpy()
        f[f"{side}_out_uniq"] = pd.Series(a).map(out_uniq).fillna(0).to_numpy()
        f[f"{side}_in_uniq"] = pd.Series(a).map(in_uniq).fillna(0).to_numpy()
        o, i = pd.Series(a).map(out_usd).fillna(0).to_numpy(), pd.Series(a).map(in_usd).fillna(0).to_numpy()
        f[f"{side}_pass_through"] = np.clip(o / np.where(i > 0, i, np.nan), 0, 5)
        f[f"{side}_log_out_usd"] = np.log1p(o); f[f"{side}_log_in_usd"] = np.log1p(i)
    f["v_over_sender_median"] = v / pd.Series(s).map(out_med).to_numpy()
    f["v_over_receiver_in_median"] = v / pd.Series(r).map(in_med).to_numpy()

    # --- bursts (fan-out from sender, fan-in to receiver), self excluded ---------------
    for name, w in (("10m", M10), ("1h", H1), ("24h", H24)):
        f[f"s_sends_pm{name}"] = sends.count(s, t - w, t + w) - 1
        f[f"r_recvs_pm{name}"] = recvs.count(r, t - w, t + w) - 1

    # --- relay / layering: what the receiver does next, what the sender did before -----
    for name, w in (("1h", H1), ("6h", H6), ("24h", H24)):
        r_next_n = sends.count(r, t, t + w)
        r_next_usd = sends.total(r, t, t + w)
        s_prev_n = recvs.count(s, t - w, t)
        s_prev_usd = recvs.total(s, t - w, t)
        f[f"r_fwd_n_{name}"] = r_next_n
        f[f"r_fwd_usd_ratio_{name}"] = np.clip(r_next_usd / np.maximum(v, 1e-9), 0, 20)
        f[f"s_got_n_{name}"] = s_prev_n
        f[f"s_got_usd_ratio_{name}"] = np.clip(s_prev_usd / np.maximum(v, 1e-9), 0, 20)
    for tol in (0.02, 0.10):
        for name, w in (("1h", H1), ("24h", H24)):
            f[f"r_fwd_similar_{name}_{int(tol*100)}"] = sends.similar(r, t, t + w, v, tol).astype(float)
            f[f"s_got_similar_{name}_{int(tol*100)}"] = recvs.similar(s, t - w, t, v, tol).astype(float)

    # --- timing gaps (seconds, capped at 24h) ------------------------------------------
    f["s_gap_since_recv"] = recvs.gap_before(s, t, H24)
    f["r_gap_to_next_send"] = sends.gap_after(r, t, H24)
    f["s_gap_since_send"] = sends.gap_before(s, t, H24)
    f["s_gap_to_next_send"] = sends.gap_after(s, t, H24)
    f["r_gap_since_recv"] = recvs.gap_before(r, t, H24)
    return f
