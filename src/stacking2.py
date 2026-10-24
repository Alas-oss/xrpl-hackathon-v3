"""Collective classification v2: richer neighbour scores, a third pass, and an ensemble.

Builds on stacking.py (same leak-free scheme: training rows carry OUT-OF-FOLD scores, rows
being predicted carry scores from a model that never saw them). What is new:

  1. More neighbour definitions ("nb" groups, switchable for ablation):
       win   sender / receiver windows, now up to 6h and 24h, plus "what the receiver sends next"
       pair  same sender->receiver pair (1h, 24h, 7d)
       prov  same first liquidity provider (10m, 1h, 6h)
       link  EXACT money-trail links: score of the specific next / previous hop (and 2 hops),
             best score further down / up the chain
       clu   cluster = connected component of the chain links: size, mean / max score of the others
  2. N passes (default 3): pass k feeds the out-of-fold scores of pass k-1 back in as neighbours.
  3. Final-stage ensemble: LightGBM seeds + HistGradientBoosting + ExtraTrees, rank-averaged.

All neighbour features are computed inside ONE split's base table (never across splits).
Empty neighbourhoods are encoded as -1 (scores live in [0, 1]).
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.stats import rankdata
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier

from config import LGB_PARAMS
from stacking import _Idx, stage1
from windows import EventIndex

M10, H1, H6, H24, D7 = 600, 3600, 6 * 3600, 86400, 7 * 86400
ALL_GROUPS = ("win", "pair", "prov", "link", "clu")

# (name, index attr, account attr, window start, window end, exclude self)
WIN_SPECS = [
    ("s_send_pm10m", "send", "S", -M10, M10, True), ("s_send_pm1h", "send", "S", -H1, H1, True),
    ("s_send_pm6h", "send", "S", -H6, H6, True), ("s_send_pm24h", "send", "S", -H24, H24, True),
    ("r_recv_pm10m", "recv", "R", -M10, M10, True), ("r_recv_pm1h", "recv", "R", -H1, H1, True),
    ("r_recv_pm6h", "recv", "R", -H6, H6, True),
    ("r_next10m", "send", "R", 0, M10, False), ("r_next1h", "send", "R", 0, H1, False),
    ("r_next6h", "send", "R", 0, H6, False),
    ("s_prev1h", "recv", "S", -H1, 0, False), ("s_prev6h", "recv", "S", -H6, 0, False),
]
PAIR_SPECS = [("pair_pm1h", "pair", "P", -H1, H1, True), ("pair_pm24h", "pair", "P", -H24, H24, True),
              ("pair_pm7d", "pair", "P", -D7, D7, True)]
PROV_SPECS = [("prov_pm10m", "prov", "PR", -M10, M10, True), ("prov_pm1h", "prov", "PR", -H1, H1, True),
              ("prov_pm6h", "prov", "PR", -H6, H6, True)]
LINK_NAMES = ["next", "prev", "next2", "prev2", "fwd_max", "bwd_max"]
CLU_NAMES = ["size", "mean", "max"]


# ------------------------------------------------------------------ per-split context -----
class Ctx:
    """Everything about ONE split's base table that does not depend on scores (build once)."""

    def __init__(self, base, window=7200, tol=0.15):
        n = self.n = len(base)
        codes, _ = pd.factorize(pd.concat([base.sender, base.receiver]))
        self.S, self.R = codes[:n], codes[n:]
        self.T = base.ts.to_numpy().astype(np.int64)
        self.V = base.dst_usd.to_numpy()
        self.P = pd.factorize(pd.Series(self.S.astype(np.int64) * (codes.max() + 1) + self.R))[0]
        pr = pd.factorize(base.provider1)[0]                     # NaN (single-leg) -> -1
        miss = pr < 0                                            # unique code each: no neighbours
        pr[miss] = pr.max() + 1 + np.arange(miss.sum())
        self.PR = pr
        self.idx = {"send": _Idx(self.S, self.T), "recv": _Idx(self.R, self.T),
                    "pair": _Idx(self.P, self.T), "prov": _Idx(self.PR, self.T)}
        self.acct = {"S": self.S, "R": self.R, "P": self.P, "PR": self.PR}
        self.nxt, self.prv = self._links(window, tol)
        m = self.nxt >= 0
        g = coo_matrix((np.ones(m.sum()), (np.flatnonzero(m), self.nxt[m])), shape=(n, n))
        _, self.comp = connected_components(g, directed=False)
        self.comp_size = np.bincount(self.comp)[self.comp]

    def _links(self, window, tol):
        """Same hop definition as features_chain.py: the receiver's next send of a similar amount
        (within `tol`, within `window` seconds); prv is the sender's previous similar receipt.
        nxt[i] > i and prv[i] < i always (rows are sorted by time, links are strictly later/earlier)."""
        n, S, R, T, V = self.n, self.S, self.R, self.T, self.V
        sends, recvs = EventIndex(S, T, V), EventIndex(R, T, V)
        s_order, r_order = np.lexsort((T, S)), np.lexsort((T, R))
        nxt, prv = -np.ones(n, int), -np.ones(n, int)
        lo, hi = sends._rng(R, T + 1, T + window)
        for i in np.flatnonzero(hi > lo):
            ids = s_order[lo[i]:hi[i]]
            ok = np.abs(V[ids] - V[i]) <= tol * V[i]
            if ok.any():
                nxt[i] = ids[np.argmax(ok)]
        lo, hi = recvs._rng(S, T - window, T - 1)
        for i in np.flatnonzero(hi > lo):
            ids = r_order[lo[i]:hi[i]][::-1]
            ok = np.abs(V[ids] - V[i]) <= tol * V[i]
            if ok.any():
                prv[i] = ids[np.argmax(ok)]
        return nxt, prv


# ------------------------------------------------------------------ vectorised helpers ----
def _sparse_table(sc):
    tbl, j = [sc], 1
    while 2 * j <= len(sc):
        prev = tbl[-1]
        tbl.append(np.maximum(prev[:-j], prev[j:]))
        j *= 2
    return tbl

def _range_max(tbl, lo, hi):
    """max of sc[lo:hi] for each (lo, hi); 0 where the range is empty."""
    out = np.zeros(len(lo))
    ln = hi - lo
    ok = ln > 0
    k = np.floor(np.log2(np.maximum(ln, 1))).astype(int)
    for kk in np.unique(k[ok]):
        m = ok & (k == kk)
        t = tbl[kk]
        out[m] = np.maximum(t[lo[m]], t[hi[m] - (1 << kk)])
    return out

def _window_stats(ctx, s, idx, acct, w0, w1, excl):
    """Mean / max score of the other events of `acct` with time in [T+w0, T+w1]. -1 if none."""
    I = ctx.idx[idx]
    lo, hi = I.rng(ctx.acct[acct], ctx.T + w0, ctx.T + w1)
    sc = s[I.o]
    cum = np.concatenate([[0.0], np.cumsum(sc)])
    tbl = _sparse_table(sc)
    cnt, tot = hi - lo, cum[hi] - cum[lo]
    if excl:                                   # drop the row itself (it is always inside its window)
        ps = np.empty(ctx.n, np.int64)
        ps[I.o] = np.arange(ctx.n)
        cnt, tot = cnt - 1, tot - s
        mx = np.maximum(_range_max(tbl, lo, ps), _range_max(tbl, ps + 1, hi))
    else:
        mx = _range_max(tbl, lo, hi)
    has = cnt > 0
    return np.where(has, tot / np.maximum(cnt, 1), -1.0), np.where(has, mx, -1.0)

def _link_feats(ctx, s):
    n, nxt, prv = ctx.n, ctx.nxt, ctx.prv
    at = lambda arr, ix: np.where(ix >= 0, arr[np.maximum(ix, 0)], -1.0)
    nn = np.where(nxt >= 0, nxt[np.maximum(nxt, 0)], -1)
    pp = np.where(prv >= 0, prv[np.maximum(prv, 0)], -1)
    fm, bm = s.copy(), s.copy()                # best score from here forward / backward (incl. self)
    for i in range(n - 1, -1, -1):
        j = nxt[i]
        if j >= 0 and fm[j] > fm[i]:
            fm[i] = fm[j]
    for i in range(n):
        j = prv[i]
        if j >= 0 and bm[j] > bm[i]:
            bm[i] = bm[j]
    return {"next": at(s, nxt), "prev": at(s, prv), "next2": at(s, nn), "prev2": at(s, pp),
            "fwd_max": at(fm, nxt), "bwd_max": at(bm, prv)}

def _cluster_feats(ctx, s):
    c, size = ctx.comp, ctx.comp_size
    sums = np.bincount(c, weights=s)[c]
    mean = np.where(size > 1, (sums - s) / np.maximum(size - 1, 1), -1.0)
    df = pd.DataFrame({"c": c, "s": s})
    g = df.groupby("c").s
    top1 = g.transform("max").to_numpy()
    rk = g.rank(method="first", ascending=False).to_numpy()
    second = df[rk == 2].set_index("c").s
    top_other = np.where(rk == 1, pd.Series(c).map(second).fillna(-1.0).to_numpy(), top1)
    return {"size": size.astype(float), "mean": mean, "max": np.where(size > 1, top_other, -1.0)}

def neighbour_names(groups=ALL_GROUPS):
    names = []
    for grp, specs in (("win", WIN_SPECS), ("pair", PAIR_SPECS), ("prov", PROV_SPECS)):
        if grp in groups:
            names += [f"nb_{sp[0]}_{st}" for sp in specs for st in ("mean", "max")]
    if "link" in groups:
        names += [f"nb_link_{n}" for n in LINK_NAMES]
    if "clu" in groups:
        names += [f"nb_clu_{n}" for n in CLU_NAMES]
    return names

def neighbour_features(ctx, s, groups=ALL_GROUPS):
    """DataFrame of neighbour-score features for one split; `s` aligns with the base table rows."""
    out = {}
    for grp, specs in (("win", WIN_SPECS), ("pair", PAIR_SPECS), ("prov", PROV_SPECS)):
        if grp in groups:
            for name, idx, acct, w0, w1, excl in specs:
                out[f"nb_{name}_mean"], out[f"nb_{name}_max"] = _window_stats(ctx, s, idx, acct, w0, w1, excl)
    if "link" in groups:
        out.update({f"nb_link_{k}": v for k, v in _link_feats(ctx, s).items()})
    if "clu" in groups:
        out.update({f"nb_clu_{k}": v for k, v in _cluster_feats(ctx, s).items()})
    return pd.DataFrame(out)


# ------------------------------------------------------------------ passes + ensemble -----
def final_predict(A, y, A_te, seeds=(0, 1, 2), params=None, w=(0.5, 0.25, 0.25)):
    """Rank-averaged LightGBM seeds + HistGradientBoosting + ExtraTrees. Returns a dict of parts."""
    params = dict(params or LGB_PARAMS)
    rk = lambda p: rankdata(p) / len(p)
    lgbm = np.mean([rk(lgb.LGBMClassifier(**{**params, "random_state": sd}).fit(A, y).predict_proba(A_te)[:, 1])
                    for sd in seeds], axis=0)
    hgb = rk(HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                            min_samples_leaf=30, l2_regularization=5.0, random_state=0)
             .fit(A, y).predict_proba(A_te)[:, 1])
    clean = lambda M: np.nan_to_num(M, nan=-1.0, posinf=1e6, neginf=-1e6)
    et = rk(ExtraTreesClassifier(n_estimators=400, min_samples_leaf=3, max_features=0.4, n_jobs=-1,
                                 random_state=0).fit(clean(A), y).predict_proba(clean(A_te))[:, 1])
    ens = rk(w[0] * lgbm + w[1] * hgb + w[2] * et)
    return {"lgb": lgbm, "hgb": hgb, "et": et, "ens": ens}

def run_passes(X_tr, y_tr, g_tr, X_te, featurise, n_passes=3, params=None, ensemble=True):
    """Pass 1 scores each row from its own features; every later pass adds the previous pass's
    scores and their neighbour summaries (`featurise(s_tr, s_te) -> (A_tr, A_te)`).
    Returns (list of per-pass test scores, final-stage dict or None)."""
    s_tr, s_te = stage1(X_tr, y_tr, g_tr, X_te, params)
    per_pass, A_tr, A_te = [s_te], X_tr, X_te
    for _ in range(n_passes - 1):
        A_tr, A_te = featurise(s_tr, s_te)
        s_tr, s_te = stage1(A_tr, y_tr, g_tr, A_te, params)
        per_pass.append(s_te)
    return per_pass, (final_predict(A_tr, y_tr, A_te, params=params) if ensemble else None)

def cv_featuriser(X, ctx, tr, va, groups=ALL_GROUPS):
    """Outer-CV version: one base table, scores of train rows (OOF) and held-out rows combined."""
    def f(s_tr, s_te):
        s = np.zeros(len(X))
        s[tr], s[va] = s_tr, s_te
        A = np.hstack([X, s[:, None], neighbour_features(ctx, s, groups).to_numpy()])
        return A[tr], A[va]
    return f

def final_featuriser(X_tr, ctx_tr, X_te, ctx_te, groups=ALL_GROUPS):
    """Train and test are separate splits (separate weeks): neighbours never cross between them."""
    def f(s_tr, s_te):
        return (np.hstack([X_tr, s_tr[:, None], neighbour_features(ctx_tr, s_tr, groups).to_numpy()]),
                np.hstack([X_te, s_te[:, None], neighbour_features(ctx_te, s_te, groups).to_numpy()]))
    return f

def stacked_names(cols, groups=ALL_GROUPS):
    return list(cols) + ["prev_pass_score"] + neighbour_names(groups)
