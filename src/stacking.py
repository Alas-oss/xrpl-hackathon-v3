"""Two-stage model.

Stage 1 scores every transaction from its own features. Planted scenarios are GROUPS of
related payments, so a transaction is more suspicious when the transactions around it
(same sender/receiver, the next hop, the previous hop) also score high. Stage 2 adds those
neighbour scores as features. In grouped CV this lifts AP by about 5 points.

Leak-free: stage-1 scores for training rows are out-of-fold (GroupKFold by scenario); scores
for rows being predicted come from a model that never saw them.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import rankdata
from sklearn.model_selection import GroupKFold
from config import LGB_PARAMS
from windows import BIG

class _Idx:
    def __init__(self, acct, t):
        self.o = np.lexsort((t, acct))
        self.keys = acct[self.o].astype(np.int64) * BIG + t[self.o]
    def rng(self, acct, t0, t1):
        a = acct.astype(np.int64) * BIG
        return np.searchsorted(self.keys, a + t0, "left"), np.searchsorted(self.keys, a + t1, "right")

def neighbour_features(base, score):
    """Mean/max stage-1 score of related transactions, within ONE split (`base` rows align with `score`)."""
    n = len(base)
    codes, _ = pd.factorize(pd.concat([base.sender, base.receiver]))
    S, R = codes[:n], codes[n:]
    T = base.ts.to_numpy().astype(np.int64)
    sendI, recvI = _Idx(S, T), _Idx(R, T)
    out = {}
    specs = [("s_send_pm10m", sendI, S, -600, 600, True), ("s_send_pm1h", sendI, S, -3600, 3600, True),
             ("r_recv_pm10m", recvI, R, -600, 600, True), ("r_next1h", sendI, R, 0, 3600, False),
             ("s_prev1h", recvI, S, -3600, 0, False), ("r_next10m", sendI, R, 0, 600, False)]
    for name, I, A, w0, w1, excl in specs:
        lo, hi = I.rng(A, T + w0, T + w1)
        sc = score[I.o]
        mean, mx = np.zeros(n), np.zeros(n)
        for i in np.flatnonzero(hi > lo):
            v = sc[lo[i]:hi[i]]
            if excl:
                v = v[I.o[lo[i]:hi[i]] != i]
            if len(v):
                mean[i], mx[i] = v.mean(), v.max()
        out[f"nb_{name}_mean"], out[f"nb_{name}_max"] = mean, mx
    return pd.DataFrame(out)

def stage1(X_tr, y_tr, groups_tr, X_te, params=None, k=4):
    """Out-of-fold scores for the training rows + full-model scores for X_te."""
    params = params or LGB_PARAMS
    oof = np.zeros(len(y_tr))
    for a, b in GroupKFold(k).split(X_tr, y_tr, groups=groups_tr):
        oof[b] = lgb.LGBMClassifier(**params).fit(X_tr[a], y_tr[a]).predict_proba(X_tr[b])[:, 1]
    return oof, lgb.LGBMClassifier(**params).fit(X_tr, y_tr).predict_proba(X_te)[:, 1]

def stage2_matrix(X, base, score):
    return np.hstack([X, score[:, None], neighbour_features(base, score).to_numpy()])

def stage2_names(cols):
    nb = [f"nb_{n}_{s}" for n in ("s_send_pm10m", "s_send_pm1h", "r_recv_pm10m", "r_next1h", "s_prev1h", "r_next10m")
          for s in ("mean", "max")]
    return list(cols) + ["stage1_score"] + nb

def stacked_cv(X, y, groups, base, folds, params=None):
    """Nested, leak-free CV of the two-stage model. Returns (stage1_oof, stage2_oof)."""
    params = params or LGB_PARAMS
    o1, o2 = np.zeros(len(y)), np.zeros(len(y))
    for k, (tr, va) in enumerate(folds):
        s = np.zeros(len(y))
        s[tr], s[va] = stage1(X[tr], y[tr], groups[tr], X[va], params)
        X2 = stage2_matrix(X, base, s)
        o1[va] = s[va]
        o2[va] = lgb.LGBMClassifier(**params).fit(X2[tr], y[tr]).predict_proba(X2[va])[:, 1]
        print(f"  fold {k}: stage1 {100 * _ap(y[va], o1[va]):.1f} -> stage2 {100 * _ap(y[va], o2[va]):.1f}", flush=True)
    return o1, o2

def stacked_fit_predict(X, y, groups, X_test, base_tr, base_te, params=None, seeds=(0, 1, 2)):
    """Final model: stage 1 on all labelled data, stage 2 averaged over seeds. Rank-normalised output."""
    params = dict(params or LGB_PARAMS)
    s_tr, s_te = stage1(X, y, groups, X_test, params)
    X2, X2_te = stage2_matrix(X, base_tr, s_tr), stage2_matrix(X_test, base_te, s_te)
    pred = np.zeros(len(X_test))
    for sd in seeds:
        params["random_state"] = sd
        m = lgb.LGBMClassifier(**params).fit(X2, y)
        pred += rankdata(m.predict_proba(X2_te)[:, 1]) / len(X_test) / len(seeds)
    return pred, m

def _ap(y, s):
    from sklearn.metrics import average_precision_score
    return average_precision_score(y, s)
