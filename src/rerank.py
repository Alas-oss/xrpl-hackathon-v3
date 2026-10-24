"""Second-stage reranker for the TOP BAND of the ranking.

Why: in your grouped CV almost all easy positives sit inside the top ~10% of the ranking, but only
~40% of them fit in the top 1%, and the medium ones are mixed with look-alike decoys. A model
trained on every row spends its capacity separating positives from obviously-normal rows. This
one is trained ONLY on rows the main model already ranks in the top `frac`, so it can focus on
positives versus the look-alikes that got that far.

Leak-free: the band is defined by OUT-OF-FOLD scores on the training split, and the reranker is
itself evaluated with GroupKFold by scenario. Rows outside the band keep their original order;
band rows are placed above them and re-ordered by a blend of the old and new scores.
"""
import numpy as np
import lightgbm as lgb
from scipy.stats import rankdata
from sklearn.model_selection import GroupKFold
from config import LGB_PARAMS

RERANK_PARAMS = {**LGB_PARAMS, "n_estimators": 300, "num_leaves": 7, "min_child_samples": 20,
                 "reg_lambda": 10.0}

def rank01(x):
    return rankdata(x) / len(x)

def band_mask(score, frac):
    k = int(round(len(score) * frac))
    m = np.zeros(len(score), bool)
    m[np.argsort(-np.asarray(score), kind="stable")[:k]] = True
    return m

def _design(X, score):
    return np.hstack([X, rank01(score)[:, None]])        # own features + the main model's rank

def combine(score, band, rr, w):
    """Final ordering: band rows above everything else; inside the band, blend of the old rank
    (weight 1-w) and the reranker (weight w). Rows outside the band keep their old order."""
    out = rank01(score)
    b = np.flatnonzero(band)
    out[b] = 1 + rank01(w * rank01(rr) + (1 - w) * rank01(score[b]))
    return out

def _fit_mean(A, y, A_te, params, seeds):
    return np.mean([lgb.LGBMClassifier(**{**params, "random_state": s}).fit(A, y).predict_proba(A_te)[:, 1]
                    for s in seeds], axis=0)

def rerank_cv(X, y, groups, score, frac=0.10, ws=(0.0, 0.25, 0.5, 0.75, 1.0), k=5,
              params=RERANK_PARAMS, seeds=(0, 1, 2)):
    """score: out-of-fold scores of the main model. Returns (band, reranker OOF, {w: final order})."""
    band = band_mask(score, frac)
    b = np.flatnonzero(band)
    A, yb, gb = _design(X, score)[b], y[b], groups[b]
    rr = np.zeros(len(b))
    for tr, va in GroupKFold(k).split(A, yb, groups=gb):
        rr[va] = _fit_mean(A[tr], yb[tr], A[va], params, seeds)
    return band, rr, {w: combine(score, band, rr, w) for w in ws}

def rerank_fit_predict(X, y, score_tr, X_te, score_te, frac=0.10, w=0.5,
                       params=RERANK_PARAMS, seeds=(0, 1, 2)):
    """Train on the training band (by out-of-fold score), rerank the test band (by test score)."""
    btr, bte = np.flatnonzero(band_mask(score_tr, frac)), band_mask(score_te, frac)
    rr = _fit_mean(_design(X, score_tr)[btr], y[btr], _design(X_te, score_te)[np.flatnonzero(bte)], params, seeds)
    return combine(score_te, bte, rr, w)
