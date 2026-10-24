"""Grouped cross-validation + LightGBM, scored with Average Precision (the site's metric)."""
import numpy as np
import lightgbm as lgb
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupKFold
from config import LGB_PARAMS, N_FOLDS

def rank01(x):
    """Rank-normalise to (0, 1]; AP only needs the ordering, and ties are avoided."""
    return rankdata(x, method="average") / len(x)

def make_folds(X, y, groups):
    """Keep every scenario (and unrelated rows by sender) inside ONE fold.
    A plain random split leaks near-duplicate scenario members and overstates AP badly."""
    return list(GroupKFold(N_FOLDS).split(X, y, groups=groups))

def cross_validate(X, y, folds, params=None):
    params = params or LGB_PARAMS
    oof = np.zeros(len(y))
    for k, (tr, va) in enumerate(folds):
        m = lgb.LGBMClassifier(**params).fit(X[tr], y[tr])
        oof[va] = m.predict_proba(X[va])[:, 1]
        print(f"  fold {k}: AP*100 = {100 * average_precision_score(y[va], oof[va]):.1f}")
    return oof

def fit_predict(X, y, X_test, params=None, seeds=(0, 1, 2)):
    """Average a few seeds of the final model trained on ALL labelled data."""
    params = dict(params or LGB_PARAMS)
    pred = np.zeros(len(X_test))
    for sd in seeds:
        params["random_state"] = sd
        m = lgb.LGBMClassifier(**params).fit(X, y)
        pred += rankdata(m.predict_proba(X_test)[:, 1]) / len(X_test) / len(seeds)
    return pred, m

def precision_at_top(y, scores, frac=0.01):
    n = max(int(len(y) * frac), 1)
    return y[np.argsort(-scores, kind="stable")[:n]].mean()
