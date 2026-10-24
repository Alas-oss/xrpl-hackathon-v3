"""Money-trail features: follow a payment forward/backward through accounts that pass on a
similar amount shortly afterwards (layering chains / cycles), plus relative burst sizes."""
import numpy as np
import pandas as pd
from config import ID_COL
from windows import EventIndex

def build(base, window=7200, tol=0.15):
    n = len(base)
    codes, _ = pd.factorize(pd.concat([base.sender, base.receiver]))
    S, R = codes[:n], codes[n:]
    T = base.ts.to_numpy().astype(np.int64)
    V = base.dst_usd.to_numpy()
    sends, recvs = EventIndex(S, T, V), EventIndex(R, T, V)
    # order[k] = original row of the k-th sorted event (EventIndex sorts by (account, time))
    s_order, r_order = np.lexsort((T, S)), np.lexsort((T, R))

    nxt, prv = -np.ones(n, int), -np.ones(n, int)
    lo, hi = sends._rng(R, T + 1, T + window)           # receiver's later sends
    for i in np.flatnonzero(hi > lo):
        ids = s_order[lo[i]:hi[i]]
        ok = np.abs(V[ids] - V[i]) <= tol * V[i]
        if ok.any():
            nxt[i] = ids[np.argmax(ok)]
    lo, hi = recvs._rng(S, T - window, T - 1)           # sender's earlier receipts
    for i in np.flatnonzero(hi > lo):
        ids = r_order[lo[i]:hi[i]][::-1]
        ok = np.abs(V[ids] - V[i]) <= tol * V[i]
        if ok.any():
            prv[i] = ids[np.argmax(ok)]

    fwd, bwd, ret = np.zeros(n), np.zeros(n), np.zeros(n)
    for i in range(n - 1, -1, -1):
        if nxt[i] >= 0:
            fwd[i] = 1 + fwd[nxt[i]]
    for i in range(n):
        if prv[i] >= 0:
            bwd[i] = 1 + bwd[prv[i]]
    for i in range(n):
        j, h = nxt[i], 0
        while j >= 0 and h < 5:
            if R[j] == S[i]:
                ret[i] = 1
                break
            j, h = nxt[j], h + 1

    out_n = np.bincount(S, minlength=codes.max() + 1)
    in_n = np.bincount(R, minlength=codes.max() + 1)
    f = pd.DataFrame(index=base[ID_COL])
    f["chain_fwd"], f["chain_bwd"], f["chain_len"], f["chain_returns"] = fwd, bwd, fwd + bwd, ret
    f["s_burst_rel"] = (sends.count(S, T - 600, T + 600) - 1) / (out_n[S] + 1)
    f["r_burst_rel"] = (recvs.count(R, T - 600, T + 600) - 1) / (in_n[R] + 1)
    return f
