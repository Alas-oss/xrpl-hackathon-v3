"""Fast per-account time-window queries (vectorised with searchsorted)."""
import numpy as np

BIG = 10**10  # > any unix timestamp, so key = account*BIG + ts sorts by (account, ts)

class EventIndex:
    """Events (account, time, value) sorted for O(log n) window counts / sums."""
    def __init__(self, acct, t, v):
        acct, t, v = np.asarray(acct, np.int64), np.asarray(t, np.int64), np.asarray(v, float)
        order = np.lexsort((t, acct))
        self.keys = acct[order] * BIG + t[order]
        self.v = v[order]
        self.cum = np.concatenate([[0.0], np.cumsum(self.v)])

    def _rng(self, acct, t0, t1):
        a = np.asarray(acct, np.int64) * BIG
        return (np.searchsorted(self.keys, a + t0, "left"),
                np.searchsorted(self.keys, a + t1, "right"))

    def count(self, acct, t0, t1):
        lo, hi = self._rng(acct, t0, t1)
        return hi - lo

    def total(self, acct, t0, t1):
        lo, hi = self._rng(acct, t0, t1)
        return self.cum[hi] - self.cum[lo]

    def similar(self, acct, t0, t1, x, tol):
        """Is there an event in [t0, t1] whose value is within tol (relative) of x?"""
        lo, hi = self._rng(acct, t0, t1)
        x = np.asarray(x, float)
        out = np.zeros(len(lo), bool)
        for i in np.flatnonzero(hi > lo):
            out[i] = np.any(np.abs(self.v[lo[i]:hi[i]] - x[i]) <= tol * x[i])
        return out

    def gap_before(self, acct, t, cap):
        """Seconds since this account's previous event strictly before t (cap if none)."""
        a = np.asarray(acct, np.int64)
        pos = np.searchsorted(self.keys, a * BIG + t, "left") - 1
        ok = pos >= 0
        k = self.keys[np.where(ok, pos, 0)]
        ok &= (k // BIG) == a
        return np.where(ok, np.minimum(t - (k % BIG), cap), cap).astype(float)

    def gap_after(self, acct, t, cap):
        """Seconds until this account's next event strictly after t (cap if none)."""
        a = np.asarray(acct, np.int64)
        pos = np.searchsorted(self.keys, a * BIG + t, "right")
        ok = pos < len(self.keys)
        k = self.keys[np.where(ok, pos, 0)]
        ok &= (k // BIG) == a
        return np.where(ok, np.minimum((k % BIG) - t, cap), cap).astype(float)
