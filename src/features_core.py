"""Per-transaction features that need no other transactions (amounts, assets, conversion)."""
import numpy as np
import pandas as pd
from config import ID_COL

def _sym(a):
    return a.str.split(":").str[-1]

def build(base):
    f = pd.DataFrame(index=base[ID_COL])
    v = base.dst_usd.to_numpy()
    f["log_dst_usd"] = np.log1p(v)
    f["log_src_usd"] = np.log1p(base.src_usd.to_numpy())
    f["n_legs"] = base.n_legs.to_numpy()
    f["is_conversion"] = base.is_conversion.astype(float).to_numpy()
    f["is_xrp_bridge"] = (base.n_legs == 3).astype(float).to_numpy()
    f["has_paths"] = base.has_paths.astype(float).to_numpy()
    for sym in ["XRP", "USD", "GBP", "EUR"]:
        f[f"dst_is_{sym}"] = (_sym(base.dst_asset) == sym).astype(float).to_numpy()
        f[f"src_is_{sym}"] = (_sym(base.src_asset) == sym).astype(float).to_numpy()
    f["cross_issuer_same_symbol"] = ((_sym(base.dst_asset) == _sym(base.src_asset))
                                     & (base.dst_asset != base.src_asset)).astype(float).to_numpy()
    # What the customer really paid vs what arrived, in reference USD (the conversion cost).
    f["spread"] = (base.src_usd / base.dst_usd - 1).clip(-1, 5).to_numpy()
    # How loose SendMax was versus the real debit (NaN-safe: only conversions have SendMax).
    f["sendmax_slack"] = (base.sendmax_units / base.src_units - 1).clip(-1, 5).fillna(-2).to_numpy()
    f["fee_micro_xrp"] = base.fee.to_numpy() * 1e6
    f["tx_index"] = base.tx_index.to_numpy()
    # Round amounts (units of the delivered asset and in USD).
    u = base.dst_units.to_numpy()
    f["round_units_1"] = (np.abs(u - np.round(u)) < 1e-6).astype(float)
    f["round_units_100"] = (np.abs(u / 100 - np.round(u / 100)) < 1e-6).astype(float)
    f["round_usd_100"] = (np.abs(v / 100 - np.round(v / 100)) < 0.002).astype(float)
    hour = (base.ts.to_numpy() % 86400) / 3600
    f["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    f["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    f["is_night"] = ((hour < 6)).astype(float)
    return f
