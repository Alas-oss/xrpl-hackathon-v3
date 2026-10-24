"""Raw tables -> ONE ROW PER TRANSACTION (the 'base' table).

Inputs (per split folder): transactions.parquet, transfers.parquet, market_rates.csv
(blocks.parquet is not needed: it only carries ledger close metadata.)

Key facts from the README:
  * one score per TRANSACTION (1-3 transfer legs per transaction, ordered by TRANSFER_INDEX)
  * leg 0 = what the sender actually paid; last leg = what the receiver got
  * SendMax is only a ceiling, not the real debit
  * value assets in USD with the HOURLY reference price of the hour containing the timestamp
  * group assets by (issuer, currency); XRP is the sentinel token address
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from config import ID_COL, XRP_SENTINEL

def _asset(token, symbol):
    return np.where(token == XRP_SENTINEL, "XRP", token + ":" + symbol)

def _decode(wire):
    """DELIVERED_AMOUNT / SendMax JSON -> (asset key, units)."""
    if wire is None:
        return None, np.nan
    if isinstance(wire, str):
        return "XRP", int(wire) / 1e6
    return f"{wire['issuer']}:{wire['currency']}", float(wire["value"])

def load_split(directory):
    d = Path(directory)
    tx = pd.read_parquet(d / "transactions.parquet")
    tr = pd.read_parquet(d / "transfers.parquet")
    rates = pd.read_csv(d / "market_rates.csv")
    rate = rates.assign(price=rates.PRICE_MICRO_USD / 1e6).set_index(["TIMESTAMP", "ASSET"]).price

    def usd(ts, asset, units):
        idx = pd.MultiIndex.from_arrays([(np.asarray(ts) // 3600) * 3600, np.asarray(asset, dtype=object)])
        return units * rate.reindex(idx).to_numpy()

    ts = pd.to_numeric(tx.TIMESTAMP).astype("int64").to_numpy()
    extra = [json.loads(s) for s in tx.EXTRA_TRANSACTION_FIELDS]
    dst = [_decode(json.loads(w)) for w in tx.DELIVERED_AMOUNT]
    smx = [_decode(e.get("SendMax")) for e in extra]

    base = pd.DataFrame({
        ID_COL: tx[ID_COL].to_numpy(),
        "ts": ts,
        "sender": tx.ACCOUNT.to_numpy(),
        "receiver": [e["Destination"] for e in extra],
        "dst_asset": [a for a, _ in dst], "dst_units": [u for _, u in dst],
        "sendmax_asset": [a for a, _ in smx], "sendmax_units": [u for _, u in smx],
        "has_paths": [("Paths" in e) for e in extra],
        "fee": tx.FEE.to_numpy(),
        "tx_index": pd.to_numeric(tx.TRANSACTION_INDEX).to_numpy(),
    })
    base["dst_usd"] = usd(ts, base.dst_asset, base.dst_units.to_numpy())
    base["sendmax_usd"] = usd(ts, base.sendmax_asset.fillna("XRP"), base.sendmax_units.to_numpy())

    # ---- transfer legs -> first / last / providers -------------------------------------
    tr = tr.assign(idx=pd.to_numeric(tr.TRANSFER_INDEX),
                   asset=_asset(tr.TOKEN_ADDRESS.to_numpy(), tr.SYMBOL.to_numpy()))
    tr = tr.sort_values([ID_COL, "idx"], kind="stable")
    g = tr.groupby(ID_COL, sort=False)
    first = g.head(1).set_index(ID_COL)
    last = g.tail(1).set_index(ID_COL)
    n_legs = g.size()
    second = tr[tr.idx == 1].set_index(ID_COL)
    h = base[ID_COL]
    base["n_legs"] = h.map(n_legs).to_numpy()
    base["is_conversion"] = (h.map(first.TRANSFER_TYPE) == "path_payment").to_numpy()
    base["src_asset"] = h.map(first.asset).to_numpy()
    base["src_units"] = h.map(first.VALUE).to_numpy()
    base["src_usd"] = usd(ts, base.src_asset, base.src_units.to_numpy())
    base["sender_leg0"] = h.map(first.FROM_ADDRESS).to_numpy()
    base["receiver_last"] = h.map(last.TO_ADDRESS).to_numpy()
    mid = h.map(first.TO_ADDRESS).where(base.n_legs > 1)
    base["provider1"] = mid.to_numpy()
    base["provider2"] = h.map(second.TO_ADDRESS).where(base.n_legs > 2).to_numpy()
    return base.sort_values(["ts", "tx_index"], kind="stable").reset_index(drop=True)

def sanity(base, name=""):
    print(f"[{name}] rows={len(base):,} unique ids={base[ID_COL].is_unique} "
          f"nulls(dst_usd/src_usd)={base.dst_usd.isna().sum()}/{base.src_usd.isna().sum()} "
          f"sender==leg0 sender: {(base.sender == base.sender_leg0).mean():.4f} "
          f"receiver==last leg: {(base.receiver == base.receiver_last).mean():.4f}")
