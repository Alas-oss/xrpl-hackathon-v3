"""Graph / relationship features (sender -> receiver edges within one split)."""
import numpy as np
import pandas as pd
import networkx as nx
from config import ID_COL

def build(base):
    s, r = base.sender.to_numpy(), base.receiver.to_numpy()
    v = base.dst_usd.to_numpy()
    f = pd.DataFrame(index=base[ID_COL])
    edges = pd.DataFrame({"s": s, "r": r, "v": v})

    # pair statistics
    pair = edges.groupby(["s", "r"]).v.agg(["size", "sum"])
    key = pd.MultiIndex.from_arrays([s, r])
    f["pair_n"] = pair["size"].reindex(key).to_numpy()
    f["pair_log_usd"] = np.log1p(pair["sum"].reindex(key).to_numpy())
    rev = pair.reindex(pd.MultiIndex.from_arrays([r, s]))
    f["rev_pair_n"] = rev["size"].fillna(0).to_numpy()
    f["rev_pair_log_usd"] = np.log1p(rev["sum"].fillna(0).to_numpy())

    # repeated near-identical amounts on the same pair (structuring / scheduled settlement)
    rep = np.zeros(len(edges))
    for _, idx in edges.groupby(["s", "r"]).indices.items():
        if len(idx) > 1:
            vv = v[idx]
            rep[idx] = ((np.abs(vv[:, None] - vv[None, :]) <= 0.02 * vv[:, None]).sum(1) - 1)
    f["pair_same_amount_n"] = rep

    # directed graph: neighbours, 3-cycles, common neighbours, PageRank
    out_nb = edges.groupby("s").r.agg(set).to_dict()
    in_nb = edges.groupby("r").s.agg(set).to_dict()
    uniq = edges[["s", "r"]].drop_duplicates()
    cyc3, common = {}, {}
    for a, b in zip(uniq.s, uniq.r):
        cyc3[(a, b)] = len(out_nb.get(b, set()) & in_nb.get(a, set()))     # a->b->c->a
        common[(a, b)] = len((out_nb.get(a, set()) | in_nb.get(a, set())) &
                             (out_nb.get(b, set()) | in_nb.get(b, set())))
    f["cycle3_count"] = [cyc3[(a, b)] for a, b in zip(s, r)]
    f["common_neighbours"] = [common[(a, b)] for a, b in zip(s, r)]
    G = nx.DiGraph()
    for (a, b), w in pair["size"].items():
        G.add_edge(a, b, weight=float(w))
    pr = nx.pagerank(G, weight="weight")
    f["s_pagerank"] = pd.Series(s).map(pr).to_numpy() * len(pr)
    f["r_pagerank"] = pd.Series(r).map(pr).to_numpy() * len(pr)

    # liquidity-provider "hubness": how often an account sits in the middle of a conversion
    prov = pd.concat([base.provider1, base.provider2]).value_counts()
    for side, a in (("s", s), ("r", r)):
        f[f"{side}_provider_n"] = pd.Series(a).map(prov).fillna(0).to_numpy()
    f["provider_hit"] = (base.provider1.map(prov).fillna(0) +
                         base.provider2.map(prov).fillna(0)).to_numpy()
    return f
