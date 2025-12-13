from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Dict

import pandas as pd
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from spc.portfolio_construction import BasisSelector
from spc.graph import DistancesUtils, MST
from spc.utils import cfg_val


def load_config(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    cfg_path = Path(__file__).resolve().parent / "basis_timeseries_config.json"
    cfg = load_config(cfg_path)

    prices_path = Path(
        cfg_val(cfg, "paths", "prices_path", "synthetic_data/prices_monthly.csv")
    )
    marketcap_path = cfg_val(
        cfg, "paths", "marketcap_path", "synthetic_data/market_cap_values.csv"
    )
    out_path = Path(
        cfg_val(cfg, "paths", "output_path", "analysis_outputs/basis_timeseries.csv")
    )

    k = int(cfg_val(cfg, "params", "k", 10))
    method = cfg_val(cfg, "params", "method", "max_spread")
    method_kwargs = cfg_val(cfg, "params", "method_kwargs", {}) or {}

    min_periods = int(cfg_val(cfg, "distance_computation", "min_periods", 1))
    corr_method = cfg_val(cfg, "distance_computation", "corr_method", "pearson")
    shrink_method = cfg_val(cfg, "distance_computation", "shrink_method", None)
    pca_n_components = (
        int(cfg_val(cfg, "distance_computation", "pca_n_components", None))
        if cfg_val(cfg, "distance_computation", "pca_n_components", None) is not None
        else None
    )
    pca_explained_variance = (
        float(cfg_val(cfg, "distance_computation", "pca_explained_variance", None))
        if cfg_val(cfg, "distance_computation", "pca_explained_variance", None)
        is not None
        else None
    )

    prices = pd.read_csv(prices_path, index_col=0, parse_dates=True)
    mc_df = pd.read_csv(Path(marketcap_path), index_col=0, parse_dates=True)

    dates = list(prices.index)

    results = []
    prev_basis = None

    for t in tqdm(dates, desc="Dates", unit="date"):
        past = prices[prices.index < t]
        if past.shape[0] < 1:
            results.append({"date": t.isoformat(), "basis": []})
            prev_basis = []
            continue

        dist = DistancesUtils.price_to_distance_df(
            past,
            min_periods=min_periods,
            corr_method=corr_method,
            shrink_method=shrink_method,
            pca_n_components=pca_n_components,
            pca_explained_variance=pca_explained_variance,
        )

        # build MST and selector
        nodes = dist.columns.tolist()
        adj_matrix = dist.values.tolist()
        mst = MST(adj_matrix, nodes=nodes)
        mst_adj = mst.get_adj_dict()
        selector = BasisSelector(mst_adj)

        # compute weights using most recent marketcap strictly before t
        if mc_df is not None:
            mc_past = mc_df[mc_df.index < t]
            if mc_past.shape[0] >= 1:
                mc_row = mc_past.iloc[-1]
                weights = {n: mc_row.get(n, 0.0) for n in selector.nodes}
            else:
                weights = {n: 1.0 for n in selector.nodes}
        else:
            weights = {n: 1.0 for n in selector.nodes}

        # select basis according to method
        if method == "max_spread":
            alpha = float(method_kwargs.get("alpha", 0.5))
            stickiness = float(method_kwargs.get("stickiness", 0.0))
            basis = selector.select_max_spread(
                k=k,
                weights=weights,
                alpha=alpha,
                stickiness=stickiness,
                prev_basis=prev_basis,
            )
        elif method in ("hub_branch"):
            h = method_kwargs.get("h", None)
            h = int(h) if h is not None else None
            alpha = float(method_kwargs.get("alpha", 0.5))
            weight_gamma = float(method_kwargs.get("weight_gamma", 0.0))
            rep_alpha = float(method_kwargs.get("rep_alpha", 0.5))
            stickiness = float(method_kwargs.get("stickiness", 0.0))
            basis = selector.select_hub_branch(
                k=k,
                h=h,
                alpha=alpha,
                weights=weights,
                weight_gamma=weight_gamma,
                rep_alpha=rep_alpha,
                stickiness=stickiness,
                prev_basis=prev_basis,
            )
        else:
            raise ValueError(f"unknown method: {method}")

        results.append({"date": t.isoformat(), "basis": basis})
        prev_basis = list(basis)

    out_dir = out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [{"date": r["date"], "basis": ";".join(r["basis"])} for r in results]
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Wrote basis timeseries for {len(rows)} dates to: {out_path}")


if __name__ == "__main__":
    main()
