import json
from pathlib import Path
import sys
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from spc.portfolio_construction import BasisSelector
from spc.graph import DistancesUtils, MST
from spc.utils import cfg_val


def load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    cfg_path = Path(__file__).resolve().parent / "basis_config.json"
    cfg = load_config(cfg_path)

    prices_path = Path(
        cfg_val(cfg, "paths", "prices_path", "synthetic_data/prices_monthly.csv")
    )
    marketcap_path = Path(
        cfg_val(cfg, "paths", "marketcap_path", "synthetic_data/market_cap_values.csv")
    )
    out_path = Path(cfg_val(cfg, "paths", "output_path", "outputs/basis_selected.csv"))
    k = int(cfg_val(cfg, "params", "k", 10))
    method = cfg_val(cfg, "params", "method", "max_spread")
    method_kwargs = cfg_val(cfg, "params", "method_kwargs", {}) or {}

    prices = pd.read_csv(prices_path, index_col=0, parse_dates=True)
    dist = DistancesUtils.price_to_distance_df(prices)

    # build MST
    nodes = dist.columns.tolist()
    adj_matrix = dist.values.tolist()
    mst = MST(adj_matrix, nodes=nodes)
    mst_adj = mst.get_adj_dict()

    selector = BasisSelector(mst_adj)

    # compute weights
    mc = pd.read_csv(marketcap_path, index_col=0, parse_dates=True)
    last_mc = mc.iloc[-1]
    # build weights dict
    weights = {n: last_mc.get(n, 0.0) for n in selector.nodes}

    # choose selection method
    if method == "max_spread":
        alpha = float(method_kwargs.get("alpha", 0.5))
        stickiness = float(method_kwargs.get("stickiness", 0.0))
        prev_basis = method_kwargs.get("prev_basis", None)
        basis = selector.select_max_spread(
            k=k,
            weights=weights,
            alpha=alpha,
            stickiness=stickiness,
            prev_basis=prev_basis,
        )
    elif method == "hub_branch":
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
        )
    else:
        raise ValueError(f"unknown method: {method}")

    out_dir = out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"basis": basis}).to_csv(out_path, index=False)
    print(f"Wrote {len(basis)} basis tickers to: {out_path}")


if __name__ == "__main__":
    main()
