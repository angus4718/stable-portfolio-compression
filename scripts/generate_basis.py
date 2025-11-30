"""
Generate basis stocks from synthetic monthly prices using SPC.

Configuration:
- Edit basis_config.json to customize parameters
"""

import json
from pathlib import Path
import pandas as pd

# Ensure project root is on sys.path so local package imports work when running script
import sys

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from spc.graph import DistancesUtils, MST
from spc.portfolio_construction import BasisSelector
from spc.utils import cfg_val


def select_basis(
    prices_path: str,
    weights_path: str = None,
    basis_size: int = 10,
    shrink_method: str = None,
    pca_n_components: int = None,
    pca_explained_variance: float = None,
    corr_method: str = "pearson",
    min_periods: int = 1,
    basis_selection_method: str = "max_spread",
    weight_alpha: float = 0.5,
    hub_branch_h: int = None,
    hub_branch_alpha: float = 0.5,
    hub_branch_weight_gamma: float = 0.0,
    hub_branch_rep_alpha: float = 0.5,
):
    prices = pd.read_csv(prices_path, index_col=0, parse_dates=True).sort_index()

    # Load weights if provided and filter to assets with positive weights
    avg_weights = None
    if weights_path:
        weights = pd.read_csv(weights_path, index_col=0, parse_dates=True).sort_index()
        positive_cols = weights.columns[(weights > 0).any(axis=0)]
        avg_weights = weights[positive_cols].mean(axis=0)
        prices = prices[positive_cols]

    # Compute distance matrix
    dist_df = DistancesUtils.price_to_distance_df(
        prices,
        min_periods=min_periods,
        corr_method=corr_method,
        shrink_method=shrink_method,
        pca_n_components=pca_n_components,
        pca_explained_variance=pca_explained_variance,
    )

    # Build MST adjacency
    dist_matrix = dist_df.values.tolist()
    tickers = list(dist_df.columns)
    mst_adj = MST(dist_matrix, tickers).get_adj_dict()

    # Select basis using BasisSelector
    selector = BasisSelector(mst_adj, nodes=list(mst_adj.keys()))
    k = int(min(basis_size, len(selector.nodes)))

    method = basis_selection_method.lower()
    if method in ("max_spread", "max_spread_weighted"):
        if avg_weights is None:
            raise ValueError("Weighted selection requested but no weights provided.")
        weight_dict = avg_weights.to_dict()
        alpha = 1.0 if method == "max_spread" else weight_alpha
        return selector.select_max_spread_weighted(k, weight_dict, alpha=alpha)
    if method == "hub_branch":
        return selector.select_hub_branch(
            k,
            h=hub_branch_h,
            alpha=hub_branch_alpha,
            weight_gamma=hub_branch_weight_gamma,
            rep_alpha=hub_branch_rep_alpha,
        )
    raise ValueError(
        "basis_selection_method must be one of {'max_spread', 'max_spread_weighted', 'hub_branch'}"
    )


def main():
    config_path = _ROOT / "scripts" / "basis_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    print(f"Loaded configuration from {config_path}")

    prices_path = cfg_val(config, "input", "prices_path")
    weights_path = cfg_val(config, "input", "weights_path", None)
    basis_size = int(cfg_val(config, "basis_selection", "basis_size", 10))
    weight_alpha = cfg_val(config, "basis_selection", "weight_alpha", 0.5)
    basis_selection_method = cfg_val(
        config, "basis_selection", "basis_selection_method", "max_spread"
    )
    hub_branch_h = cfg_val(config, "basis_selection", "hub_branch_h", None)
    hub_branch_alpha = cfg_val(config, "basis_selection", "hub_branch_alpha", 0.5)
    hub_branch_weight_gamma = cfg_val(
        config, "basis_selection", "hub_branch_weight_gamma", 0.0
    )
    hub_branch_rep_alpha = cfg_val(
        config, "basis_selection", "hub_branch_rep_alpha", 0.5
    )
    corr_method = cfg_val(config, "distance_computation", "corr_method", "pearson")
    min_periods = int(cfg_val(config, "distance_computation", "min_periods", 1))
    shrink_method = cfg_val(config, "distance_computation", "shrink_method", None)
    pca_n_components = cfg_val(config, "pca_denoising", "pca_n_components", None)
    pca_explained_variance = cfg_val(
        config, "pca_denoising", "pca_explained_variance", None
    )
    output_path = cfg_val(config, "output", "output_path", "outputs/basis_selected.csv")

    print(
        f"Loaded config: prices={prices_path}, basis_k={basis_size}, method={basis_selection_method}"
    )

    basis = select_basis(
        prices_path=prices_path,
        weights_path=weights_path,
        basis_size=basis_size,
        shrink_method=shrink_method,
        pca_n_components=pca_n_components,
        pca_explained_variance=pca_explained_variance,
        corr_method=corr_method,
        min_periods=min_periods,
        basis_selection_method=basis_selection_method,
        weight_alpha=weight_alpha,
        hub_branch_h=hub_branch_h,
        hub_branch_alpha=hub_branch_alpha,
        hub_branch_weight_gamma=hub_branch_weight_gamma,
        hub_branch_rep_alpha=hub_branch_rep_alpha,
    )

    out_dir = _ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / Path(output_path).name
    pd.Series(basis, name="ticker").to_csv(out_path, index=False)
    print(f"\nSaved basis (k={len(basis)}) to {out_path}")
    print(f"Selected basis tickers:\n{basis}")


if __name__ == "__main__":
    main()
