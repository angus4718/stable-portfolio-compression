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
from spc.basis import BasisSelector


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
    use_weighted_selection: bool = True,
    weight_alpha: float = 0.5,
    hub_branch_h: int = None,
    hub_branch_alpha: float = 0.5,
    hub_branch_weight_gamma: float = 0.0,
    hub_branch_rep_alpha: float = 0.5,
):
    prices = pd.read_csv(prices_path, index_col=0, parse_dates=True)
    prices = prices.sort_index()

    # Load weights if provided and filter to assets with positive weights
    avg_weights = None
    if weights_path:
        weights = pd.read_csv(weights_path, index_col=0, parse_dates=True)
        weights = weights.sort_index()
        # Find assets that have any positive weight across any month
        assets_with_positive_weight = weights.columns[(weights > 0).any(axis=0)]
        print(
            f"Assets with positive weights: {len(assets_with_positive_weight)} out of {len(weights.columns)}"
        )
        # Compute average weight for each asset (across all months)
        avg_weights = weights[assets_with_positive_weight].mean(axis=0)
        print(
            f"Average weights - min: {avg_weights.min():.4f}, max: {avg_weights.max():.4f}, mean: {avg_weights.mean():.4f}"
        )
        # Filter prices to only these assets
        prices = prices[assets_with_positive_weight]
        print(f"Filtered prices to {len(prices.columns)} assets")

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

    # Route to the correct selection method
    method = basis_selection_method.lower()
    if method in ("max_spread", "max_spread_weighted"):
        if avg_weights is None:
            raise ValueError("Weighted selection requested but no weights provided.")
        weight_dict = avg_weights.to_dict()
        # max_spread: alpha=1.0 (pure spread), max_spread_weighted: use configured alpha
        alpha = 1.0 if method == "max_spread" else weight_alpha
        print(f"Using max-spread-weighted selection (alpha={alpha})")
        basis = selector.select_max_spread_weighted(k, weight_dict, alpha=alpha)
        return basis
    elif method == "hub_branch":
        print(
            f"Using hub-and-branch selection (h={hub_branch_h}, alpha={hub_branch_alpha}, weight_gamma={hub_branch_weight_gamma}, rep_alpha={hub_branch_rep_alpha})"
        )
        basis = selector.select_hub_branch(
            k,
            h=hub_branch_h,
            alpha=hub_branch_alpha,
            weight_gamma=hub_branch_weight_gamma,
            rep_alpha=hub_branch_rep_alpha,
        )
        return basis
    else:
        raise ValueError(
            "basis_selection_method must be one of {'max_spread', 'max_spread_weighted', 'hub_branch'}"
        )


def main():

    # Always load config from JSON file in the same directory as this script
    config_path = Path(__file__).parent / "basis_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as f:
        config = json.load(f)
    print(f"Loaded configuration from {config_path}")

    # Extract config values
    prices_path = config["input"]["prices_path"]["value"]
    weights_path = config["input"].get("weights_path", {}).get("value")
    basis_size = config["basis_selection"]["basis_size"]["value"]
    use_weighted = (
        config["basis_selection"].get("use_weighted_selection", {}).get("value", True)
    )
    weight_alpha = config["basis_selection"].get("weight_alpha", {}).get("value", 0.5)
    basis_selection_method = (
        config["basis_selection"]
        .get("basis_selection_method", {})
        .get("value", "max_spread")
    )
    hub_branch_h = config["basis_selection"].get("hub_branch_h", {}).get("value", None)
    hub_branch_alpha = (
        config["basis_selection"].get("hub_branch_alpha", {}).get("value", 0.5)
    )
    hub_branch_weight_gamma = (
        config["basis_selection"].get("hub_branch_weight_gamma", {}).get("value", 0.0)
    )
    hub_branch_rep_alpha = (
        config["basis_selection"].get("hub_branch_rep_alpha", {}).get("value", 0.5)
    )
    corr_method = config["distance_computation"]["corr_method"]["value"]
    min_periods = config["distance_computation"]["min_periods"]["value"]
    shrink_method = config["distance_computation"]["shrink_method"]["value"]
    pca_n_components = config["pca_denoising"]["pca_n_components"]["value"]
    pca_explained_variance = config["pca_denoising"]["pca_explained_variance"]["value"]
    output_path = config["output"]["output_path"]["value"]

    print("\n" + "=" * 70)
    print("BASIS ASSET SELECTION")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Prices file: {prices_path}")
    print(f"  Weights file: {weights_path}")
    print(f"  Basis size: {basis_size}")
    print(f"  Basis selection method: {basis_selection_method}")
    print(f"  Use weighted selection: {use_weighted}")
    if use_weighted:
        print(
            f"    - Weight alpha: {weight_alpha} (0=pure weight, 0.5=balanced, 1=pure spread)"
        )
    print(f"  Correlation method: {corr_method}")
    print(f"  Min periods: {min_periods}")
    print(f"  Shrink method: {shrink_method}")
    if shrink_method == "pca":
        print(f"    - PCA components: {pca_n_components}")
        print(f"    - Explained variance: {pca_explained_variance}")
    print(f"  Output path: {output_path}")

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
        use_weighted_selection=use_weighted,
        weight_alpha=weight_alpha,
        hub_branch_h=hub_branch_h,
        hub_branch_alpha=hub_branch_alpha,
        hub_branch_weight_gamma=hub_branch_weight_gamma,
        hub_branch_rep_alpha=hub_branch_rep_alpha,
    )

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.Series(basis, name="ticker").to_csv(out_path, index=False)
    print(f"\nSaved basis (k={len(basis)}) to {out_path}")
    print(f"Selected basis tickers:\n{basis}")
    print("=" * 70)


if __name__ == "__main__":
    main()
