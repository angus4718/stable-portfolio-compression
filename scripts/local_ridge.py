"""
Local Ridge regression using basis assets

Reads config from scripts/basis_config.json and a basis list from
synthetic_data/basis_selected.csv (column 'ticker'). Loads monthly prices,
computes returns, and for each non-basis asset fits a Ridge regression of the
asset's returns on selected basis returns (nearest by distance).

Outputs:
- synthetic_data/coefficients_ridge.csv : coefficients (rows = asset, cols = basis)
- synthetic_data/recon_returns.csv      : predicted returns (same index as returns)
- synthetic_data/regression_errors.csv  : RMSE per asset
"""
from __future__ import annotations

from pathlib import Path
import json
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

import sys

# Ensure project root is on sys.path so local package imports work when running script
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from spc.graph import DistancesUtils

def load_config(config_path: Path) -> dict:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r") as f:
        return json.load(f)


def get_config_value(config: dict, *keys, default=None):
    cur = config
    for k in keys:
        cur = cur.get(k, {})
    if isinstance(cur, dict) and "value" in cur:
        return cur["value"]
    return default


def resolve_paths(config: dict, script_root: Path) -> Tuple[Path, Path]:
    prices_path = Path(get_config_value(config, "input", "prices_path", default="synthetic_data/prices_monthly.csv"))
    basis_override = get_config_value(config, "output", "basis_path", default=None)
    if basis_override:
        basis_path = Path(basis_override)
    else:
        basis_path = script_root.parent / "synthetic_data" / "basis_selected.csv"
    return prices_path, basis_path


def load_prices(prices_path: Path) -> pd.DataFrame:
    print(f"Loading prices from: {prices_path}")
    prices = pd.read_csv(prices_path, index_col=0, parse_dates=True)
    return prices.sort_index()


def load_basis_list(basis_path: Path) -> List[str]:
    print(f"Loading selected basis from: {basis_path}")
    basis_df = pd.read_csv(basis_path)
    basis_list = basis_df["ticker"].astype(str).tolist()
    return basis_list


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return DistancesUtils.price_to_return_df(prices).dropna(how="all")


def compute_distance_df(prices: pd.DataFrame, config: dict) -> pd.DataFrame:
    return DistancesUtils.price_to_distance_df(
        prices,
        min_periods=get_config_value(config, "distance_computation", "min_periods", default=1),
        corr_method=get_config_value(config, "distance_computation", "corr_method", default="pearson"),
        shrink_method=get_config_value(config, "distance_computation", "shrink_method", default=None),
        pca_n_components=get_config_value(config, "pca_denoising", "pca_n_components", default=None),
        pca_explained_variance=get_config_value(config, "pca_denoising", "pca_explained_variance", default=None),
    )


def filter_basis_in_returns(basis_list: List[str], returns: pd.DataFrame) -> List[str]:
    basis_present = [b for b in basis_list if b in returns.columns]
    if len(basis_present) != len(basis_list):
        missing = set(basis_list) - set(basis_present)
        print(f"Warning: {len(missing)} basis tickers not in price data: {missing}")
    if not basis_present:
        raise ValueError("No basis tickers found in returns data")
    return basis_present


def select_neighbors_for_asset(
    asset: str,
    basis_list: List[str],
    dist_df: pd.DataFrame,
    q_neighbors: Optional[int],
) -> List[str]:
    if asset not in dist_df.index:
        print(f"Skipping {asset}: no distance information available")
        return []
    dists = dist_df.loc[asset, basis_list].dropna()
    if dists.empty:
        print(f"Skipping {asset}: no valid distances to basis assets")
        return []
    dists = dists.sort_values()  # smaller distance = closer
    if q_neighbors is None:
        return dists.index.tolist()
    return dists.index.tolist()[:q_neighbors]


def run_local_ridge_for_asset(
    asset: str,
    returns: pd.DataFrame,
    neighbors: List[str],
    basis_list: List[str],
    ridge_alpha: float,
    min_obs: int = 10,
) -> Optional[Tuple[np.ndarray, pd.Series, dict]]:
    # Align rows where neither target nor any chosen feature is NaN
    cols = [asset] + neighbors
    df = returns[cols].dropna()
    if df.shape[0] < min_obs:
        print(f"Skipping {asset}: insufficient observations after aligning with neighbors ({df.shape[0]})")
        return None

    y = df[asset].values
    X = df[neighbors].values

    model = Ridge(alpha=ridge_alpha)
    model.fit(X, y)

    # Map coefficients back into full-basis vector (zeros where not used)
    full_coefs = np.zeros(len(basis_list))
    basis_pos = {b: i for i, b in enumerate(basis_list)}
    for nb, coef in zip(neighbors, model.coef_):
        full_coefs[basis_pos[nb]] = coef

    y_pred = model.predict(X)
    recon = pd.Series(y_pred, index=df.index, name=asset)
    rmse = np.sqrt(np.mean((y - y_pred) ** 2))
    error_row = {"asset": asset, "rmse": rmse, "n_obs": df.shape[0], "n_basis_used": len(neighbors)}

    return full_coefs, recon, error_row


def run_all_regressions(
    returns: pd.DataFrame,
    basis_list: List[str],
    dist_df: pd.DataFrame,
    ridge_alpha: float,
    q_neighbors: Optional[int],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    non_basis = [c for c in returns.columns if c not in basis_list]
    if len(non_basis) == 0:
        print("No non-basis assets to regress. Exiting.")
        return (
            pd.DataFrame(columns=basis_list),
            pd.DataFrame(index=returns.index),
            pd.DataFrame(columns=["rmse", "n_obs", "n_basis_used"]),
        )

    coef_records: Dict[str, np.ndarray] = {}
    recon_dfs: List[pd.Series] = []
    errors: List[dict] = []

    for asset in non_basis:
        neighbors = select_neighbors_for_asset(asset, basis_list, dist_df, q_neighbors)
        if not neighbors:
            continue
        result = run_local_ridge_for_asset(asset, returns, neighbors, basis_list, ridge_alpha)
        if result is None:
            continue
        full_coefs, recon, error_row = result
        coef_records[asset] = full_coefs
        recon_dfs.append(recon)
        errors.append(error_row)

    if not coef_records:
        print("No regressions were run (no sufficient data). Exiting.")
        return (
            pd.DataFrame(columns=basis_list),
            pd.DataFrame(index=returns.index),
            pd.DataFrame(columns=["rmse", "n_obs", "n_basis_used"]),
        )

    coef_df = pd.DataFrame.from_dict(coef_records, orient="index", columns=basis_list)
    recon_df = pd.concat(recon_dfs, axis=1).sort_index()
    errors_df = pd.DataFrame(errors).set_index("asset").sort_values("rmse", ascending=False)
    return coef_df, recon_df, errors_df

def save_outputs(coef_df: pd.DataFrame, recon_df: pd.DataFrame, errors_df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    coef_out = out_dir / "coefficients_ridge.csv"
    recon_out = out_dir / "recon_returns.csv"
    err_out = out_dir / "regression_errors.csv"

    coef_df.to_csv(coef_out)
    recon_df.to_csv(recon_out)
    errors_df.to_csv(err_out)

    print(f"Saved coefficients to: {coef_out}")
    print(f"Saved reconstructed returns to: {recon_out}")
    print(f"Saved regression errors to: {err_out}")
    if not errors_df.empty:
        print("Top 5 worst-fit assets:")
        print(errors_df.head(5))

def main():
    script_root = Path(__file__).parent
    config = load_config(script_root / "basis_config.json")

    prices_path, basis_path = resolve_paths(config, script_root)
    ridge_alpha = get_config_value(config, "regression", "ridge_alpha", default=1.0)
    q_neighbors = get_config_value(config, "regression", "q_neighbors", default=None)

    prices = load_prices(prices_path)
    basis_list = load_basis_list(basis_path)
    returns = compute_returns(prices)
    dist_df = compute_distance_df(prices, config)

    basis_list = filter_basis_in_returns(basis_list, returns)

    coef_df, recon_df, errors_df = run_all_regressions(
        returns=returns,
        basis_list=basis_list,
        dist_df=dist_df,
        ridge_alpha=ridge_alpha,
        q_neighbors=q_neighbors,
    )

    save_outputs(coef_df, recon_df, errors_df, out_dir=Path("synthetic_data"))


if __name__ == "__main__":
    main()