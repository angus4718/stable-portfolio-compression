"""
Local Ridge regression using basis assets.

This script reads configuration from `scripts/basis_config.json` and a selected
basis list from `outputs/basis_selected.csv` (or an explicit path set in the
config under `output.basis_path`). It loads monthly prices (default
`synthetic_data/prices_monthly.csv`), computes simple returns, and performs
expanding-window (monthly) Ridge regressions for each non-basis asset using
nearest basis assets (by precomputed distance). The regressions record a
time-series of fitted coefficients (per date) and produce reconstructed
returns and per-asset fit diagnostics.

Key outputs (written to the `outputs/` directory by default):
- `coefficients_ridge_timeseries.csv`: long-format coefficients (date, asset, basis, coef)
- `coefficients_ridge.csv`: latest-date coefficient snapshot (asset x basis)
- `recon_returns.csv`: predicted returns (index = dates, columns = assets)
- `regression_errors.csv`: per-asset RMSE and diagnostics

Notes:
- Fits use an expanding window up to each month `t` and record coefficients at `t`.
- Predicted returns at date `t` are computed when the model's feature row at `t`
    (the selected neighbors) is fully observed.
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
    prices_path = Path(
        get_config_value(
            config, "input", "prices_path", default="synthetic_data/prices_monthly.csv"
        )
    )
    basis_path = Path(
        get_config_value(
            config, "output", "basis_path", default="outputs/basis_selected.csv"
        )
    )
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
        min_periods=get_config_value(
            config, "distance_computation", "min_periods", default=1
        ),
        corr_method=get_config_value(
            config, "distance_computation", "corr_method", default="pearson"
        ),
        shrink_method=get_config_value(
            config, "distance_computation", "shrink_method", default=None
        ),
        pca_n_components=get_config_value(
            config, "pca_denoising", "pca_n_components", default=None
        ),
        pca_explained_variance=get_config_value(
            config, "pca_denoising", "pca_explained_variance", default=None
        ),
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
        print(
            f"Skipping {asset}: insufficient observations after aligning with neighbors ({df.shape[0]})"
        )
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
    error_row = {
        "asset": asset,
        "rmse": rmse,
        "n_obs": df.shape[0],
        "n_basis_used": len(neighbors),
    }

    return full_coefs, recon, error_row


def run_all_regressions(
    returns: pd.DataFrame,
    basis_list: List[str],
    dist_df: pd.DataFrame,
    ridge_alpha: float,
    q_neighbors: Optional[int],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Perform expanding-window Ridge regressions repeated monthly.

    For each month `t` (index of `returns`) we fit a Ridge model for each
    non-basis asset using all available data up to and including `t` (after
    alignment and dropping NaNs). We then record the fitted coefficients at
    date `t` and, where the feature row at `t` is available, compute a
    predicted return for that asset at `t`.

    Outputs:
    - coef_long_df: long-format DataFrame with columns [date, asset, basis, coef]
    - recon_df: DataFrame (index = dates, columns = assets) with predicted returns at each date
    - errors_df: per-asset RMSE computed across dates where predictions were made
    """
    non_basis = [c for c in returns.columns if c not in basis_list]
    if len(non_basis) == 0:
        print("No non-basis assets to regress. Exiting.")
        return (
            pd.DataFrame(columns=["date", "asset", "basis", "coef"]),
            pd.DataFrame(index=returns.index),
            pd.DataFrame(columns=["rmse", "n_obs", "n_basis_used"]),
        )

    # Pre-compute neighbor lists (static nearest-basis selection)
    neighbor_map: Dict[str, List[str]] = {}
    for asset in non_basis:
        neighbor_map[asset] = select_neighbors_for_asset(
            asset, basis_list, dist_df, q_neighbors
        )

    # Prepare outputs
    coef_records: List[dict] = []
    recon_df = pd.DataFrame(index=returns.index, columns=non_basis, dtype=float)

    min_obs = 10

    # Iterate over expanding window end dates
    for t in returns.index:
        for asset in non_basis:
            neighbors = neighbor_map.get(asset, [])
            if not neighbors:
                continue

            cols = [asset] + neighbors
            # Training data: all rows up to and including t
            train = returns.loc[:t, cols].dropna()
            if train.shape[0] < min_obs:
                # insufficient data to fit at this date
                continue

            y = train[asset].values
            X = train[neighbors].values
            model = Ridge(alpha=ridge_alpha)
            model.fit(X, y)

            # Record coefficients for this date and asset (align to basis_list)
            for b in basis_list:
                coef_val = 0.0
                if b in neighbors:
                    try:
                        pos = neighbors.index(b)
                        coef_val = float(model.coef_[pos])
                    except Exception:
                        coef_val = 0.0
                coef_records.append(
                    {"date": t, "asset": asset, "basis": b, "coef": coef_val}
                )

            # Predict at date t if feature row available (no NaNs in neighbors at t)
            x_t = returns.loc[t, neighbors]
            if x_t.isnull().any():
                continue
            xvals = x_t.values.reshape(1, -1)
            try:
                y_pred = float(model.predict(xvals)[0])
                recon_df.at[t, asset] = y_pred
            except Exception:
                # prediction failed; skip
                continue

    # Build coefficients long DataFrame
    if coef_records:
        coef_long_df = pd.DataFrame(coef_records)
    else:
        coef_long_df = pd.DataFrame(columns=["date", "asset", "basis", "coef"])

    # Compute per-asset RMSE across dates where predictions exist
    errors: List[dict] = []
    for asset in non_basis:
        preds = recon_df[asset].dropna()
        if preds.empty:
            continue
        # align true returns
        true_vals = returns.loc[preds.index, asset]
        rmse = np.sqrt(np.mean((true_vals.values - preds.values) ** 2))
        errors.append(
            {
                "asset": asset,
                "rmse": float(rmse),
                "n_obs": int(len(preds)),
                "n_basis_used": len(neighbor_map.get(asset, [])),
            }
        )

    if errors:
        errors_df = (
            pd.DataFrame(errors).set_index("asset").sort_values("rmse", ascending=False)
        )
    else:
        errors_df = pd.DataFrame(columns=["rmse", "n_obs", "n_basis_used"])

    return coef_long_df, recon_df, errors_df


def save_outputs(
    coef_df: pd.DataFrame,
    recon_df: pd.DataFrame,
    errors_df: pd.DataFrame,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    coef_out = out_dir / "coefficients_ridge.csv"
    recon_out = out_dir / "recon_returns.csv"
    err_out = out_dir / "regression_errors.csv"

    # coef_df may be in one of two shapes:
    # - long-format DataFrame with columns [date, asset, basis, coef]
    # - wide-format DataFrame (asset x basis)
    if isinstance(coef_df, pd.DataFrame) and set(
        ["date", "asset", "basis", "coef"]
    ).issubset(set(coef_df.columns)):
        # save long-format as CSV
        coef_out_ts = out_dir / "coefficients_ridge_timeseries.csv"
        coef_df.to_csv(coef_out_ts, index=False)
        print(f"Saved time-series coefficients (long) to: {coef_out_ts}")
        # Also save a pivoted wide-format snapshot for the last date if available
        try:
            last_date = coef_df["date"].max()
            pivot = (
                coef_df[coef_df["date"] == last_date]
                .pivot(index="asset", columns="basis", values="coef")
                .fillna(0.0)
            )
            pivot.to_csv(coef_out)
            print(f"Saved latest-date coefficient snapshot to: {coef_out}")
        except Exception:
            # fallback: write an empty wide file
            pd.DataFrame().to_csv(coef_out)
    else:
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

    # Save regression outputs to `outputs/` (not to synthetic_data)
    save_outputs(
        coef_df,
        recon_df,
        errors_df,
        out_dir=script_root.parent / "outputs",
    )


if __name__ == "__main__":
    main()
