from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Dict, List

import pandas as pd
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from spc.portfolio_construction import LocalRidgeRunner
from spc.graph import DistancesUtils


def load_config(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    cfg_path = Path(__file__).resolve().parent / "local_ridge_timeseries_config.json"
    cfg = load_config(cfg_path)

    prices_path = Path(cfg.get("prices_path", "synthetic_data/prices_monthly.csv"))
    basis_ts_path = Path(
        cfg.get("basis_timeseries_path", "analysis_outputs/basis_timeseries.csv")
    )
    out_dir = Path(cfg.get("output_path", "analysis_outputs"))
    out_dir = out_dir if out_dir.is_absolute() else _ROOT / out_dir
    ridge_alpha = float(cfg.get("ridge_alpha", 0.0))
    q_neighbors = int(cfg.get("q_neighbors", 3))
    min_periods = int(cfg.get("min_periods", 1))
    corr_method = cfg.get("corr_method", "pearson")
    shrink_method = cfg.get("shrink_method", None)
    pca_n_components = (
        int(cfg.get("pca_n_components", None))
        if cfg.get("pca_n_components", None) is not None
        else None
    )
    pca_explained_variance = (
        float(cfg.get("pca_explained_variance", None))
        if cfg.get("pca_explained_variance", None) is not None
        else None
    )

    prices = pd.read_csv(prices_path, index_col=0, parse_dates=True)
    basis_ts = (
        pd.read_csv(basis_ts_path, parse_dates=["date"])
        if basis_ts_path.exists()
        else pd.DataFrame(columns=["date", "basis"])
    )

    runner = LocalRidgeRunner(
        prices_path=prices_path,
        basis_path=basis_ts_path,
        ridge_alpha=ridge_alpha,
        q_neighbors=q_neighbors,
        min_periods=min_periods,
        corr_method=corr_method,
        shrink_method=shrink_method,
        pca_n_components=pca_n_components,
        pca_explained_variance=pca_explained_variance,
        out_dir=out_dir,
    )

    coeffs: List[pd.DataFrame] = []
    errors: List[pd.DataFrame] = []

    for row in tqdm(
        basis_ts.itertuples(index=False), total=len(basis_ts), desc="Dates", unit="date"
    ):
        t = getattr(row, "date")
        basis_str = getattr(row, "basis") if hasattr(row, "basis") else None
        basis_list = (
            []
            if pd.isna(basis_str) or basis_str == ""
            else [s for s in basis_str.split(";") if s]
        )

        # use price history strictly before t
        past_prices = prices[prices.index < t]
        if past_prices.shape[0] < 1:
            continue

        returns = DistancesUtils.price_to_return_df(past_prices).dropna(how="all")
        if returns.empty:
            continue

        # filter basis to those present
        basis_list = [b for b in basis_list if b in returns.columns]
        if not basis_list:
            continue

        coef_long_df, _, errors_df = runner.run_all_regressions(
            returns=returns,
            basis_list=basis_list,
            prices=past_prices,
            ridge_alpha=runner.ridge_alpha,
            q_neighbors=runner.q_neighbors,
        )

        if not coef_long_df.empty:
            coef_long_df = coef_long_df.copy()
            coef_long_df["date"] = pd.Timestamp(t)
            coeffs.append(coef_long_df)

        if not errors_df.empty:
            e = errors_df.reset_index().copy()
            e["date"] = pd.Timestamp(t)
            errors.append(e)

    out_dir.mkdir(parents=True, exist_ok=True)

    if coeffs:
        all_coefs = pd.concat(coeffs, ignore_index=True)
        all_coefs.to_csv(
            out_dir / "coefficients_ridge_timeseries_by_date.csv", index=False
        )
        print(f"Wrote coefficients timeseries rows: {len(all_coefs)}")
    else:
        print("No coefficients were produced.")

    if errors:
        all_err = pd.concat(errors, ignore_index=True)
        all_err.to_csv(out_dir / "regression_errors_by_date.csv", index=False)
        print(f"Wrote errors timeseries rows: {len(all_err)}")
    else:
        print("No error records were produced.")


if __name__ == "__main__":
    main()
