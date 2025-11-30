from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from spc.graph import DistancesUtils


class LocalRidgeRunner:
    """Local neighbor-based Ridge regression pipeline.

    Usage: LocalRidgeRunner(config).run()
    """

    def __init__(self, config: Dict):
        self.config = config

    def _C(self, *keys, default=None):
        """Compact accessor for nested config `...{'value': v}` entries."""
        cur = self.config
        for k in keys:
            cur = cur.get(k, {})
        if isinstance(cur, dict) and "value" in cur:
            return cur["value"]
        return default

    def _load_prices(self, prices_path: Path) -> pd.DataFrame:
        prices = pd.read_csv(prices_path, index_col=0, parse_dates=True)
        return prices.sort_index()

    def _load_basis_list(self, basis_path: Path) -> List[str]:
        df = pd.read_csv(basis_path)
        if "ticker" in df.columns:
            return df["ticker"].astype(str).tolist()
        return df.iloc[:, 0].astype(str).tolist()

    def _compute_returns(self, prices: pd.DataFrame) -> pd.DataFrame:
        return DistancesUtils.price_to_return_df(prices).dropna(how="all")

    def _compute_distance_df(self, prices: pd.DataFrame) -> pd.DataFrame:
        return DistancesUtils.price_to_distance_df(
            prices,
            min_periods=self._C("distance_computation", "min_periods", default=1),
            corr_method=self._C(
                "distance_computation", "corr_method", default="pearson"
            ),
            shrink_method=self._C(
                "distance_computation", "shrink_method", default=None
            ),
            pca_n_components=self._C("pca_denoising", "pca_n_components", default=None),
            pca_explained_variance=self._C(
                "pca_denoising", "pca_explained_variance", default=None
            ),
        )

    def _filter_basis(self, basis_list: List[str], returns: pd.DataFrame) -> List[str]:
        basis_present = [b for b in basis_list if b in returns.columns]
        if len(basis_present) != len(basis_list):
            missing = set(basis_list) - set(basis_present)
            print(f"Warning: {len(missing)} basis tickers not in price data: {missing}")
        if not basis_present:
            raise ValueError("No basis tickers found in returns data")
        return basis_present

    def _select_neighbors_for_asset(
        self,
        asset: str,
        basis_list: List[str],
        dist_df: pd.DataFrame,
        q_neighbors: Optional[int],
    ) -> List[str]:
        if asset not in dist_df.index:
            return []
        dists = dist_df.loc[asset, basis_list].dropna()
        if dists.empty:
            return []
        dists = dists.sort_values()
        if q_neighbors is None:
            return dists.index.tolist()
        return dists.index.tolist()[:q_neighbors]

    def _run_all_regressions(
        self,
        returns: pd.DataFrame,
        basis_list: List[str],
        prices: pd.DataFrame,
        ridge_alpha: float,
        q_neighbors: Optional[int],
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        non_basis = [c for c in returns.columns if c not in basis_list]
        if len(non_basis) == 0:
            return (
                pd.DataFrame(columns=["date", "asset", "basis", "coef"]),
                pd.DataFrame(index=returns.index),
                pd.DataFrame(columns=["rmse", "n_obs", "n_basis_used"]),
            )

        # track which neighbors were used for each asset (aggregate over time)
        neighbor_used_sets: Dict[str, set] = {asset: set() for asset in non_basis}

        coef_records: List[dict] = []
        recon_df = pd.DataFrame(index=returns.index, columns=non_basis, dtype=float)

        min_obs = 10

        # We'll compute distances in a time-local fashion to avoid look-ahead.
        # Use a small cache keyed by (end_date, window_size, shrink_method, pca params).
        dist_cache: Dict[tuple, pd.DataFrame] = {}

        # distance window and shrink settings from config
        dist_window = self._C("distance_computation", "window_size", default=None)
        shrink_method = self._C("distance_computation", "shrink_method", default=None)
        pca_n_components = self._C("pca_denoising", "pca_n_components", default=None)
        pca_explained_variance = self._C(
            "pca_denoising", "pca_explained_variance", default=None
        )

        for t in returns.index:
            # build / fetch distance matrix up to time t (expanding or rolling)
            key = (
                pd.Timestamp(t),
                int(dist_window) if dist_window is not None else None,
                shrink_method,
                pca_n_components,
                pca_explained_variance,
            )
            if key in dist_cache:
                dist_t = dist_cache[key]
            else:
                dist_t = DistancesUtils.windowed_price_to_distance(
                    prices,
                    t,
                    window=dist_window,
                    min_periods=self._C(
                        "distance_computation", "min_periods", default=1
                    ),
                    corr_method=self._C(
                        "distance_computation", "corr_method", default="pearson"
                    ),
                    shrink_method=shrink_method,
                    pca_n_components=pca_n_components,
                    pca_explained_variance=pca_explained_variance,
                )
                dist_cache[key] = dist_t

            for asset in non_basis:
                # select neighbors using the time-local distance matrix
                neighbors = self._select_neighbors_for_asset(
                    asset, basis_list, dist_t, q_neighbors
                )
                if not neighbors:
                    continue
                neighbor_used_sets[asset].update(neighbors)
                cols = [asset] + neighbors
                train = returns.loc[:t, cols].dropna()
                if train.shape[0] < min_obs:
                    continue
                y = train[asset].values
                X = train[neighbors].values
                model = Ridge(alpha=ridge_alpha)
                model.fit(X, y)
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
                x_t = returns.loc[t, neighbors]
                if x_t.isnull().any():
                    continue
                xvals = x_t.values.reshape(1, -1)
                try:
                    y_pred = float(model.predict(xvals)[0])
                    recon_df.at[t, asset] = y_pred
                except Exception:
                    continue

        coef_long_df = (
            pd.DataFrame(coef_records)
            if coef_records
            else pd.DataFrame(columns=["date", "asset", "basis", "coef"])
        )

        errors = []
        for asset in non_basis:
            preds = recon_df[asset].dropna()
            if preds.empty:
                continue
            true_vals = returns.loc[preds.index, asset]
            rmse = np.sqrt(np.mean((true_vals.values - preds.values) ** 2))
            errors.append(
                {
                    "asset": asset,
                    "rmse": float(rmse),
                    "n_obs": int(len(preds)),
                    "n_basis_used": len(neighbor_used_sets.get(asset, set())),
                }
            )

        errors_df = (
            pd.DataFrame(errors).set_index("asset").sort_values("rmse", ascending=False)
            if errors
            else pd.DataFrame(columns=["rmse", "n_obs", "n_basis_used"])
        )

        return coef_long_df, recon_df, errors_df

    def _save_outputs(
        self,
        coef_df: pd.DataFrame,
        recon_df: pd.DataFrame,
        errors_df: pd.DataFrame,
        out_dir: Path,
    ) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        coef_out_ts = out_dir / "coefficients_ridge_timeseries.csv"
        coef_df.to_csv(coef_out_ts, index=False)
        print(f"Saved time-series coefficients (long) to: {coef_out_ts}")
        try:
            last_date = coef_df["date"].max()
            pivot = (
                coef_df[coef_df["date"] == last_date]
                .pivot(index="asset", columns="basis", values="coef")
                .fillna(0.0)
            )
            pivot.to_csv(out_dir / "coefficients_ridge.csv")
            print(
                f"Saved latest-date coefficient snapshot to: {out_dir / 'coefficients_ridge.csv'}"
            )
        except Exception:
            pd.DataFrame().to_csv(out_dir / "coefficients_ridge.csv")
        recon_df.to_csv(out_dir / "recon_returns.csv")
        errors_df.to_csv(out_dir / "regression_errors.csv")

    def run(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        prices_path = Path(
            self._C("input", "prices_path", default="synthetic_data/prices_monthly.csv")
        )
        basis_path = Path(
            self._C("output", "basis_path", default="outputs/basis_selected.csv")
        )
        ridge_alpha = self._C("regression", "ridge_alpha", default=1.0)
        q_neighbors = self._C("regression", "q_neighbors", default=None)

        prices = self._load_prices(prices_path)
        basis_list = self._load_basis_list(basis_path)
        returns = self._compute_returns(prices)

        basis_list = self._filter_basis(basis_list, returns)

        coef_df, recon_df, errors_df = self._run_all_regressions(
            returns=returns,
            basis_list=basis_list,
            prices=prices,
            ridge_alpha=ridge_alpha,
            q_neighbors=q_neighbors,
        )

        out_dir = _ROOT / "outputs"
        self._save_outputs(coef_df, recon_df, errors_df, out_dir)
        return coef_df, recon_df, errors_df


class WeightMapper:
    """Compute basis weights from index weights using Ridge coefficients.

    - Accepts a parsed config dict.
    - `run()` computes time-series basis weights (preferred) or falls back to snapshot,
      normalizes rows, writes outputs into `_ROOT / 'outputs'`, and returns the
      computed basis_weights DataFrame.
    """

    def __init__(self, config: Dict):
        self.config = config

    def _load_basis_list(self, basis_path: Path) -> list[str]:
        basis_df = pd.read_csv(basis_path)
        if "ticker" in basis_df.columns:
            return basis_df["ticker"].astype(str).tolist()
        return basis_df.iloc[:, 0].astype(str).tolist()

    def _resolve_paths(self):
        outputs_dir = _ROOT / "outputs"
        outputs_dir.mkdir(parents=True, exist_ok=True)

        basis_cfg_val = (
            self.config.get("output", {})
            .get("basis_path", {})
            .get("value", "outputs/basis_selected.csv")
        )
        basis_path = Path(basis_cfg_val)
        if not basis_path.exists():
            alt = outputs_dir / basis_path.name
            if alt.exists():
                basis_path = alt

        coeffs_path = outputs_dir / "coefficients_ridge.csv"
        coeffs_ts_path = outputs_dir / "coefficients_ridge_timeseries.csv"

        weights_cfg_val = (
            self.config.get("input", {})
            .get("weights_path", {})
            .get("value", "synthetic_data/market_index_weights.csv")
        )
        weights_path = Path(weights_cfg_val)
        if not weights_path.exists():
            altw = _ROOT / weights_path
            if altw.exists():
                weights_path = altw

        return outputs_dir, basis_path, coeffs_path, coeffs_ts_path, weights_path

    def _compute_from_timeseries(
        self, coeffs_ts: pd.DataFrame, w_spx: pd.DataFrame, basis_list: list[str]
    ) -> pd.DataFrame:
        coeffs_ts["date"] = pd.to_datetime(coeffs_ts["date"])
        coeffs_ts = coeffs_ts.sort_values("date")
        coeffs_by_date = {}
        for d, grp in coeffs_ts.groupby("date"):
            pivot = grp.pivot(index="asset", columns="basis", values="coef").fillna(0.0)
            pivot = pivot.reindex(columns=basis_list, fill_value=0.0)
            coeffs_by_date[pd.Timestamp(d)] = pivot

        coeff_dates = sorted(coeffs_by_date.keys())

        universe = [str(c) for c in w_spx.columns]
        basis_rows = []
        for w_date in w_spx.index:
            sel_date = None
            for cd in coeff_dates:
                if cd <= pd.Timestamp(w_date):
                    sel_date = cd
                else:
                    break

            A_date = pd.DataFrame(0.0, index=universe, columns=basis_list, dtype=float)
            for b in basis_list:
                if b in A_date.index and b in A_date.columns:
                    A_date.at[b, b] = 1.0

            if sel_date is not None:
                pivot = coeffs_by_date[sel_date]
                for asset in pivot.index.astype(str):
                    if asset not in A_date.index:
                        continue
                    for b in basis_list:
                        A_date.at[asset, b] = (
                            float(pivot.at[asset, b]) if b in pivot.columns else 0.0
                        )

            row_sums_A = A_date.sum(axis=1)
            positive_rows = row_sums_A > 0.0
            if positive_rows.any():
                A_date.loc[positive_rows] = A_date.loc[positive_rows].div(
                    row_sums_A[positive_rows], axis=0
                )

            w_row = w_spx.loc[w_date].values.reshape(1, -1)
            basis_w_row = pd.Series(
                (w_row @ A_date.values).flatten(), index=basis_list, name=w_date
            )
            basis_rows.append(basis_w_row)

        basis_weights = pd.DataFrame(basis_rows)
        basis_weights.index = w_spx.index
        return basis_weights

    def _compute_from_snapshot(
        self, coeffs: pd.DataFrame, w_spx: pd.DataFrame, basis_list: list[str]
    ) -> pd.DataFrame:
        universe = [str(c) for c in w_spx.columns]
        A = pd.DataFrame(0.0, index=universe, columns=basis_list, dtype=float)
        for b in basis_list:
            if b in A.index and b in A.columns:
                A.at[b, b] = 1.0
        for asset in coeffs.index.astype(str):
            if asset not in A.index:
                continue
            for col in coeffs.columns:
                col_str = str(col)
                if col_str in A.columns:
                    A.at[asset, col_str] = float(coeffs.loc[asset, col])
        row_sums_A = A.sum(axis=1)
        positive_rows = row_sums_A != 0.0
        if positive_rows.any():
            A.loc[positive_rows] = A.loc[positive_rows].div(
                row_sums_A[positive_rows], axis=0
            )
        basis_weights = w_spx.reindex(columns=universe).fillna(0.0).dot(A)
        return basis_weights

    def _normalize_and_save_single_row(
        self, basis_weights: pd.DataFrame, out_dir: Path
    ) -> None:
        tol = 1e-6
        row_sums = basis_weights.sum(axis=1)
        is_unit = row_sums.apply(lambda x: np.isclose(x, 1.0, atol=tol))
        n_rows = len(row_sums)
        n_unit = int(is_unit.sum())
        n_zero = int((row_sums == 0.0).sum())

        if n_unit == n_rows and n_zero == 0:
            out_path = out_dir / "basis_weights.csv"
            basis_weights.to_csv(out_path, index=True)
            print(
                f"Saved basis weights to: {out_path} (all rows sum to 1 within tol={tol})"
            )
            return

        raw_out = out_dir / "basis_weights_raw.csv"
        basis_weights.to_csv(raw_out, index=True)
        print(f"Saved raw basis weights to: {raw_out}")

        normalized = basis_weights.copy()
        positive = row_sums > 0.0
        normalized.loc[positive] = normalized.loc[positive].div(
            row_sums[positive], axis=0
        )

        norm_out = out_dir / "basis_weights.csv"
        normalized.to_csv(norm_out, index=True)
        print(
            f"Saved normalized basis weights to: {norm_out} (rows with sum>0 scaled to sum=1)"
        )

    def run(self) -> pd.DataFrame:
        out_dir, basis_path, coeffs_path, coeffs_ts_path, weights_path = (
            self._resolve_paths()
        )

        print(f"Loading basis list from: {basis_path}")
        basis_list = self._load_basis_list(basis_path)

        coeffs = None
        coeffs_ts = None
        if coeffs_ts_path.exists():
            print(f"Loading time-series coefficients from: {coeffs_ts_path}")
            coeffs_ts = pd.read_csv(coeffs_ts_path, parse_dates=["date"])
        elif coeffs_path.exists():
            print(f"Loading coefficient snapshot from: {coeffs_path}")
            coeffs = pd.read_csv(coeffs_path, index_col=0)
        else:
            raise FileNotFoundError(
                f"Coefficients file not found: {coeffs_ts_path} or {coeffs_path}"
            )

        print(f"Loading index weights from: {weights_path}")
        if weights_path.exists():
            w_spx = pd.read_csv(weights_path, index_col=0, parse_dates=True)
        else:
            raise FileNotFoundError(f"Weights file not found: {weights_path}")

        universe = [str(c) for c in w_spx.columns.tolist()]
        w_spx_aligned = w_spx.reindex(columns=universe).fillna(0.0)

        if coeffs_ts is not None:
            basis_weights = self._compute_from_timeseries(
                coeffs_ts, w_spx_aligned, basis_list
            )
        else:
            basis_weights = self._compute_from_snapshot(
                coeffs, w_spx_aligned, basis_list
            )

        self._normalize_and_save_single_row(basis_weights, out_dir)

        # Return final basis_weights
        final_path = out_dir / "basis_weights.csv"
        if final_path.exists():
            return pd.read_csv(final_path, index_col=0)
        return basis_weights
