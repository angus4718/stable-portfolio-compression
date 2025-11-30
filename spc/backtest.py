"""Backtesting utilities for SPC experiments.

This module provides light-weight backtesting helpers used by the project
to evaluate index replication performance. It includes:

- `Backtester`: General utilities for computing returns, turnover, running
    RMSE and persisting evaluation outputs and diagnostic plots.
- `StrategyBacktester`: Uses regression coefficients (timeseries or snapshot)
    to map index weights into basis weights and evaluate replication performance.
- `BaselineBacktester`: Simple top-X replicator for baseline comparisons.

The code expects CSV inputs under the project `synthetic_data` and `outputs`
directories and writes evaluation artifacts under `backtest/`.
"""

from __future__ import annotations

from pathlib import Path
import json
from typing import Optional, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

import sys

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from spc.graph import DistancesUtils


class Backtester:
    """Base backtester utilities used by specific backtest implementations.

    The base class provides reusable helpers for loading price/index data,
    converting prices to returns, computing running RMSE, turnover and saving
    evaluation time-series, summary JSON and diagnostic plots.

    Attributes:
        prices (pd.DataFrame | None): Loaded price series indexed by date.
        w_spx (pd.DataFrame | None): Loaded index weights indexed by date.

    Usage:
        bt = Backtester()
        bt.load_prices_and_weights()
        returns = bt.compute_returns(prices)
        summary = bt.evaluate_and_save(...)
    """

    def __init__(self) -> None:
        self.prices: Optional[pd.DataFrame] = None
        self.w_spx: Optional[pd.DataFrame] = None

    def load_prices_and_weights(self):
        """Load prices and index weights from the project's synthetic data CSVs.

        Reads `synthetic_data/prices_monthly.csv` and
        `synthetic_data/market_index_weights.csv` into pandas DataFrames and
        stores them on the instance as ``self.prices`` and ``self.w_spx``.

        Raises:
            FileNotFoundError: If either expected CSV is missing.
        """

        prices_path = _ROOT / "synthetic_data" / "prices_monthly.csv"
        weights_path = _ROOT / "synthetic_data" / "market_index_weights.csv"
        for p in (prices_path, weights_path):
            if not p.exists():
                raise FileNotFoundError(f"Required input not found: {p}")
        self.prices = pd.read_csv(
            prices_path, index_col=0, parse_dates=True
        ).sort_index()
        self.w_spx = pd.read_csv(
            weights_path, index_col=0, parse_dates=True
        ).sort_index()

    @staticmethod
    def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
        """Convert price series to period returns.

        Args:
            prices: Price DataFrame indexed by date with asset columns.

        Returns:
            DataFrame of returns (same columns) computed by
            :py:meth:`spc.graph.DistancesUtils.price_to_return_df`.
        """
        return DistancesUtils.price_to_return_df(prices)

    @staticmethod
    def summarize(diff: pd.Series) -> dict:
        """Compute simple summary statistics for an active-return series.

        Args:
            diff: Series of active returns (replicated - index) indexed by date.

        Returns:
            A dict containing sample size, mean, sample std, RMSE and annualized std.
            Returns an empty dict when ``diff`` contains no non-NA observations.
        """
        arr = diff.dropna().values
        if arr.size == 0:
            return {}
        mean = float(np.mean(arr))
        std = float(np.std(arr, ddof=1))
        rmse = float(np.sqrt(np.mean(arr**2)))
        ann_std = std * np.sqrt(12.0)
        return {
            "n": int(arr.size),
            "mean_diff": mean,
            "std_diff": std,
            "rmse": rmse,
            "annualized_std": ann_std,
        }

    @staticmethod
    def running_rmse(series: pd.Series) -> pd.Series:
        """Compute the cumulative running RMSE for a numeric series.

        Each timestamp's value is sqrt( (x_1^2 + ... + x_t^2) / t ). Missing
        values are treated as zero for the running computation.

        Args:
            series: Numeric Series indexed by date.

        Returns:
            Series containing the running RMSE at each index position.
        """
        diffs = series.fillna(0.0).astype(float)
        cum_sq = (diffs**2).cumsum()
        counts = np.arange(1, len(diffs) + 1)
        return pd.Series(np.sqrt(cum_sq / counts), index=series.index)

    @staticmethod
    def compute_turnover(
        rep_weights: pd.DataFrame, tail_idx: List[pd.Timestamp]
    ) -> pd.Series:
        """Compute turnover for a replicate-weight time-series.

        Turnover is computed as 0.5 * sum_abs(w_t - w_{t-1}) per date. The
        function returns turnover restricted to the provided ``tail_idx`` dates.

        Args:
            rep_weights: DataFrame of replicate weights indexed by date.
            tail_idx: List of dates (timestamps) whose turnover should be returned.

        Returns:
            Series of turnover values indexed by the provided ``tail_idx``.
        """
        bw = rep_weights.fillna(0.0).astype(float)
        bw_sorted = bw.sort_index()
        prev = bw_sorted.shift(1).fillna(0.0)
        turnover = 0.5 * (bw_sorted - prev).abs().sum(axis=1)
        turnover_tail = turnover.reindex(tail_idx).fillna(0.0)
        return turnover_tail

    def save_plot(
        self,
        out_dir: Path,
        tail_df: pd.DataFrame,
        running_rmse_series: pd.Series,
        turnover_ts: Optional[pd.Series],
        plot_name: str,
    ):
        """Save three-panel diagnostic plot to the `out_dir/plots` directory.

        The plot contains:
          1. Cumulative index vs replicated cumulative returns
          2. Monthly active return and running RMSE
          3. Replicate turnover (if available)

        Args:
            out_dir: Directory under which a `plots/` subfolder will be created.
            tail_df: DataFrame with columns ['index_return','rep_return','diff'] for the tail period.
            running_rmse_series: Series of running RMSE aligned to tail_df.index.
            turnover_ts: Optional turnover Series aligned to tail_df.index.
            plot_name: Filename (not full path) for the saved PNG.
        """
        plots_dir = out_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        plot_path = plots_dir / plot_name

        cum_index = (1 + tail_df["index_return"].fillna(0.0)).cumprod() - 1
        cum_rep = (1 + tail_df["rep_return"].fillna(0.0)).cumprod() - 1

        fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
        axes[0].plot(cum_index.index, cum_index.values, label="Index Cumulative Return")
        axes[0].plot(
            cum_rep.index, cum_rep.values, label="Replicated Cumulative Return"
        )
        axes[0].legend(loc="best")
        axes[0].set_ylabel("Cumulative Return")

        axes[1].plot(
            tail_df.index,
            tail_df["diff"].fillna(0.0).values,
            label="Monthly Active Return",
            color="C1",
            alpha=0.6,
        )
        axes[1].plot(
            running_rmse_series.index,
            running_rmse_series.values,
            label="Running RMSE",
            color="C2",
        )
        axes[1].legend(loc="best")
        axes[1].set_ylabel("Active / RMSE")

        if turnover_ts is not None:
            axes[2].plot(
                turnover_ts.index,
                turnover_ts.values,
                label="Replicate Turnover",
                color="C3",
            )
            axes[2].set_ylabel("Turnover")
            axes[2].legend(loc="best")
        else:
            axes[2].text(
                0.5, 0.5, "Turnover data unavailable", ha="center", va="center"
            )
            axes[2].set_ylabel("")

        axes[2].set_xlabel("Date")
        plt.tight_layout()
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)

    def evaluate_and_save(
        self,
        idx_returns: pd.Series,
        rep_returns: pd.Series,
        rep_weights: pd.DataFrame,
        tail_percent: float,
        timeseries_name: str,
        summary_name: str,
        plot_name: str,
        extra_meta: Optional[dict] = None,
    ) -> dict:
        """Evaluate replication performance, save time-series, summary and plot.

        This function aligns the provided index and replicate returns, computes
        tail statistics (last ``tail_percent`` of available dates), computes
        running RMSE and turnover, writes a CSV timeseries and a JSON summary
        to the project's `backtest/` folder, and saves a diagnostic plot.

        Args:
            idx_returns: Series of index returns aligned to replicate timing.
            rep_returns: Series of replicated returns aligned with idx_returns.
            rep_weights: DataFrame of replicate weights used to compute turnover.
            tail_percent: Fraction (0-100) of the most recent observations used
                for tail statistics.
            timeseries_name: Filename for the saved tail timeseries CSV.
            summary_name: Filename for the saved summary JSON.
            plot_name: Filename for the saved diagnostic plot PNG.
            extra_meta: Optional dict of additional fields to include in the summary.

        Returns:
            A dict containing the computed summary statistics that were saved.
        """
        if extra_meta is None:
            extra_meta = {}

        # align and compute diff
        idx_returns = idx_returns.loc[rep_returns.index]
        rep_returns = rep_returns.loc[idx_returns.index]
        diff = rep_returns - idx_returns

        dates = diff.dropna().index
        if len(dates) == 0:
            raise RuntimeError("No overlapping return observations to evaluate.")
        n_tail = max(1, int(len(dates) * (float(tail_percent) / 100.0)))
        tail_idx = dates[-n_tail:]

        tail_idx_ret = idx_returns.loc[tail_idx]
        tail_rep_ret = rep_returns.loc[tail_idx]
        tail_diff = diff.loc[tail_idx]
        tail_df = pd.DataFrame(
            {
                "index_return": tail_idx_ret,
                "rep_return": tail_rep_ret,
                "diff": tail_diff,
            }
        )

        stats = self.summarize(tail_diff)
        try:
            r2 = float(
                r2_score(
                    tail_idx_ret.fillna(0.0).values, tail_rep_ret.fillna(0.0).values
                )
            )
        except Exception:
            r2 = float("nan")

        tail_mean = stats.get("mean_diff")
        tail_std = stats.get("std_diff")
        info_ratio = None
        if tail_mean is not None and tail_std and tail_std > 0:
            info_ratio = float(tail_mean / tail_std * np.sqrt(12.0))

        diffs_tail = tail_df["diff"].fillna(0.0).astype(float)
        running_rmse = self.running_rmse(diffs_tail)

        diffs_all = diff.dropna().astype(float)
        if len(diffs_all) > 0:
            tracking_error_rmse = float(np.sqrt(np.mean(diffs_all.values**2)))
            tracking_error_annualized = float(tracking_error_rmse * np.sqrt(12.0))
        else:
            tracking_error_rmse = None
            tracking_error_annualized = None

        turnover_ts = None
        avg_turnover_tail = None
        try:
            turnover_ts = self.compute_turnover(rep_weights, tail_idx)
            if len(tail_idx) > 0:
                avg_turnover_tail = float(turnover_ts.mean())
        except Exception:
            turnover_ts = None

        summary = {}
        summary.update(extra_meta)
        summary.update({"n_tail": stats.get("n", 0), "r2": r2})
        summary.update(stats)
        summary.update(
            {
                "information_ratio_annualized": info_ratio,
                "tracking_error_rmse": tracking_error_rmse,
                "tracking_error_annualized": tracking_error_annualized,
                "avg_turnover_tail": avg_turnover_tail,
            }
        )

        out_dir = _ROOT / "backtest"
        out_dir.mkdir(parents=True, exist_ok=True)

        timeseries_out = out_dir / timeseries_name
        tail_df.to_csv(timeseries_out)

        summary_out = out_dir / summary_name
        with open(summary_out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        running_rmse_series = pd.Series(running_rmse.values, index=diffs_tail.index)
        self.save_plot(out_dir, tail_df, running_rmse_series, turnover_ts, plot_name)

        if turnover_ts is not None:
            # persist turnover series for inspection
            turnover_name = timeseries_name.replace("timeseries", "replicate_turnover")
            turnover_path = out_dir / turnover_name
            turnover_ts.to_csv(turnover_path, index=True, header=["turnover"])

        return summary


class StrategyBacktester(Backtester):
    """Backtester that maps index weights to basis weights using coefficients.

    This backtester loads regression coefficient outputs (either a
    time-series of coefficients or a snapshot), loads the index weights and
    basis list saved under `outputs/`, and produces a time-series of basis
    weights which is then used to compute replicated returns and evaluation
    summaries.

    Notes:
        - When a coefficients time-series is present, the implementation
          aligns each index weights date to the most recent coefficient date
          at or before that date.
    """

    def __init__(self):
        super().__init__()
        # Do not store filesystem roots on the instance. `load_inputs` accepts
        # a `root` parameter and will read from the appropriate paths.
        self.basis_list: Optional[List[str]] = None
        self.coeffs_ts: Optional[pd.DataFrame] = None
        self.coeffs_snap: Optional[pd.DataFrame] = None

    def load_inputs(self):
        """Load prices, index weights, basis list and available coefficients.

        This routine loads price/index inputs (via
        :py:meth:`Backtester.load_prices_and_weights`), reads the saved basis
        list from `outputs/basis_selected.csv`, and attempts to load either a
        coefficients timeseries (`coefficients_ridge_timeseries.csv`) or a
        coefficient snapshot (`coefficients_ridge.csv`) into instance fields.
        """
        self.load_prices_and_weights()
        outputs_dir = _ROOT / "outputs"
        basis_path = outputs_dir / "basis_selected.csv"
        if not basis_path.exists():
            raise FileNotFoundError(f"Basis list not found: {basis_path}")
        basis_df = pd.read_csv(basis_path)
        self.basis_list = basis_df["ticker"].astype(str).tolist()

        coeffs_ts_path = outputs_dir / "coefficients_ridge_timeseries.csv"
        coeffs_snap_path = outputs_dir / "coefficients_ridge.csv"
        if coeffs_ts_path.exists():
            self.coeffs_ts = pd.read_csv(
                coeffs_ts_path, parse_dates=["date"]
            )  # long format
        elif coeffs_snap_path.exists():
            self.coeffs_snap = pd.read_csv(coeffs_snap_path, index_col=0)

    def compute_basis_weights_time_series(self) -> pd.DataFrame:
        """Compute basis weight time-series from coefficients or snapshot.

        When a coefficients timeseries is present, each index weight date is
        aligned with the most recent coefficient date at or before it and the
        corresponding asset->basis A matrix is constructed and applied to the
        index weights. If only a snapshot is present, the snapshot is used to
        build a single A matrix for all dates.

        Returns:
            DataFrame of basis weights indexed by date with columns equal to the
            currently loaded basis list.
        """
        if self.basis_list is None:
            raise ValueError("basis_list not loaded")
        universe = [str(c) for c in self.w_spx.columns.tolist()]
        w_spx_aligned = self.w_spx.reindex(columns=universe).fillna(0.0)

        if self.coeffs_ts is not None:
            coeffs_ts = self.coeffs_ts.copy()
            coeffs_ts["date"] = pd.to_datetime(coeffs_ts["date"])
            coeffs_by_date = {
                d: grp.pivot(index="asset", columns="basis", values="coef").fillna(0.0)
                for d, grp in coeffs_ts.groupby("date")
            }
            coeff_dates = sorted(coeffs_by_date.keys())

            basis_rows = []
            for w_date in w_spx_aligned.index:
                sel_date = None
                for cd in coeff_dates:
                    if cd <= pd.Timestamp(w_date):
                        sel_date = cd
                    else:
                        break

                A_date = pd.DataFrame(
                    0.0, index=universe, columns=self.basis_list, dtype=float
                )
                for b in self.basis_list:
                    if b in A_date.index and b in A_date.columns:
                        A_date.at[b, b] = 1.0

                if sel_date is not None:
                    pivot = coeffs_by_date[sel_date].reindex(
                        columns=self.basis_list, fill_value=0.0
                    )
                    for asset in pivot.index.astype(str):
                        if asset not in A_date.index:
                            continue
                        for b in self.basis_list:
                            A_date.at[asset, b] = (
                                float(pivot.at[asset, b]) if b in pivot.columns else 0.0
                            )

                row_sums_A = A_date.sum(axis=1)
                positive = row_sums_A > 0.0
                if positive.any():
                    A_date.loc[positive] = A_date.loc[positive].div(
                        row_sums_A[positive], axis=0
                    )

                w_row = w_spx_aligned.loc[w_date].values.reshape(1, -1)
                basis_w_row = pd.Series(
                    (w_row @ A_date.values).flatten(),
                    index=self.basis_list,
                    name=w_date,
                )
                basis_rows.append(basis_w_row)

            basis_weights = pd.DataFrame(basis_rows)
            basis_weights.index = w_spx_aligned.index
            return basis_weights

        # snapshot
        A = pd.DataFrame(0.0, index=universe, columns=self.basis_list, dtype=float)
        for b in self.basis_list:
            if b in A.index and b in A.columns:
                A.at[b, b] = 1.0
        if self.coeffs_snap is not None:
            for asset in self.coeffs_snap.index.astype(str):
                if asset not in A.index:
                    continue
                for col in self.coeffs_snap.columns:
                    col_str = str(col)
                    if col_str in A.columns:
                        A.at[asset, col_str] = float(self.coeffs_snap.loc[asset, col])

        row_sums_A = A.sum(axis=1)
        positive = row_sums_A != 0.0
        if positive.any():
            A.loc[positive] = A.loc[positive].div(row_sums_A[positive], axis=0)

        basis_weights = self.w_spx.reindex(columns=universe).fillna(0.0).dot(A)
        return basis_weights

    def run(self, tail_percent: float = 20.0):
        """Execute the strategy backtest and persist results.

        This method loads inputs, computes basis weights (time-series or
        snapshot), computes replicated returns using lagged weights, then
        delegates evaluation and persistence to
        :py:meth:`Backtester.evaluate_and_save`.

        Args:
            tail_percent: Percentage of recent observations used for tail stats.
        """
        self.load_inputs()
        basis_weights = self.compute_basis_weights_time_series()

        returns = self.compute_returns(self.prices)
        # index returns at t use weights at t-1
        w_prev = self.w_spx.shift(1).reindex(index=returns.index).fillna(0.0)
        idx_returns = (w_prev * returns).sum(axis=1)

        basis_returns = returns.reindex(columns=self.basis_list).fillna(0.0)
        b_prev = basis_weights.shift(1).reindex(index=returns.index).fillna(0.0)
        rep_returns = (b_prev * basis_returns).sum(axis=1)
        # Delegate evaluation, saving and plotting to reusable helper
        summary = self.evaluate_and_save(
            idx_returns=idx_returns,
            rep_returns=rep_returns,
            rep_weights=basis_weights,
            tail_percent=tail_percent,
            timeseries_name="backtest_timeseries.csv",
            summary_name="backtest_summary.json",
            plot_name="backtest_plot.png",
            extra_meta={"tail_percent": tail_percent},
        )

        out_dir = _ROOT / "backtest"
        print("Backtest summary (tail):")
        for k, v in summary.items():
            print(f" - {k}: {v}")
        print(f"Timeseries saved to: {out_dir / 'backtest_timeseries.csv'}")
        print(f"Summary saved to: {out_dir / 'backtest_summary.json'}")


class BaselineBacktester(Backtester):
    """Baseline replicator that uses the top-N index stocks as the basis.

    For each index date the top-N largest index constituents by weight are
    chosen and re-normalized to form replicate weights. This class provides a
    simple benchmark to compare against regression-based replication.
    """

    def __init__(self):
        super().__init__()
        self.basis_df = None

    def load_inputs(self):
        """Load prices, index weights and the saved basis DataFrame.

        The saved basis file is expected under `outputs/basis_selected.csv`.
        """
        self.load_prices_and_weights()
        outputs_dir = _ROOT / "outputs"
        basis_path = outputs_dir / "basis_selected.csv"
        if not basis_path.exists():
            raise FileNotFoundError(f"Basis list not found: {basis_path}")
        self.basis_df = pd.read_csv(basis_path)

    def compute_topx_weights_time_series(self, top_n: int) -> pd.DataFrame:
        """Construct a time-series of replicate weights using the top-N index stocks.

        For each date, the top-N constituents by index weight are selected and
        re-normalized to sum to 1. When no positive weights exist, a fallback
        uniform/nonzero distribution is used.

        Args:
            top_n: Number of top constituents to include each date.

        Returns:
            DataFrame of replicate weights indexed by date with columns equal to
            the full universe (string column names).
        """
        universe = [str(c) for c in self.w_spx.columns.tolist()]
        w_aligned = self.w_spx.reindex(columns=universe).fillna(0.0)

        rows = []
        for dt in w_aligned.index:
            w_row = w_aligned.loc[dt].astype(float)
            top = w_row.nlargest(top_n)
            total = top.sum()
            if total <= 0:
                nonzero = w_row > 0
                if nonzero.any():
                    rw = (
                        (nonzero.astype(float) / nonzero.sum())
                        .reindex(universe)
                        .fillna(0.0)
                    )
                else:
                    rw = pd.Series(0.0, index=universe)
            else:
                rw = (top / total).reindex(universe).fillna(0.0)
            rows.append(rw)

        rep_weights = pd.DataFrame(rows, index=w_aligned.index)
        rep_weights.columns = universe
        return rep_weights

    def run(self, top_n: int, tail_percent: float = 20.0):
        """Run the baseline replicate backtest using top-N constituents.

        Args:
            top_n: Number of top constituents to use for replication.
            tail_percent: Percentage of recent observations used for tail stats.
        """
        self.load_inputs()
        rep_weights = self.compute_topx_weights_time_series(top_n)

        returns = self.compute_returns(self.prices)
        w_prev = self.w_spx.shift(1).reindex(index=returns.index).fillna(0.0)
        idx_returns = (w_prev * returns).sum(axis=1)

        b_prev = rep_weights.shift(1).reindex(index=returns.index).fillna(0.0)
        rep_returns = (b_prev * returns).sum(axis=1)

        # Delegate evaluation, saving and plotting to reusable helper
        summary = self.evaluate_and_save(
            idx_returns=idx_returns,
            rep_returns=rep_returns,
            rep_weights=rep_weights,
            tail_percent=tail_percent,
            timeseries_name="baseline_backtest_timeseries.csv",
            summary_name="baseline_backtest_summary.json",
            plot_name="baseline_backtest_plot.png",
            extra_meta={"top_n": int(top_n)},
        )

        out_dir = _ROOT / "backtest"
        print("Baseline backtest summary:")
        for k, v in summary.items():
            print(f" - {k}: {v}")
        print(f"Timeseries saved to: {out_dir / 'baseline_backtest_timeseries.csv'}")
        print(f"Summary saved to: {out_dir / 'baseline_backtest_summary.json'}")
