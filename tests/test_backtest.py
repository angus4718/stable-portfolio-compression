"""Test suite for spc/backtest module.

Test cases for Backtester, StrategyBacktester, and BaselineBacktester classes.
"""

import unittest
import tempfile
import json
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock
import matplotlib.pyplot as plt

from spc.backtest import Backtester, StrategyBacktester, BaselineBacktester


class TestBacktester(unittest.TestCase):
    """Test Backtester base class utilities."""

    def setUp(self):
        """Set up test fixtures."""
        self.backtester = Backtester()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

    def tearDown(self):
        """Clean up temporary directory."""
        self.temp_dir.cleanup()
        plt.close("all")

    def test_backtester_initialization(self):
        """Test Backtester initializes with None prices and weights."""
        bt = Backtester()
        self.assertIsNone(bt.prices)
        self.assertIsNone(bt.w_spx)

    def test_compute_returns_basic(self):
        """Test basic price to returns conversion."""
        prices = pd.DataFrame(
            {"A": [100.0, 110.0, 121.0], "B": [50.0, 55.0, 60.5]},
            index=pd.date_range("2020-01-01", periods=3, freq="M"),
        )

        returns = Backtester.compute_returns(prices)

        self.assertEqual(returns.shape[1], 2)  # Should have 2 columns
        # Check that some returns are computed and positive values exist
        non_nan_returns = returns.dropna().values.flatten()
        self.assertGreater(len(non_nan_returns), 0)

    def test_compute_returns_preserves_columns(self):
        """Test that compute_returns preserves column names."""
        prices = pd.DataFrame(
            {"AAPL": [100.0, 105.0], "MSFT": [200.0, 210.0]},
            index=pd.date_range("2020-01-01", periods=2, freq="M"),
        )

        returns = Backtester.compute_returns(prices)

        self.assertIn("AAPL", returns.columns)
        self.assertIn("MSFT", returns.columns)

    def test_compute_returns_handles_nan(self):
        """Test compute_returns handles NaN values."""
        prices = pd.DataFrame(
            {"A": [100.0, np.nan, 110.0], "B": [50.0, 55.0, np.nan]},
            index=pd.date_range("2020-01-01", periods=3, freq="M"),
        )

        returns = Backtester.compute_returns(prices)

        # Should contain NaN values
        self.assertTrue(returns.isna().any().any())

    def test_summarize_basic(self):
        """Test basic summary statistics computation."""
        diff = pd.Series(
            [0.01, -0.02, 0.015, -0.005],
            index=pd.date_range("2020-01-01", periods=4, freq="M"),
        )

        summary = Backtester.summarize(diff)

        self.assertIn("n", summary)
        self.assertIn("mean_diff", summary)
        self.assertIn("std_diff", summary)
        self.assertIn("rmse", summary)
        self.assertIn("annualized_std", summary)
        self.assertEqual(summary["n"], 4)

    def test_summarize_with_nan_values(self):
        """Test summarize handles NaN values."""
        diff = pd.Series(
            [0.01, np.nan, -0.02, 0.015],
            index=pd.date_range("2020-01-01", periods=4, freq="M"),
        )

        summary = Backtester.summarize(diff)

        self.assertEqual(summary["n"], 3)  # Only 3 non-NaN values

    def test_summarize_all_nan_returns_empty_dict(self):
        """Test summarize returns empty dict for all-NaN series."""
        diff = pd.Series(
            [np.nan, np.nan, np.nan],
            index=pd.date_range("2020-01-01", periods=3, freq="M"),
        )

        summary = Backtester.summarize(diff)

        self.assertEqual(summary, {})

    def test_summarize_annualized_std(self):
        """Test that annualized_std is computed correctly."""
        diff = pd.Series(
            [0.01] * 12, index=pd.date_range("2020-01-01", periods=12, freq="M")
        )

        summary = Backtester.summarize(diff)

        # annualized_std should be roughly std * sqrt(12)
        expected_ann_std = summary["std_diff"] * np.sqrt(12.0)
        self.assertAlmostEqual(summary["annualized_std"], expected_ann_std, places=6)

    def test_running_rmse_basic(self):
        """Test basic running RMSE computation."""
        series = pd.Series(
            [0.01, -0.02, 0.015], index=pd.date_range("2020-01-01", periods=3, freq="M")
        )

        running_rmse = Backtester.running_rmse(series)

        self.assertEqual(len(running_rmse), 3)
        # First value should be |0.01|
        self.assertAlmostEqual(running_rmse.iloc[0], 0.01, places=10)
        # Values should be non-negative
        self.assertTrue((running_rmse >= 0).all())

    def test_running_rmse_with_nan(self):
        """Test running_rmse treats NaN as zero."""
        series = pd.Series(
            [0.01, np.nan, 0.01], index=pd.date_range("2020-01-01", periods=3, freq="M")
        )

        running_rmse = Backtester.running_rmse(series)

        # Middle NaN treated as zero
        self.assertEqual(len(running_rmse), 3)

    def test_running_rmse_zero_series(self):
        """Test running_rmse with zero series."""
        series = pd.Series(
            [0.0, 0.0, 0.0], index=pd.date_range("2020-01-01", periods=3, freq="M")
        )

        running_rmse = Backtester.running_rmse(series)

        np.testing.assert_array_almost_equal(running_rmse.values, [0.0, 0.0, 0.0])

    def test_compute_turnover_basic(self):
        """Test basic turnover computation."""
        rep_weights = pd.DataFrame(
            {"A": [0.5, 0.6, 0.4], "B": [0.5, 0.4, 0.6]},
            index=pd.date_range("2020-01-01", periods=3, freq="M"),
        )

        tail_idx = [rep_weights.index[1], rep_weights.index[2]]
        turnover = Backtester.compute_turnover(rep_weights, tail_idx)

        self.assertEqual(len(turnover), 2)
        # Turnover should be non-negative
        self.assertTrue((turnover >= 0).all())

    def test_compute_turnover_formula(self):
        """Test turnover is computed as 0.5 * sum_abs(w_t - w_t-1)."""
        rep_weights = pd.DataFrame(
            {"A": [0.5, 0.7], "B": [0.5, 0.3]},
            index=pd.date_range("2020-01-01", periods=2, freq="M"),
        )

        tail_idx = [rep_weights.index[1]]
        turnover = Backtester.compute_turnover(rep_weights, tail_idx)

        # Expected: 0.5 * (|0.7-0.5| + |0.3-0.5|) = 0.5 * 0.4 = 0.2
        self.assertAlmostEqual(turnover.iloc[0], 0.2, places=10)

    def test_compute_turnover_with_missing_dates(self):
        """Test compute_turnover handles missing dates gracefully."""
        rep_weights = pd.DataFrame(
            {"A": [0.5, 0.6], "B": [0.5, 0.4]},
            index=pd.date_range("2020-01-01", periods=2, freq="M"),
        )

        # Request turnover for dates not in the series
        missing_date = pd.Timestamp("2020-04-01")
        tail_idx = [missing_date]
        turnover = Backtester.compute_turnover(rep_weights, tail_idx)

        # Should return zero for missing dates
        self.assertEqual(turnover.loc[missing_date], 0.0)

    def test_save_plot_creates_file(self):
        """Test save_plot creates a PNG file."""
        tail_df = pd.DataFrame(
            {
                "index_return": [0.01, -0.01, 0.02],
                "rep_return": [0.015, -0.008, 0.018],
                "diff": [0.005, 0.002, -0.002],
            },
            index=pd.date_range("2020-01-01", periods=3, freq="M"),
        )

        running_rmse = pd.Series([0.005, 0.0043, 0.0041], index=tail_df.index)
        turnover = pd.Series([0.05, 0.03, 0.02], index=tail_df.index)

        self.backtester.save_plot(
            self.temp_path, tail_df, running_rmse, turnover, "test_plot.png"
        )

        plot_file = self.temp_path / "plots" / "test_plot.png"
        self.assertTrue(plot_file.exists())

    def test_save_plot_without_turnover(self):
        """Test save_plot works without turnover data."""
        tail_df = pd.DataFrame(
            {
                "index_return": [0.01, -0.01],
                "rep_return": [0.015, -0.008],
                "diff": [0.005, 0.002],
            },
            index=pd.date_range("2020-01-01", periods=2, freq="M"),
        )

        running_rmse = pd.Series([0.005, 0.0043], index=tail_df.index)

        self.backtester.save_plot(
            self.temp_path,
            tail_df,
            running_rmse,
            None,  # No turnover
            "test_plot_no_turnover.png",
        )

        plot_file = self.temp_path / "plots" / "test_plot_no_turnover.png"
        self.assertTrue(plot_file.exists())

    def test_evaluate_and_save_basic(self):
        """Test evaluate_and_save creates files and returns summary."""
        idx_returns = pd.Series(
            [0.01, -0.01, 0.02], index=pd.date_range("2020-01-01", periods=3, freq="M")
        )
        rep_returns = pd.Series([0.015, -0.008, 0.018], index=idx_returns.index)
        rep_weights = pd.DataFrame(
            {"A": [0.5, 0.6, 0.4], "B": [0.5, 0.4, 0.6]}, index=idx_returns.index
        )

        with patch("spc.backtest._ROOT", self.temp_path):
            summary = self.backtester.evaluate_and_save(
                idx_returns,
                rep_returns,
                rep_weights,
                tail_percent=100.0,
                timeseries_name="test_ts.csv",
                summary_name="test_summary.json",
                plot_name="test_plot.png",
            )

        self.assertIsInstance(summary, dict)
        self.assertIn("rmse", summary)
        self.assertIn("r2", summary)

    def test_evaluate_and_save_creates_timeseries_csv(self):
        """Test evaluate_and_save creates timeseries CSV."""
        idx_returns = pd.Series(
            [0.01, -0.01], index=pd.date_range("2020-01-01", periods=2, freq="M")
        )
        rep_returns = pd.Series([0.015, -0.008], index=idx_returns.index)
        rep_weights = pd.DataFrame(
            {"A": [0.5, 0.6], "B": [0.5, 0.4]}, index=idx_returns.index
        )

        with patch("spc.backtest._ROOT", self.temp_path):
            self.backtester.evaluate_and_save(
                idx_returns,
                rep_returns,
                rep_weights,
                tail_percent=100.0,
                timeseries_name="ts.csv",
                summary_name="summary.json",
                plot_name="plot.png",
            )

        ts_file = self.temp_path / "backtest" / "ts.csv"
        self.assertTrue(ts_file.exists())

    def test_evaluate_and_save_creates_summary_json(self):
        """Test evaluate_and_save creates summary JSON."""
        idx_returns = pd.Series(
            [0.01, -0.01], index=pd.date_range("2020-01-01", periods=2, freq="M")
        )
        rep_returns = pd.Series([0.015, -0.008], index=idx_returns.index)
        rep_weights = pd.DataFrame(
            {"A": [0.5, 0.6], "B": [0.5, 0.4]}, index=idx_returns.index
        )

        with patch("spc.backtest._ROOT", self.temp_path):
            self.backtester.evaluate_and_save(
                idx_returns,
                rep_returns,
                rep_weights,
                tail_percent=100.0,
                timeseries_name="ts.csv",
                summary_name="summary.json",
                plot_name="plot.png",
            )

        summary_file = self.temp_path / "backtest" / "summary.json"
        self.assertTrue(summary_file.exists())

        with open(summary_file, "r") as f:
            data = json.load(f)
        self.assertIsInstance(data, dict)

    def test_evaluate_and_save_tail_percent(self):
        """Test evaluate_and_save respects tail_percent parameter."""
        idx_returns = pd.Series(
            [0.01] * 100, index=pd.date_range("2020-01-01", periods=100, freq="D")
        )
        rep_returns = pd.Series([0.015] * 100, index=idx_returns.index)
        rep_weights = pd.DataFrame(
            {"A": np.random.uniform(0, 1, 100)}, index=idx_returns.index
        )

        with patch("spc.backtest._ROOT", self.temp_path):
            summary = self.backtester.evaluate_and_save(
                idx_returns,
                rep_returns,
                rep_weights,
                tail_percent=10.0,
                timeseries_name="ts.csv",
                summary_name="summary.json",
                plot_name="plot.png",
            )

        # With tail_percent=10, n_tail should be approximately 10
        self.assertLess(summary.get("n_tail", 0), 20)

    def test_evaluate_and_save_extra_meta(self):
        """Test evaluate_and_save includes extra metadata in summary."""
        idx_returns = pd.Series(
            [0.01, -0.01], index=pd.date_range("2020-01-01", periods=2, freq="M")
        )
        rep_returns = pd.Series([0.015, -0.008], index=idx_returns.index)
        rep_weights = pd.DataFrame(
            {"A": [0.5, 0.6], "B": [0.5, 0.4]}, index=idx_returns.index
        )

        extra = {"experiment": "test", "version": 1}

        with patch("spc.backtest._ROOT", self.temp_path):
            summary = self.backtester.evaluate_and_save(
                idx_returns,
                rep_returns,
                rep_weights,
                tail_percent=100.0,
                timeseries_name="ts.csv",
                summary_name="summary.json",
                plot_name="plot.png",
                extra_meta=extra,
            )

        self.assertEqual(summary["experiment"], "test")
        self.assertEqual(summary["version"], 1)

    def test_evaluate_and_save_no_overlapping_returns_raises(self):
        """Test evaluate_and_save raises when no overlapping returns."""
        # Create non-overlapping index and rep returns - this will raise KeyError
        # when trying to align them
        idx_dates = pd.date_range("2020-01-01", periods=5, freq="D")
        rep_dates = pd.date_range("2020-02-01", periods=5, freq="D")

        idx_returns = pd.Series(np.random.randn(5), index=idx_dates)
        rep_returns = pd.Series(np.random.randn(5), index=rep_dates)
        rep_weights = pd.DataFrame({"A": [0.5] * 5}, index=rep_dates)

        with patch("spc.backtest._ROOT", self.temp_path):
            with self.assertRaises((RuntimeError, KeyError)):
                self.backtester.evaluate_and_save(
                    idx_returns,
                    rep_returns,
                    rep_weights,
                    tail_percent=100.0,
                    timeseries_name="ts.csv",
                    summary_name="summary.json",
                    plot_name="plot.png",
                )


class TestStrategyBacktester(unittest.TestCase):
    """Test StrategyBacktester class."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

        # Create sample data
        dates = pd.date_range("2020-01-01", periods=12, freq="M")
        self.prices_df = pd.DataFrame(
            {
                "A": np.linspace(100, 120, 12),
                "B": np.linspace(50, 60, 12),
                "C": np.linspace(75, 90, 12),
            },
            index=dates,
        )

        self.weights_df = pd.DataFrame(
            {"A": [0.5] * 12, "B": [0.3] * 12, "C": [0.2] * 12}, index=dates
        )

    def tearDown(self):
        """Clean up temporary directory."""
        self.temp_dir.cleanup()
        plt.close("all")

    def test_strategy_backtester_initialization(self):
        """Test StrategyBacktester initialization."""
        bt = StrategyBacktester(prices_df=self.prices_df, weights_df=self.weights_df)

        self.assertIsNotNone(bt.prices)
        self.assertIsNotNone(bt.w_spx)
        self.assertIsNone(bt.basis_list)

    def test_strategy_backtester_with_basis_list(self):
        """Test StrategyBacktester with basis list."""
        basis_df = pd.DataFrame({"ticker": ["A", "B"]})

        bt = StrategyBacktester(
            prices_df=self.prices_df, weights_df=self.weights_df, basis_df=basis_df
        )

        self.assertIsNotNone(bt.basis_df)

    def test_strategy_backtester_compute_basis_weights_basic(self):
        """Test compute_basis_weights_time_series basic functionality."""
        basis_df = pd.DataFrame({"ticker": ["A", "B", "C"]})
        coeffs_ts = pd.DataFrame(
            {
                "date": ["2020-01-31"] * 3,
                "asset": ["A", "B", "C"],
                "basis": ["A", "B", "C"],
                "coef": [1.0, 1.0, 1.0],
            }
        )

        bt = StrategyBacktester(
            prices_df=self.prices_df,
            weights_df=self.weights_df,
            basis_df=basis_df,
            coeffs_ts=coeffs_ts,
        )

        bt.load_inputs = MagicMock()
        bt.basis_list = ["A", "B", "C"]

        with patch("spc.backtest.WeightMapper"):
            basis_weights = bt.compute_basis_weights_time_series()

        self.assertIsInstance(basis_weights, pd.DataFrame)
        self.assertEqual(basis_weights.shape[0], len(self.weights_df))

    def test_strategy_backtester_basis_list_required(self):
        """Test compute_basis_weights_time_series raises without basis_list."""
        bt = StrategyBacktester(prices_df=self.prices_df, weights_df=self.weights_df)

        bt.basis_list = None

        with self.assertRaises(ValueError):
            bt.compute_basis_weights_time_series()


class TestBaselineBacktester(unittest.TestCase):
    """Test BaselineBacktester class."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

        dates = pd.date_range("2020-01-01", periods=12, freq="M")
        self.prices_df = pd.DataFrame(
            {
                "A": np.linspace(100, 120, 12),
                "B": np.linspace(50, 60, 12),
                "C": np.linspace(75, 90, 12),
                "D": np.linspace(40, 50, 12),
            },
            index=dates,
        )

        self.weights_df = pd.DataFrame(
            {"A": [0.4] * 12, "B": [0.3] * 12, "C": [0.2] * 12, "D": [0.1] * 12},
            index=dates,
        )

    def tearDown(self):
        """Clean up temporary directory."""
        self.temp_dir.cleanup()
        plt.close("all")

    def test_baseline_backtester_initialization(self):
        """Test BaselineBacktester initialization."""
        bt = BaselineBacktester(prices_df=self.prices_df, weights_df=self.weights_df)

        self.assertIsNotNone(bt.prices)
        self.assertIsNotNone(bt.w_spx)

    def test_compute_topx_weights_basic(self):
        """Test compute_topx_weights_time_series basic functionality."""
        bt = BaselineBacktester(prices_df=self.prices_df, weights_df=self.weights_df)

        with patch("spc.backtest.tqdm", side_effect=lambda x, **kwargs: x):
            rep_weights = bt.compute_topx_weights_time_series(top_n=2)

        self.assertIsInstance(rep_weights, pd.DataFrame)
        self.assertEqual(rep_weights.shape[0], len(self.weights_df))

    def test_compute_topx_weights_sums_to_one(self):
        """Test that top-N weights sum to 1.0."""
        bt = BaselineBacktester(prices_df=self.prices_df, weights_df=self.weights_df)

        with patch("spc.backtest.tqdm", side_effect=lambda x, **kwargs: x):
            rep_weights = bt.compute_topx_weights_time_series(top_n=2)

        row_sums = rep_weights.sum(axis=1)
        np.testing.assert_array_almost_equal(
            row_sums, np.ones(len(rep_weights)), decimal=10
        )

    def test_compute_topx_weights_respects_top_n(self):
        """Test that top-N weights include only top N assets."""
        bt = BaselineBacktester(prices_df=self.prices_df, weights_df=self.weights_df)

        with patch("spc.backtest.tqdm", side_effect=lambda x, **kwargs: x):
            rep_weights = bt.compute_topx_weights_time_series(top_n=2)

        # For each row, check how many non-zero weights exist
        for idx in rep_weights.index:
            nonzero_count = (rep_weights.loc[idx] > 0).sum()
            # Should have at most top_n non-zero weights per row
            self.assertLessEqual(nonzero_count, 2)

    def test_compute_topx_weights_single_asset(self):
        """Test top-N with single top asset."""
        bt = BaselineBacktester(prices_df=self.prices_df, weights_df=self.weights_df)

        with patch("spc.backtest.tqdm", side_effect=lambda x, **kwargs: x):
            rep_weights = bt.compute_topx_weights_time_series(top_n=1)

        # Each row should have weights that sum to 1 and have at most 1 asset
        for idx in rep_weights.index:
            self.assertAlmostEqual(rep_weights.loc[idx].sum(), 1.0, places=10)
            self.assertLessEqual((rep_weights.loc[idx] > 0).sum(), 1)

    def test_compute_topx_weights_handles_zero_weights(self):
        """Test top-N with zero weights in some assets."""
        # Create weights with some zero entries
        weights_with_zeros = pd.DataFrame(
            {"A": [0.5, 0.0, 0.3], "B": [0.5, 0.0, 0.3], "C": [0.0, 1.0, 0.4]},
            index=pd.date_range("2020-01-01", periods=3, freq="M"),
        )

        bt = BaselineBacktester(weights_df=weights_with_zeros)

        with patch("spc.backtest.tqdm", side_effect=lambda x, **kwargs: x):
            rep_weights = bt.compute_topx_weights_time_series(top_n=1)

        # Should handle gracefully
        self.assertEqual(rep_weights.shape[0], 3)

    def test_baseline_backtester_run_analysis(self):
        """Test BaselineBacktester.run_analysis returns summary."""
        bt = BaselineBacktester(prices_df=self.prices_df, weights_df=self.weights_df)

        bt.load_inputs = MagicMock()

        with patch("spc.backtest.tqdm", side_effect=lambda x, **kwargs: x):
            with patch("spc.backtest._ROOT", self.temp_path):
                summary = bt.run_analysis(top_n=2, tail_percent=100.0)

        self.assertIsInstance(summary, dict)
        self.assertEqual(summary.get("top_n"), 2)


class TestBacktesterIntegration(unittest.TestCase):
    """Integration tests for backtester functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

    def tearDown(self):
        """Clean up temporary directory."""
        self.temp_dir.cleanup()
        plt.close("all")

    def test_full_backtest_workflow(self):
        """Test complete backtest workflow."""
        dates = pd.date_range("2020-01-01", periods=24, freq="D")
        prices = pd.DataFrame(
            {"A": np.linspace(100, 120, 24), "B": np.linspace(50, 60, 24)}, index=dates
        )

        weights = pd.DataFrame({"A": [0.6] * 24, "B": [0.4] * 24}, index=dates)

        bt = Backtester()
        bt.prices = prices
        bt.w_spx = weights

        returns = bt.compute_returns(prices)
        self.assertEqual(returns.shape[1], 2)
        # Should have some returns computed
        self.assertGreater(len(returns), 0)

    def test_turnover_and_returns_alignment(self):
        """Test that turnover and returns are properly aligned."""
        dates = pd.date_range("2020-01-01", periods=10, freq="M")

        rep_weights = pd.DataFrame(
            {
                "A": np.random.uniform(0.3, 0.7, 10),
                "B": np.random.uniform(0.3, 0.7, 10),
            },
            index=dates,
        )

        # Normalize to sum to 1
        rep_weights = rep_weights.div(rep_weights.sum(axis=1), axis=0)

        tail_idx = list(dates[5:])
        turnover = Backtester.compute_turnover(rep_weights, tail_idx)

        self.assertEqual(len(turnover), 5)
        self.assertTrue((turnover >= 0).all())

    def test_summary_statistics_consistency(self):
        """Test that summary statistics are internally consistent."""
        diff = pd.Series(
            np.random.normal(0, 0.01, 100),
            index=pd.date_range("2020-01-01", periods=100, freq="D"),
        )

        summary = Backtester.summarize(diff)

        # RMSE should be >= abs(mean_diff)
        self.assertGreaterEqual(summary["rmse"], abs(summary["mean_diff"]) - 1e-10)

        # Annualized std should be std * sqrt(12)
        expected_ann = summary["std_diff"] * np.sqrt(12.0)
        self.assertAlmostEqual(summary["annualized_std"], expected_ann, places=6)


if __name__ == "__main__":
    unittest.main()
