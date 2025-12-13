"""Test suite for spc/synthetic_data module.

Test cases for DynamicIndexSimulator and helper functions.
"""

import unittest
import numpy as np
import pandas as pd

from spc.synthetic_data import DynamicIndexSimulator


class TestDynamicIndexSimulatorInit(unittest.TestCase):
    """Test DynamicIndexSimulator initialization."""

    def test_init_defaults(self):
        """Test initialization with default parameters."""
        sim = DynamicIndexSimulator(
            n_sectors=3, assets_per_sector=5, n_months=12, random_seed=42
        )
        self.assertEqual(sim.n_sectors, 3)
        self.assertEqual(sim.assets_per_sector, 5)
        self.assertEqual(sim.n_months, 12)
        self.assertEqual(sim.index_size, 15)  # 3 * 5
        self.assertEqual(sim.removal_rate, 0.01)
        self.assertEqual(sim.addition_rate, 0.01)

    def test_init_custom_index_size(self):
        """Test initialization with custom index_size."""
        sim = DynamicIndexSimulator(
            n_sectors=3,
            assets_per_sector=5,
            n_months=12,
            index_size=10,
            random_seed=42,
        )
        self.assertEqual(sim.index_size, 10)

    def test_init_index_size_defaults_to_product(self):
        """Test that index_size defaults to n_sectors * assets_per_sector."""
        sim = DynamicIndexSimulator(
            n_sectors=4, assets_per_sector=8, n_months=12, random_seed=42
        )
        self.assertEqual(sim.index_size, 32)

    def test_init_sector_names_limited(self):
        """Test sector names with n_sectors < 5."""
        sim = DynamicIndexSimulator(
            n_sectors=2, assets_per_sector=3, n_months=12, random_seed=42
        )
        self.assertEqual(len(sim.sector_names), 2)
        self.assertIn("Technology", sim.sector_names)
        self.assertIn("Healthcare", sim.sector_names)

    def test_init_all_sector_names(self):
        """Test all five sector names available."""
        sim = DynamicIndexSimulator(
            n_sectors=5, assets_per_sector=3, n_months=12, random_seed=42
        )
        expected = ["Technology", "Healthcare", "Finance", "Energy", "Consumer"]
        self.assertEqual(sim.sector_names, expected)

    def test_init_initial_assets_count(self):
        """Test correct number of initial assets created."""
        n_sectors = 3
        assets_per_sector = 5
        sim = DynamicIndexSimulator(
            n_sectors=n_sectors,
            assets_per_sector=assets_per_sector,
            n_months=12,
            random_seed=42,
        )
        expected_count = n_sectors * assets_per_sector
        self.assertEqual(len(sim.initial_assets), expected_count)

    def test_init_initial_asset_sectors_mapping(self):
        """Test that all initial assets are mapped to sectors."""
        sim = DynamicIndexSimulator(
            n_sectors=2, assets_per_sector=3, n_months=12, random_seed=42
        )
        self.assertEqual(len(sim.initial_asset_sectors), 6)
        for ticker, sector in sim.initial_asset_sectors.items():
            self.assertIn(sector, sim.sector_names)

    def test_init_ticker_history_created(self):
        """Test that ticker history is properly initialized."""
        sim = DynamicIndexSimulator(
            n_sectors=2, assets_per_sector=2, n_months=12, random_seed=42
        )
        self.assertEqual(len(sim.ticker_history), 4)
        for ticker, info in sim.ticker_history.items():
            self.assertEqual(info["created_month"], 0)
            self.assertIsNone(info["removed_month"])
            self.assertIn(info["sector"], sim.sector_names)

    def test_init_current_assets_matches_initial(self):
        """Test that current_assets is initialized to initial_assets."""
        sim = DynamicIndexSimulator(
            n_sectors=2, assets_per_sector=3, n_months=12, random_seed=42
        )
        self.assertEqual(sim.current_assets, set(sim.initial_assets))

    def test_init_asset_sectors_copy(self):
        """Test that asset_sectors is a copy of initial_asset_sectors."""
        sim = DynamicIndexSimulator(
            n_sectors=2, assets_per_sector=3, n_months=12, random_seed=42
        )
        self.assertEqual(sim.asset_sectors, sim.initial_asset_sectors)
        # Verify it's a copy not a reference
        sim.asset_sectors["NewTicker"] = "Technology"
        self.assertNotIn("NewTicker", sim.initial_asset_sectors)

    def test_init_custom_sector_volatility(self):
        """Test custom sector volatility configuration."""
        custom_vol = {"Technology": 0.35, "Healthcare": 0.25}
        sim = DynamicIndexSimulator(
            n_sectors=2,
            assets_per_sector=3,
            n_months=12,
            sector_volatility=custom_vol,
            random_seed=42,
        )
        self.assertEqual(sim.sector_volatility["Technology"], 0.35)
        self.assertEqual(sim.sector_volatility["Healthcare"], 0.25)

    def test_init_default_sector_volatility(self):
        """Test default sector volatility values."""
        sim = DynamicIndexSimulator(
            n_sectors=3, assets_per_sector=3, n_months=12, random_seed=42
        )
        self.assertEqual(sim.sector_volatility["Technology"], 0.25)
        self.assertEqual(sim.sector_volatility["Healthcare"], 0.20)
        self.assertEqual(sim.sector_volatility["Finance"], 0.22)

    def test_init_custom_sector_correlation(self):
        """Test custom sector correlation configuration."""
        custom_corr = {"Technology": 0.75, "Healthcare": 0.65}
        sim = DynamicIndexSimulator(
            n_sectors=2,
            assets_per_sector=3,
            n_months=12,
            sector_correlation=custom_corr,
            random_seed=42,
        )
        self.assertEqual(sim.sector_correlation["Technology"], 0.75)
        self.assertEqual(sim.sector_correlation["Healthcare"], 0.65)

    def test_init_default_sector_correlation(self):
        """Test default sector correlation values."""
        sim = DynamicIndexSimulator(
            n_sectors=3, assets_per_sector=3, n_months=12, random_seed=42
        )
        self.assertEqual(sim.sector_correlation["Technology"], 0.60)
        self.assertEqual(sim.sector_correlation["Healthcare"], 0.50)
        self.assertEqual(sim.sector_correlation["Finance"], 0.70)

    def test_init_sector_drift_empty_by_default(self):
        """Test that sector_drift defaults to empty dict."""
        sim = DynamicIndexSimulator(
            n_sectors=2, assets_per_sector=3, n_months=12, random_seed=42
        )
        self.assertEqual(sim.sector_drift, {})

    def test_init_custom_sector_drift(self):
        """Test custom sector drift configuration."""
        custom_drift = {"low": -0.02, "high": 0.02}
        sim = DynamicIndexSimulator(
            n_sectors=2,
            assets_per_sector=3,
            n_months=12,
            sector_drift=custom_drift,
            random_seed=42,
        )
        self.assertEqual(sim.sector_drift["low"], -0.02)
        self.assertEqual(sim.sector_drift["high"], 0.02)

    def test_init_pareto_alpha(self):
        """Test Pareto alpha parameter for market cap distribution."""
        sim = DynamicIndexSimulator(
            n_sectors=2,
            assets_per_sector=3,
            n_months=12,
            cap_pareto_alpha=1.5,
            random_seed=42,
        )
        self.assertEqual(sim.cap_pareto_alpha, 1.5)

    def test_init_cap_scale(self):
        """Test cap scale parameter."""
        sim = DynamicIndexSimulator(
            n_sectors=2,
            assets_per_sector=3,
            n_months=12,
            cap_scale=100,
            random_seed=42,
        )
        self.assertEqual(sim.cap_scale, 100)

    def test_init_removal_rates(self):
        """Test removal and addition rate parameters."""
        sim = DynamicIndexSimulator(
            n_sectors=2,
            assets_per_sector=3,
            n_months=12,
            removal_rate=0.05,
            addition_rate=0.03,
            random_seed=42,
        )
        self.assertEqual(sim.removal_rate, 0.05)
        self.assertEqual(sim.addition_rate, 0.03)

    def test_init_reproducibility_same_seed(self):
        """Test that same seed produces same initialization."""
        sim1 = DynamicIndexSimulator(
            n_sectors=3, assets_per_sector=5, n_months=12, random_seed=999
        )
        sim2 = DynamicIndexSimulator(
            n_sectors=3, assets_per_sector=5, n_months=12, random_seed=999
        )
        self.assertEqual(sim1.initial_assets, sim2.initial_assets)
        self.assertEqual(sim1.initial_asset_sectors, sim2.initial_asset_sectors)

    def test_init_random_seed_storage(self):
        """Test that random seed parameter is stored."""
        sim = DynamicIndexSimulator(
            n_sectors=3, assets_per_sector=5, n_months=12, random_seed=42
        )
        self.assertEqual(sim.random_seed, 42)
        sim2 = DynamicIndexSimulator(
            n_sectors=3, assets_per_sector=5, n_months=12, random_seed=123
        )
        self.assertEqual(sim2.random_seed, 123)

    def test_init_prices_initialized_empty(self):
        """Test that prices dict is initialized empty."""
        sim = DynamicIndexSimulator(
            n_sectors=2, assets_per_sector=3, n_months=12, random_seed=42
        )
        self.assertEqual(sim.prices, {})

    def test_init_removal_bottom_pct(self):
        """Test removal_bottom_pct parameter."""
        sim = DynamicIndexSimulator(
            n_sectors=2,
            assets_per_sector=3,
            n_months=12,
            removal_bottom_pct=0.15,
            random_seed=42,
        )
        self.assertEqual(sim.removal_bottom_pct, 0.15)

    def test_init_n_months_parameter(self):
        """Test n_months parameter is stored correctly."""
        sim = DynamicIndexSimulator(
            n_sectors=2, assets_per_sector=3, n_months=60, random_seed=42
        )
        self.assertEqual(sim.n_months, 60)

    def test_init_random_seed_parameter(self):
        """Test random_seed parameter is stored correctly."""
        sim = DynamicIndexSimulator(
            n_sectors=2, assets_per_sector=3, n_months=12, random_seed=777
        )
        self.assertEqual(sim.random_seed, 777)

    """Test individual methods without full initialization."""

    def setUp(self):
        """Set up test fixtures by creating a partially initialized simulator."""
        # Create simulator
        self.sim = DynamicIndexSimulator.__new__(DynamicIndexSimulator)
        self.sim.n_sectors = 3
        self.sim.assets_per_sector = 5
        self.sim.n_months = 12
        self.sim.removal_rate = 0.01
        self.sim.addition_rate = 0.01
        self.sim.random_seed = 42
        self.sim.index_size = 15
        self.sim.removal_bottom_pct = 0.2
        self.sim.cap_pareto_alpha = 1.2
        self.sim.cap_scale = 50
        self.sim.total_assets_created = 0
        self.sim.ticker_history = {}
        self.sim.sector_names = ["Technology", "Healthcare", "Finance"]
        self.sim.sector_volatility = {
            "Technology": 0.25,
            "Healthcare": 0.20,
            "Finance": 0.22,
        }
        self.sim.sector_correlation = {
            "Technology": 0.60,
            "Healthcare": 0.50,
            "Finance": 0.70,
        }
        self.sim.asset_sectors = {}
        self.sim.prices = {}
        self.sim.market_caps = {}
        self.sim.current_assets = set()
        np.random.seed(42)

    def test_sample_shares_outstanding(self):
        """Test shares outstanding sampling."""
        for _ in range(100):
            shares = self.sim._sample_shares_outstanding()
            self.assertGreaterEqual(shares, 1.0)
            self.assertIsInstance(shares, (float, np.floating))

    def test_sample_shares_outstanding_distribution(self):
        """Test that shares outstanding shows heavy-tail behavior."""
        shares_list = [self.sim._sample_shares_outstanding() for _ in range(1000)]
        median_shares = np.median(shares_list)
        max_shares = np.max(shares_list)
        # Heavy tailed should have max much larger than median
        self.assertGreater(max_shares, 10 * median_shares)

    def test_generate_ticker(self):
        """Test ticker generation."""
        ticker1 = self.sim._generate_ticker("Technology")
        self.assertTrue(ticker1.startswith("TEC"))
        self.assertEqual(self.sim.total_assets_created, 1)

        ticker2 = self.sim._generate_ticker("Healthcare")
        self.assertTrue(ticker2.startswith("HEA"))
        self.assertEqual(self.sim.total_assets_created, 2)
        self.assertNotEqual(ticker1, ticker2)

    def test_generate_ticker_uniqueness(self):
        """Test ticker uniqueness."""
        tickers = set()
        for i in range(50):
            ticker = self.sim._generate_ticker("Technology")
            self.assertNotIn(ticker, tickers)
            tickers.add(ticker)
        self.assertEqual(len(tickers), 50)

    def test_get_sector_correlation_matrix_shape(self):
        """Test correlation matrix has correct shape."""
        n_assets = 5
        corr_matrix = self.sim._get_sector_correlation_matrix("Technology", n_assets)
        self.assertEqual(corr_matrix.shape, (n_assets, n_assets))

    def test_get_sector_correlation_matrix_diagonal(self):
        """Test correlation matrix diagonal is 1.0."""
        n_assets = 5
        corr_matrix = self.sim._get_sector_correlation_matrix("Technology", n_assets)
        diag = np.diag(corr_matrix)
        np.testing.assert_array_almost_equal(diag, np.ones(n_assets))

    def test_get_sector_correlation_matrix_symmetric(self):
        """Test correlation matrix is symmetric."""
        n_assets = 5
        corr_matrix = self.sim._get_sector_correlation_matrix("Technology", n_assets)
        np.testing.assert_array_almost_equal(corr_matrix, corr_matrix.T)

    def test_get_sector_correlation_matrix_positive_definite(self):
        """Test correlation matrix is positive semi-definite."""
        n_assets = 10
        corr_matrix = self.sim._get_sector_correlation_matrix("Technology", n_assets)
        eigvals = np.linalg.eigvals(corr_matrix)
        # All eigenvalues should be non-negative
        self.assertTrue(np.all(eigvals >= -1e-10))

    def test_generate_monthly_returns_coverage(self):
        """Test monthly returns generation."""
        # Set up ticker tracking
        self.sim.initial_assets = ["TEC0", "HEA1", "FIN2"]
        self.sim.asset_sectors = {
            "TEC0": "Technology",
            "HEA1": "Healthcare",
            "FIN2": "Finance",
        }

        tickers = set(self.sim.initial_assets)
        returns = self.sim._generate_monthly_returns(tickers)

        self.assertEqual(len(returns), len(tickers))
        for ticker in tickers:
            self.assertIn(ticker, returns)
            self.assertIsInstance(returns[ticker], (float, np.floating))

    def test_generate_monthly_returns_finite(self):
        """Test monthly returns are finite."""
        self.sim.initial_assets = ["TEC0", "HEA1", "FIN2"]
        self.sim.asset_sectors = {
            "TEC0": "Technology",
            "HEA1": "Healthcare",
            "FIN2": "Finance",
        }

        tickers = set(self.sim.initial_assets)
        returns = self.sim._generate_monthly_returns(tickers)

        for ticker, ret in returns.items():
            self.assertTrue(np.isfinite(ret))
            self.assertGreater(ret, -1.0)
            self.assertLess(ret, 1.0)

    def test_generate_monthly_returns_empty_set(self):
        """Test monthly returns with empty set."""
        returns = self.sim._generate_monthly_returns(set())
        self.assertEqual(len(returns), 0)

    def test_generate_monthly_returns_single_ticker(self):
        """Test monthly returns for single ticker."""
        self.sim.asset_sectors = {"TEC0": "Technology"}
        returns = self.sim._generate_monthly_returns({"TEC0"})

        self.assertEqual(len(returns), 1)
        self.assertIn("TEC0", returns)


class TestMarketWeightsFunctions(unittest.TestCase):
    """Test market weight generation and manipulation."""

    def setUp(self):
        """Create simulator instance for weight testing."""
        self.sim = DynamicIndexSimulator.__new__(DynamicIndexSimulator)
        self.sim.n_sectors = 2
        self.sim.assets_per_sector = 3
        self.sim.removal_bottom_pct = 0.2

    def test_apply_max_stock_weight_returns_dataframe(self):
        """Test that apply_max_stock_weight returns DataFrame."""
        weights_df = pd.DataFrame(
            {"A": [1.0 / 3, 1.0 / 3], "B": [1.0 / 3, 1.0 / 3], "C": [1.0 / 3, 1.0 / 3]}
        )
        result = self.sim.apply_max_stock_weight(weights_df, max_weight=None)
        self.assertIsInstance(result, pd.DataFrame)

    def test_apply_max_stock_weight_no_cap_returns_unchanged(self):
        """Test that None cap returns unchanged DataFrame."""
        weights_df = pd.DataFrame({"A": [0.5, 0.3], "B": [0.3, 0.5], "C": [0.2, 0.2]})
        result = self.sim.apply_max_stock_weight(weights_df, max_weight=None)
        pd.testing.assert_frame_equal(result, weights_df)

    def test_apply_max_stock_weight_enforces_cap(self):
        """Test that weights don't exceed cap."""
        weights_df = pd.DataFrame({"A": [0.5, 0.5], "B": [0.3, 0.3], "C": [0.2, 0.2]})
        max_weight = 0.4  # Feasible cap for 3 tickers (0.4 * 3 = 1.2 >= 1.0)
        result = self.sim.apply_max_stock_weight(weights_df, max_weight=max_weight)

        # Check no weight exceeds cap
        valid_weights = result.dropna().values.flatten()
        self.assertTrue(np.all(valid_weights <= max_weight + 1e-10))

    def test_apply_max_stock_weight_sums_to_one(self):
        """Test that capped weights still sum to 1.0."""
        weights_df = pd.DataFrame({"A": [0.5, 0.6], "B": [0.3, 0.2], "C": [0.2, 0.2]})
        max_weight = 0.35
        result = self.sim.apply_max_stock_weight(weights_df, max_weight=max_weight)

        for idx in result.index:
            row = result.loc[idx].dropna()
            if len(row) > 0:
                row_sum = row.sum()
                self.assertAlmostEqual(row_sum, 1.0, places=10)

    def test_apply_max_stock_weight_infeasible_cap_raises(self):
        """Test that infeasible cap raises ValueError."""
        weights_df = pd.DataFrame(
            {"A": [1.0 / 3, 1.0 / 3], "B": [1.0 / 3, 1.0 / 3], "C": [1.0 / 3, 1.0 / 3]}
        )
        # Cap too small for 3 tickers
        max_weight = 0.001

        with self.assertRaises(ValueError):
            self.sim.apply_max_stock_weight(weights_df, max_weight=max_weight)

    def test_apply_max_stock_weight_one_third_cap(self):
        """Test capping at 1/3 for three active tickers."""
        weights_df = pd.DataFrame(
            {"A": [1.0 / 3, 0.5], "B": [1.0 / 3, 0.3], "C": [1.0 / 3, 0.2]}
        )
        result = self.sim.apply_max_stock_weight(weights_df, max_weight=1.0 / 3)

        for col in result.columns:
            self.assertTrue((result[col] <= 1.0 / 3 + 1e-10).all())

    def test_apply_max_stock_weight_preserves_nan(self):
        """Test that NaN values are preserved."""
        weights_df = pd.DataFrame(
            {"A": [1.0 / 3, np.nan], "B": [1.0 / 3, 0.5], "C": [1.0 / 3, 0.5]}
        )
        result = self.sim.apply_max_stock_weight(weights_df, max_weight=0.5)

        # Check NaN is preserved
        self.assertTrue(pd.isna(result.loc[1, "A"]))


class TestMarketWeightsGeneration(unittest.TestCase):
    """Test generate_market_weights method."""

    def setUp(self):
        """Create simulator and price data."""
        self.sim = DynamicIndexSimulator.__new__(DynamicIndexSimulator)
        self.sim.ticker_history = {
            "A": {"created_month": 0, "removed_month": None},
            "B": {"created_month": 0, "removed_month": None},
            "C": {"created_month": 0, "removed_month": None},
        }
        self.sim.market_caps = {
            "A": [100.0, 110.0, 120.0],
            "B": [50.0, 55.0, 60.0],
            "C": [25.0, 28.0, 30.0],
        }
        self.sim.shares_outstanding = {"A": 10.0, "B": 5.0, "C": 2.5}

    def test_generate_market_weights_returns_two_dataframes(self):
        """Test output format."""
        prices_df = pd.DataFrame(
            {
                "A": [100.0, 110.0, 120.0],
                "B": [50.0, 55.0, 60.0],
                "C": [25.0, 28.0, 30.0],
            },
            index=pd.date_range("2020-01-01", periods=3, freq="MS"),
        )
        weights_df, market_cap_df = self.sim.generate_market_weights(prices_df)

        self.assertIsInstance(weights_df, pd.DataFrame)
        self.assertIsInstance(market_cap_df, pd.DataFrame)

    def test_generate_market_weights_shapes(self):
        """Test output shapes match input."""
        prices_df = pd.DataFrame(
            {
                "A": [100.0, 110.0, 120.0],
                "B": [50.0, 55.0, 60.0],
                "C": [25.0, 28.0, 30.0],
            },
            index=pd.date_range("2020-01-01", periods=3, freq="MS"),
        )
        weights_df, market_cap_df = self.sim.generate_market_weights(prices_df)

        self.assertEqual(weights_df.shape, market_cap_df.shape)
        self.assertEqual(weights_df.shape[0], prices_df.shape[0])

    def test_generate_market_weights_sums_to_one(self):
        """Test that weights sum to 1.0."""
        prices_df = pd.DataFrame(
            {"A": [100.0, 110.0], "B": [50.0, 55.0], "C": [25.0, 28.0]},
            index=pd.date_range("2020-01-01", periods=2, freq="MS"),
        )
        weights_df, _ = self.sim.generate_market_weights(prices_df)

        for idx in weights_df.index:
            row = weights_df.loc[idx].dropna()
            if len(row) > 0:
                row_sum = row.sum()
                self.assertAlmostEqual(row_sum, 1.0, places=10)

    def test_generate_market_weights_non_negative(self):
        """Test that weights are non-negative."""
        prices_df = pd.DataFrame(
            {"A": [100.0, 110.0], "B": [50.0, 55.0], "C": [25.0, 28.0]},
            index=pd.date_range("2020-01-01", periods=2, freq="MS"),
        )
        weights_df, _ = self.sim.generate_market_weights(prices_df)

        valid_weights = weights_df.dropna().values.flatten()
        self.assertTrue(np.all(valid_weights >= 0.0))

    def test_generate_market_weights_market_caps_positive(self):
        """Test that market caps are non-negative."""
        prices_df = pd.DataFrame(
            {"A": [100.0, 110.0], "B": [50.0, 55.0], "C": [25.0, 28.0]},
            index=pd.date_range("2020-01-01", periods=2, freq="MS"),
        )
        _, market_cap_df = self.sim.generate_market_weights(prices_df)

        valid_caps = market_cap_df.dropna().values.flatten()
        self.assertTrue(np.all(valid_caps >= 0.0))


class TestEdgeCases(unittest.TestCase):
    """Test edge cases."""

    def setUp(self):
        """Set up simulator."""
        self.sim = DynamicIndexSimulator.__new__(DynamicIndexSimulator)
        self.sim.ticker_history = {}
        self.sim.removal_bottom_pct = 0.2

    def test_generate_market_weights_single_row(self):
        """Test with single row DataFrame."""
        prices_df = pd.DataFrame(
            {"A": [100.0], "B": [50.0], "C": [25.0]},
            index=pd.date_range("2020-01-01", periods=1, freq="MS"),
        )
        self.sim.ticker_history = {
            "A": {"created_month": 0, "removed_month": None},
            "B": {"created_month": 0, "removed_month": None},
            "C": {"created_month": 0, "removed_month": None},
        }
        self.sim.market_caps = {"A": [100.0], "B": [50.0], "C": [25.0]}

        weights_df, market_cap_df = self.sim.generate_market_weights(prices_df)
        self.assertEqual(weights_df.shape[0], 1)

    def test_generate_market_weights_many_columns(self):
        """Test with many tickers."""
        n_tickers = 50
        data = {f"T{i}": np.linspace(100, 120, 10) for i in range(n_tickers)}
        prices_df = pd.DataFrame(
            data, index=pd.date_range("2020-01-01", periods=10, freq="MS")
        )

        self.sim.ticker_history = {
            f"T{i}": {"created_month": 0, "removed_month": None}
            for i in range(n_tickers)
        }
        self.sim.market_caps = {f"T{i}": [100.0 - i] * 10 for i in range(n_tickers)}

        weights_df, market_cap_df = self.sim.generate_market_weights(prices_df)
        self.assertEqual(weights_df.shape[1], n_tickers)


if __name__ == "__main__":
    unittest.main()
