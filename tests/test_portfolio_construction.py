"""Test suite for spc/portfolio_construction module.

Test cases for BasisSelector, LocalRidgeRunner, and WeightMapper classes.
"""

import unittest
import numpy as np
import pandas as pd
from pathlib import Path
from tempfile import TemporaryDirectory

from spc.portfolio_construction import (
    BasisSelector,
    LocalRidgeRunner,
    WeightMapper,
)


class TestBasisSelector(unittest.TestCase):
    """Test suite for BasisSelector class."""

    def setUp(self):
        """Set up test fixtures."""
        # Simple chain tree: 0 -- 1 -- 2 -- 3 -- 4
        self.chain_adj = {
            0: [(1, 1.0)],
            1: [(0, 1.0), (2, 1.0)],
            2: [(1, 1.0), (3, 1.0)],
            3: [(2, 1.0), (4, 1.0)],
            4: [(3, 1.0)],
        }

        # Star tree: 0 at center connected to 1, 2, 3
        self.star_adj = {
            0: [(1, 1.0), (2, 1.0), (3, 1.0)],
            1: [(0, 1.0)],
            2: [(0, 1.0)],
            3: [(0, 1.0)],
        }

    def test_basis_selector_init(self):
        """Test BasisSelector initialization."""
        selector = BasisSelector(self.chain_adj)
        self.assertEqual(selector.n, 5)
        self.assertEqual(len(selector.nodes), 5)
        self.assertIn(0, selector.nodes)

    def test_basis_selector_with_custom_nodes(self):
        """Test BasisSelector with custom node ordering."""
        custom_nodes = [4, 3, 2, 1, 0]
        selector = BasisSelector(self.chain_adj, nodes=custom_nodes)
        self.assertEqual(selector.nodes, custom_nodes)

    def test_normalize_dict_empty(self):
        """Test normalization of empty dict."""
        selector = BasisSelector(self.chain_adj)
        result = selector._normalize_dict({}, fallback=1.0)
        for node in selector.nodes:
            self.assertEqual(result[node], 1.0)

    def test_normalize_dict_single_value(self):
        """Test normalization with single value."""
        selector = BasisSelector(self.chain_adj)
        vals = {0: 5.0, 1: 5.0, 2: 5.0, 3: 5.0, 4: 5.0}
        result = selector._normalize_dict(vals)
        # All equal values should normalize to 0.0
        for node in selector.nodes:
            self.assertEqual(result[node], 0.0)

    def test_normalize_dict_range(self):
        """Test normalization produces values in [0, 1]."""
        selector = BasisSelector(self.chain_adj)
        vals = {0: 0.0, 1: 10.0, 2: 20.0, 3: 30.0, 4: 40.0}
        result = selector._normalize_dict(vals)
        for node in selector.nodes:
            self.assertGreaterEqual(result[node], 0.0)
            self.assertLessEqual(result[node], 1.0)

    def test_avg_dist(self):
        """Test average distance computation."""
        selector = BasisSelector(self.chain_adj)
        avg = selector._avg_dist()
        self.assertEqual(len(avg), 5)
        # Center node (2) should have smaller avg distance than edge nodes
        self.assertLess(avg[2], avg[0])

    def test_dist_uv(self):
        """Test pairwise distance lookup."""
        selector = BasisSelector(self.chain_adj)
        # In chain, 0 to 4 is 4 hops
        self.assertEqual(selector._dist_uv(0, 4), 4.0)
        # Distance from node to itself
        self.assertEqual(selector._dist_uv(0, 0), 0.0)

    def test_select_max_spread_k_zero(self):
        """Test select_max_spread with k=0."""
        selector = BasisSelector(self.chain_adj)
        basis = selector.select_max_spread(k=0, weights={})
        self.assertEqual(basis, [])

    def test_select_max_spread_k_ge_n(self):
        """Test select_max_spread with k >= n."""
        selector = BasisSelector(self.chain_adj)
        basis = selector.select_max_spread(k=10, weights={})
        self.assertEqual(len(basis), 5)

    def test_select_max_spread_basic(self):
        """Test basic max-spread selection."""
        selector = BasisSelector(self.chain_adj)
        basis = selector.select_max_spread(k=2, weights={})
        self.assertEqual(len(basis), 2)
        self.assertIn(basis[0], selector.nodes)
        self.assertIn(basis[1], selector.nodes)

    def test_select_max_spread_with_weights(self):
        """Test max-spread with node weights."""
        selector = BasisSelector(self.chain_adj)
        weights = {0: 10.0, 1: 5.0, 2: 1.0, 3: 5.0, 4: 10.0}
        basis = selector.select_max_spread(k=3, weights=weights, alpha=0.5)
        self.assertEqual(len(basis), 3)
        # With weights favoring edges and spread, endpoints should be favored
        self.assertIn(0, basis)
        self.assertIn(4, basis)

    def test_select_max_spread_alpha_pure_spread(self):
        """Test max-spread with alpha=1.0 (pure spread)."""
        selector = BasisSelector(self.chain_adj)
        basis = selector.select_max_spread(k=3, weights={}, alpha=1.0)
        self.assertEqual(len(basis), 3)

    def test_select_max_spread_alpha_pure_weight(self):
        """Test max-spread with alpha=0.0 (pure weight)."""
        selector = BasisSelector(self.chain_adj)
        weights = {0: 0.0, 1: 0.0, 2: 100.0, 3: 0.0, 4: 0.0}
        basis = selector.select_max_spread(k=1, weights=weights, alpha=0.0)
        self.assertEqual(basis, [2])

    def test_select_max_spread_with_prev_basis(self):
        """Test max-spread with previous basis and stickiness."""
        selector = BasisSelector(self.chain_adj)
        prev_basis = [0]
        basis = selector.select_max_spread(
            k=3, weights={}, prev_basis=prev_basis, stickiness=0.5
        )
        self.assertLessEqual(len(basis), 5)
        # First element should be from prev_basis (receives bonus)
        self.assertEqual(basis[0], 0)

    def test_select_max_spread_stickiness_without_prev(self):
        """Test that stickiness without prev_basis is handled."""
        selector = BasisSelector(self.chain_adj)
        basis = selector.select_max_spread(k=2, weights={}, stickiness=0.5)
        self.assertEqual(len(basis), 2)

    def test_select_hub_branch_k_zero(self):
        """Test select_hub_branch with k=0."""
        selector = BasisSelector(self.chain_adj)
        basis = selector.select_hub_branch(k=0, weights={})
        self.assertEqual(basis, [])

    def test_select_hub_branch_k_ge_n(self):
        """Test select_hub_branch with k >= n."""
        selector = BasisSelector(self.chain_adj)
        basis = selector.select_hub_branch(k=10, weights={})
        self.assertEqual(len(basis), 5)

    def test_select_hub_branch_basic(self):
        """Test basic hub-branch selection."""
        selector = BasisSelector(self.star_adj)
        basis = selector.select_hub_branch(k=2, h=1, weights={}, prev_basis=[])
        self.assertEqual(len(basis), 2)
        # Center should be selected as hub
        self.assertIn(0, basis)

    def test_select_hub_branch_auto_h(self):
        """Test hub-branch with auto h calculation."""
        selector = BasisSelector(self.chain_adj)
        basis = selector.select_hub_branch(k=5, weights={})
        self.assertEqual(len(basis), 5)

    def test_select_hub_branch_with_weights(self):
        """Test hub-branch with external weights."""
        selector = BasisSelector(self.star_adj)
        weights = {0: 1.0, 1: 10.0, 2: 10.0, 3: 10.0}
        basis = selector.select_hub_branch(
            k=3, h=1, weights=weights, weight_gamma=0.5, prev_basis=[]
        )
        self.assertLessEqual(len(basis), 4)

    def test_select_hub_branch_alpha_variations(self):
        """Test hub-branch with different alpha values."""
        selector = BasisSelector(self.chain_adj)
        basis_alpha0 = selector.select_hub_branch(
            k=2, alpha=0.0, weights={}, prev_basis=[]
        )
        basis_alpha1 = selector.select_hub_branch(
            k=2, alpha=1.0, weights={}, prev_basis=[]
        )
        self.assertEqual(len(basis_alpha0), 2)
        self.assertEqual(len(basis_alpha1), 2)

    def test_greedy_farthest_fill(self):
        """Test greedy farthest fill helper."""
        selector = BasisSelector(self.chain_adj)
        B = [0]  # Start with node 0
        selector._greedy_farthest_fill(B, k=3)
        self.assertEqual(len(B), 3)
        # Should add farthest nodes
        self.assertIn(4, B)  # Farthest from 0 is 4

    def test_greedy_farthest_fill_empty_basis(self):
        """Test greedy fill with empty initial basis."""
        selector = BasisSelector(self.chain_adj)
        B = []
        selector._greedy_farthest_fill(B, k=2)
        # Should handle empty basis gracefully
        self.assertEqual(len(B), 0)

    def test_greedy_farthest_fill_already_full(self):
        """Test greedy fill when basis already has k nodes."""
        selector = BasisSelector(self.chain_adj)
        B = [0, 1, 2, 3, 4]
        initial_len = len(B)
        selector._greedy_farthest_fill(B, k=3)
        # Should not remove nodes
        self.assertEqual(len(B), initial_len)


class TestLocalRidgeRunner(unittest.TestCase):
    """Test suite for LocalRidgeRunner class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create sample price data
        dates = pd.date_range("2020-01-01", periods=12, freq="M")
        self.prices_df = pd.DataFrame(
            {
                "AAPL": [
                    100.0,
                    101.0,
                    102.0,
                    103.0,
                    104.0,
                    105.0,
                    106.0,
                    107.0,
                    108.0,
                    109.0,
                    110.0,
                    111.0,
                ],
                "MSFT": [
                    200.0,
                    202.0,
                    204.0,
                    206.0,
                    208.0,
                    210.0,
                    212.0,
                    214.0,
                    216.0,
                    218.0,
                    220.0,
                    222.0,
                ],
                "GOOG": [
                    150.0,
                    151.5,
                    153.0,
                    154.5,
                    156.0,
                    157.5,
                    159.0,
                    160.5,
                    162.0,
                    163.5,
                    165.0,
                    166.5,
                ],
            },
            index=dates,
        )
        self.basis_list = ["AAPL", "MSFT"]

    def test_local_ridge_runner_init(self):
        """Test LocalRidgeRunner initialization."""
        with TemporaryDirectory() as tmpdir:
            prices_path = Path(tmpdir) / "prices.csv"
            basis_path = Path(tmpdir) / "basis.csv"
            prices_path.write_text("")
            basis_path.write_text("")

            runner = LocalRidgeRunner(
                prices_path=prices_path,
                basis_path=basis_path,
                ridge_alpha=2.0,
                q_neighbors=5,
            )
            self.assertEqual(runner.ridge_alpha, 2.0)
            self.assertEqual(runner.q_neighbors, 5)
            self.assertEqual(runner.corr_method, "pearson")

    def test_compute_returns(self):
        """Test return computation."""
        with TemporaryDirectory() as tmpdir:
            runner = LocalRidgeRunner(
                prices_path=Path(tmpdir) / "prices.csv",
                basis_path=Path(tmpdir) / "basis.csv",
            )
            returns = runner._compute_returns(self.prices_df)
            self.assertEqual(returns.shape[1], 3)  # Same assets
            self.assertEqual(returns.shape[0], 11)  # One less row due to pct_change
            # First row should be NaN and dropped
            self.assertTrue(returns.index[0] > pd.Timestamp("2020-01-01"))

    def test_compute_distance_df(self):
        """Test distance matrix computation."""
        with TemporaryDirectory() as tmpdir:
            runner = LocalRidgeRunner(
                prices_path=Path(tmpdir) / "prices.csv",
                basis_path=Path(tmpdir) / "basis.csv",
            )
            distances = runner._compute_distance_df(self.prices_df)
            self.assertEqual(distances.shape, (3, 3))
            # Diagonal should be zero
            np.testing.assert_almost_equal(distances.loc["AAPL", "AAPL"], 0.0)

    def test_filter_basis(self):
        """Test basis filtering."""
        with TemporaryDirectory() as tmpdir:
            runner = LocalRidgeRunner(
                prices_path=Path(tmpdir) / "prices.csv",
                basis_path=Path(tmpdir) / "basis.csv",
            )
            returns = runner._compute_returns(self.prices_df)
            filtered = runner._filter_basis(self.basis_list, returns)
            self.assertEqual(filtered, self.basis_list)

    def test_filter_basis_missing_asset(self):
        """Test basis filtering with missing assets."""
        with TemporaryDirectory() as tmpdir:
            runner = LocalRidgeRunner(
                prices_path=Path(tmpdir) / "prices.csv",
                basis_path=Path(tmpdir) / "basis.csv",
            )
            returns = runner._compute_returns(self.prices_df)
            basis_with_missing = ["AAPL", "MSFT", "NONEXISTENT"]
            filtered = runner._filter_basis(basis_with_missing, returns)
            self.assertEqual(filtered, ["AAPL", "MSFT"])

    def test_filter_basis_no_match_raises(self):
        """Test that filtering with no matches raises ValueError."""
        with TemporaryDirectory() as tmpdir:
            runner = LocalRidgeRunner(
                prices_path=Path(tmpdir) / "prices.csv",
                basis_path=Path(tmpdir) / "basis.csv",
            )
            returns = runner._compute_returns(self.prices_df)
            basis_invalid = ["NONEXISTENT1", "NONEXISTENT2"]
            with self.assertRaises(ValueError):
                runner._filter_basis(basis_invalid, returns)

    def test_select_neighbors_for_asset(self):
        """Test neighbor selection for an asset."""
        with TemporaryDirectory() as tmpdir:
            runner = LocalRidgeRunner(
                prices_path=Path(tmpdir) / "prices.csv",
                basis_path=Path(tmpdir) / "basis.csv",
                q_neighbors=1,
            )
            distances = runner._compute_distance_df(self.prices_df)
            neighbors = runner._select_neighbors_for_asset(
                "GOOG", self.basis_list, distances, q_neighbors=1
            )
            self.assertEqual(len(neighbors), 1)
            self.assertIn(neighbors[0], self.basis_list)

    def test_select_neighbors_no_q_limit(self):
        """Test neighbor selection without q_neighbors limit."""
        with TemporaryDirectory() as tmpdir:
            runner = LocalRidgeRunner(
                prices_path=Path(tmpdir) / "prices.csv",
                basis_path=Path(tmpdir) / "basis.csv",
            )
            distances = runner._compute_distance_df(self.prices_df)
            neighbors = runner._select_neighbors_for_asset(
                "GOOG", self.basis_list, distances, q_neighbors=None
            )
            self.assertEqual(len(neighbors), 2)

    def test_select_neighbors_missing_asset(self):
        """Test neighbor selection for missing asset."""
        with TemporaryDirectory() as tmpdir:
            runner = LocalRidgeRunner(
                prices_path=Path(tmpdir) / "prices.csv",
                basis_path=Path(tmpdir) / "basis.csv",
            )
            distances = runner._compute_distance_df(self.prices_df)
            neighbors = runner._select_neighbors_for_asset(
                "NONEXISTENT", self.basis_list, distances, q_neighbors=None
            )
            self.assertEqual(neighbors, [])

    def test_select_neighbors_sorted_by_distance(self):
        """Test that neighbors are sorted by distance."""
        with TemporaryDirectory() as tmpdir:
            runner = LocalRidgeRunner(
                prices_path=Path(tmpdir) / "prices.csv",
                basis_path=Path(tmpdir) / "basis.csv",
            )
            distances = runner._compute_distance_df(self.prices_df)
            neighbors = runner._select_neighbors_for_asset(
                "GOOG", self.basis_list, distances, q_neighbors=None
            )
            # Should be ordered by increasing distance
            if len(neighbors) == 2:
                d1 = distances.loc["GOOG", neighbors[0]]
                d2 = distances.loc["GOOG", neighbors[1]]
                self.assertLessEqual(d1, d2)

    def test_run_all_regressions_basic(self):
        """Test running all regressions."""
        with TemporaryDirectory() as tmpdir:
            runner = LocalRidgeRunner(
                prices_path=Path(tmpdir) / "prices.csv",
                basis_path=Path(tmpdir) / "basis.csv",
                ridge_alpha=1.0,
            )
            returns = runner._compute_returns(self.prices_df)
            coef_df, recon_df, errors_df = runner.run_all_regressions(
                returns=returns,
                basis_list=self.basis_list,
                prices=self.prices_df,
                ridge_alpha=1.0,
                q_neighbors=None,
            )
            # Should have non-basis asset (GOOG) in coefficients
            self.assertGreater(len(coef_df), 0)
            # Reconstructed returns should match shape
            self.assertEqual(recon_df.shape[1], 3)
            # Errors should have entries
            self.assertGreater(len(errors_df), 0)

    def test_run_all_regressions_with_q_neighbors(self):
        """Test regressions with neighbor limit."""
        with TemporaryDirectory() as tmpdir:
            runner = LocalRidgeRunner(
                prices_path=Path(tmpdir) / "prices.csv",
                basis_path=Path(tmpdir) / "basis.csv",
                ridge_alpha=1.0,
                q_neighbors=1,
            )
            returns = runner._compute_returns(self.prices_df)
            coef_df, recon_df, errors_df = runner.run_all_regressions(
                returns=returns,
                basis_list=self.basis_list,
                prices=self.prices_df,
                ridge_alpha=1.0,
                q_neighbors=1,
            )
            self.assertGreater(len(coef_df), 0)

    def test_save_outputs(self):
        """Test saving outputs to CSV files."""
        with TemporaryDirectory() as tmpdir:
            runner = LocalRidgeRunner(
                prices_path=Path(tmpdir) / "prices.csv",
                basis_path=Path(tmpdir) / "basis.csv",
                out_dir=tmpdir,
            )
            coef_df = pd.DataFrame(
                {"asset": ["GOOG"], "basis": ["AAPL"], "coef": [0.5]}
            )
            recon_df = pd.DataFrame({"AAPL": [0.01], "MSFT": [0.02]})
            errors_df = pd.DataFrame({"rmse": [0.01], "n_obs": [10]}, index=["GOOG"])

            runner._save_outputs(coef_df, recon_df, errors_df, Path(tmpdir))

            # Check files exist
            self.assertTrue((Path(tmpdir) / "coefficients_ridge.csv").exists())
            self.assertTrue((Path(tmpdir) / "recon_returns.csv").exists())
            self.assertTrue((Path(tmpdir) / "regression_errors.csv").exists())


class TestWeightMapper(unittest.TestCase):
    """Test suite for WeightMapper class."""

    def setUp(self):
        """Set up test fixtures."""
        self.basis_list = ["AAPL", "MSFT"]
        # Coefficient pivot: assets x basis
        self.coeffs_pivot = pd.DataFrame(
            {"AAPL": [1.0, 0.0], "MSFT": [0.0, 1.0]},
            index=["AAPL", "MSFT"],
        )
        # Index weights
        self.index_weights = pd.DataFrame(
            {"AAPL": [0.6], "MSFT": [0.4]}, index=["2020-01-31"]
        )

    def test_weight_mapper_init(self):
        """Test WeightMapper initialization."""
        with TemporaryDirectory() as tmpdir:
            basis_path = Path(tmpdir) / "basis.csv"
            coeffs_path = Path(tmpdir) / "coeffs.csv"
            basis_path.write_text("")
            coeffs_path.write_text("")

            mapper = WeightMapper(
                basis_path=basis_path,
                coeffs_ts_path=coeffs_path,
                turnover_lambda=0.5,
            )
            self.assertEqual(mapper.turnover_lambda, 0.5)
            self.assertIsNone(mapper._weights_df)

    def test_weight_mapper_with_weights_df(self):
        """Test WeightMapper initialized with in-memory weights."""
        with TemporaryDirectory() as tmpdir:
            mapper = WeightMapper(
                basis_path=Path(tmpdir) / "basis.csv",
                coeffs_ts_path=Path(tmpdir) / "coeffs.csv",
                weights_df=self.index_weights,
            )
            self.assertIsNotNone(mapper._weights_df)
            self.assertEqual(mapper._weights_df.shape[1], 2)

    def test_weight_mapper_turnover_lambda_cast(self):
        """Test that turnover_lambda is properly cast to float."""
        with TemporaryDirectory() as tmpdir:
            mapper = WeightMapper(
                basis_path=Path(tmpdir) / "basis.csv",
                coeffs_ts_path=Path(tmpdir) / "coeffs.csv",
                turnover_lambda="0.5",
            )
            self.assertEqual(mapper.turnover_lambda, 0.5)
            self.assertIsInstance(mapper.turnover_lambda, float)

    def test_compute_from_coeffs_simple(self):
        """Test weight computation from coefficients."""
        with TemporaryDirectory() as tmpdir:
            mapper = WeightMapper(
                basis_path=Path(tmpdir) / "basis.csv",
                coeffs_ts_path=Path(tmpdir) / "coeffs.csv",
            )
            w_spx = pd.Series([0.6, 0.4], index=["AAPL", "MSFT"])
            result = mapper._compute_from_coeffs(
                self.coeffs_pivot, w_spx, self.basis_list
            )
            # With identity coefficients, should recover weights
            self.assertIsInstance(result, pd.Series)
            self.assertGreaterEqual(result.sum(), 0.0)

    def test_solve_qp_basis_weights(self):
        """Test QP solver for basis weights."""
        with TemporaryDirectory() as tmpdir:
            mapper = WeightMapper(
                basis_path=Path(tmpdir) / "basis.csv",
                coeffs_ts_path=Path(tmpdir) / "coeffs.csv",
                turnover_lambda=0.1,
            )
            A_date = self.coeffs_pivot
            b_vec = np.array([0.6, 0.4])  # target index weights
            basis_cols = self.basis_list
            w_prev = np.array([0.5, 0.5])

            result = mapper.solve_qp_basis_weights(
                A_date, b_vec, basis_cols, w_prev, lam=0.1
            )
            self.assertIsInstance(result, pd.Series)
            self.assertGreaterEqual(result.sum(), 0.0)


if __name__ == "__main__":
    unittest.main()
