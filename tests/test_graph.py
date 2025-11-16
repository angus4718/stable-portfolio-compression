import numpy as np
from pathlib import Path
import pandas as pd
import unittest

# Ensure project root is on sys.path so local package imports work when running script
import sys

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from spc.graph import DistancesUtils, MST


import logging

logging.basicConfig(level=logging.DEBUG)
logging.getLogger("spc.graph").setLevel(logging.DEBUG)


class TestDistancesUtils(unittest.TestCase):
    def test_corr_to_distance_df_basic(self):
        corr = pd.DataFrame(
            [[1.0, 0.5, np.nan], [0.5, 1.0, -0.2], [np.nan, -0.2, 1.0]],
            index=["A", "B", "C"],
            columns=["A", "B", "C"],
        )

        dist = DistancesUtils.corr_to_distance_df(corr)

        # diagonal zeros
        self.assertTrue(np.allclose(np.diag(dist.values), 0.0))
        # symmetric
        self.assertEqual(dist.shape, (3, 3))
        self.assertTrue(np.allclose(dist.values, dist.values.T, equal_nan=True))
        # check a known mapping: rho=0.5 -> sqrt(2*(1-0.5)) = 1
        self.assertAlmostEqual(dist.loc["A", "B"], 1.0, places=6)
        # NaN correlation maps to NaN distance
        self.assertTrue(np.isnan(dist.loc["A", "C"]))

    def test_price_to_return_df_simple(self):
        prices = pd.DataFrame(
            {"X": [100.0, 110.0, 121.0], "Y": [50.0, 50.0, 55.0]},
            index=pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01"]),
        )

        returns = DistancesUtils.price_to_return_df(prices)
        # first row is NaN
        self.assertTrue(returns.iloc[0].isna().all())
        # second row returns
        self.assertAlmostEqual(returns.loc["2020-02-01", "X"], 0.10, places=6)
        self.assertAlmostEqual(returns.loc["2020-03-01", "X"], 0.10, places=6)

    def test_return_to_corr_df_min_periods(self):
        r = pd.DataFrame(
            {
                "A": [0.01, 0.02, np.nan, np.nan],
                "B": [0.02, 0.03, 0.01, np.nan],
                "C": [np.nan, np.nan, np.nan, np.nan],
            },
            index=pd.date_range("2020-01-01", periods=4),
        )

        corr = DistancesUtils.return_to_corr_df(r, min_periods=2, corr_method="pearson")
        self.assertEqual(corr.shape, (3, 3))
        self.assertEqual(corr.loc["A", "A"], 1.0)
        self.assertEqual(corr.loc["B", "B"], 1.0)
        self.assertTrue(np.isnan(corr.loc["C", "A"]) or np.isnan(corr.loc["A", "C"]))

    def test_pca_denoise_corr_handles_nans_and_returns_valid_matrix(self):
        r = pd.DataFrame(np.random.randn(50, 4), columns=list("ABCD"))
        r.iloc[:45, 3] = np.nan

        corr = DistancesUtils.pca_denoise_corr(r, n_components=2, min_periods=1)

        self.assertEqual(corr.shape, (4, 4))
        self.assertTrue(np.allclose(np.diag(corr.values), 1.0))
        self.assertTrue(np.allclose(corr.values, corr.values.T, atol=1e-8))

    def test_price_to_distance_df_pipeline_pca(self):
        np.random.seed(0)
        returns = pd.DataFrame(np.random.randn(60, 3) * 0.01, columns=["A", "B", "C"])
        prices = (1 + returns).cumprod()

        dist = DistancesUtils.price_to_distance_df(
            prices, shrink_method="pca", pca_n_components=2
        )
        self.assertEqual(dist.shape, (3, 3))
        self.assertTrue(np.allclose(np.diag(dist.values), 0.0))
        self.assertTrue(np.all(dist.values >= -1e-12))
        self.assertTrue(np.allclose(dist.values, dist.values.T, equal_nan=True))


class TestMST(unittest.TestCase):
    def test_mst_prim_labels_and_symmetry(self):
        nodes = ["N1", "N2", "N3", "N4"]
        mat = np.array(
            [
                [0.0, 1.0, 2.0, 3.0],
                [1.0, 0.0, 1.5, 2.5],
                [2.0, 1.5, 0.0, 1.0],
                [3.0, 2.5, 1.0, 0.0],
            ]
        )

        mst = MST(mat, nodes)
        adj = mst.mst_adj

        self.assertEqual(set(adj.keys()), set(nodes))
        for u in nodes:
            for v, w in adj[u]:
                found = [x for x, _ in adj[v]]
                self.assertIn(u, found)


if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestDistancesUtils))
    suite.addTests(loader.loadTestsFromTestCase(TestMST))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    if not result.wasSuccessful():
        sys.exit(1)
