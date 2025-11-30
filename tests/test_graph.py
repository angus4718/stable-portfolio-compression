import numpy as np
from pathlib import Path
import pandas as pd
import unittest

# Ensure project root is on sys.path so local package imports work when running script
import sys

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from spc.graph import DistancesUtils, MST, UnionFind, TreeUtils


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

    def test_corr_to_distance_bounds(self):
        # Distances should lie in [0, 2] for rho in [-1, 1]
        corr = pd.DataFrame(
            [[1.0, -1.0], [-1.0, 1.0]], index=["A", "B"], columns=["A", "B"]
        )
        dist = DistancesUtils.corr_to_distance_df(corr)
        self.assertAlmostEqual(dist.loc["A", "B"], 2.0, places=8)
        self.assertTrue(np.all(dist.values >= -1e-12))
        self.assertTrue(np.all(dist.values <= 2.0 + 1e-12))

    def test_return_to_corr_df_spearman(self):
        # Spearman should reflect monotonic ranking relationships
        r = pd.DataFrame(
            {
                "A": [1.0, 2.0, 3.0, 4.0],
                "B": [10.0, 20.0, 30.0, 40.0],
                "C": [4.0, 3.0, 2.0, 1.0],
            }
        )
        corr = DistancesUtils.return_to_corr_df(
            r, min_periods=1, corr_method="spearman"
        )
        # A and B are perfectly monotonic increasing -> spearman ~ 1
        self.assertAlmostEqual(corr.loc["A", "B"], 1.0, places=8)
        # A and C are perfectly inverse -> spearman ~ -1
        self.assertAlmostEqual(corr.loc["A", "C"], -1.0, places=8)

    def test_pca_denoise_handles_all_nan_column(self):
        # Create returns where one column is entirely NaN; PCA denoiser should still return valid matrix
        r = pd.DataFrame(np.random.randn(50, 3), columns=["A", "B", "C"])
        r.iloc[:, 2] = np.nan
        corr = DistancesUtils.pca_denoise_corr(r, n_components=1, min_periods=1)
        self.assertEqual(corr.shape, (3, 3))
        self.assertTrue(np.allclose(np.diag(corr.values), 1.0))
        self.assertTrue(np.allclose(corr.values, corr.values.T, atol=1e-8))

    def test_corr_to_distance_df_type_error(self):
        with self.assertRaises(TypeError):
            DistancesUtils.corr_to_distance_df([1, 2, 3])

    def test_price_to_return_df_type_error(self):
        with self.assertRaises(TypeError):
            DistancesUtils.price_to_return_df([1, 2, 3])

    def test_return_to_corr_df_invalid_method(self):
        df = pd.DataFrame({"A": [1.0, 2.0], "B": [2.0, 3.0]})
        with self.assertRaises(ValueError):
            DistancesUtils.return_to_corr_df(df, corr_method="bad")

    def test_pca_denoise_type_error(self):
        with self.assertRaises(TypeError):
            DistancesUtils.pca_denoise_corr([1, 2, 3])

    def test_lw_shrink_corr_type_error(self):
        with self.assertRaises(TypeError):
            DistancesUtils.lw_shrink_corr([1, 2, 3])

    def test_cov_to_corr_matrix_basic_and_zero_variance(self):
        cov = np.array([[4.0, 1.0], [1.0, 9.0]])
        cols = ["A", "B"]
        corr_df = DistancesUtils.cov_to_corr_matrix(cov, cols)
        self.assertEqual(corr_df.shape, (2, 2))
        self.assertAlmostEqual(corr_df.loc["A", "A"], 1.0)
        self.assertAlmostEqual(corr_df.loc["B", "B"], 1.0)
        self.assertTrue(-1.0 <= corr_df.loc["A", "B"] <= 1.0)

        cov2 = np.array([[0.0, 0.0], [0.0, 9.0]])
        corr_df2 = DistancesUtils.cov_to_corr_matrix(cov2, cols)
        # zero variance -> nan diagonal for that variable
        self.assertTrue(np.isnan(corr_df2.loc["A", "A"]))
        # off-diagonals involving A should be zero (constructed from zeros)
        self.assertAlmostEqual(corr_df2.loc["A", "B"], 0.0)

    def test_lw_shrink_and_price_to_distance_lw(self):
        np.random.seed(0)
        r = pd.DataFrame(np.random.randn(20, 3), columns=["X", "Y", "Z"])
        # Introduce some NaNs
        r.iloc[:5, 2] = np.nan
        corr_lw = DistancesUtils.lw_shrink_corr(r)
        self.assertEqual(corr_lw.shape, (3, 3))
        self.assertTrue(
            np.allclose(
                np.diag(corr_lw.values)[~np.isnan(np.diag(corr_lw.values))], 1.0
            )
        )

        prices = (1 + r.fillna(0)).cumprod()
        dist = DistancesUtils.price_to_distance_df(prices, shrink_method="lw")
        self.assertEqual(dist.shape, (3, 3))

    def test_price_to_distance_invalid_shrink(self):
        prices = pd.DataFrame({"A": [100.0, 101.0], "B": [50.0, 49.0]})
        with self.assertRaises(ValueError):
            DistancesUtils.price_to_distance_df(prices, shrink_method="invalid")

    def test_corr_to_distance_clips_out_of_range(self):
        corr = pd.DataFrame(
            [[1.5, -2.0], [-2.0, 1.5]], index=["A", "B"], columns=["A", "B"]
        )
        dist = DistancesUtils.corr_to_distance_df(corr)
        # values clipped to [-1,1] before transform: 1.5->1 -> distance 0; -2->-1 -> distance 2
        self.assertAlmostEqual(dist.loc["A", "B"], 2.0)

    def test_pca_explained_variance_branch(self):
        # create data dominated by a single common factor so first PC explains most variance
        rng = np.random.default_rng(0)
        f = rng.normal(size=200)
        X = np.vstack([2.0 * f + 0.01 * rng.normal(size=200) for _ in range(5)]).T
        df = pd.DataFrame(X, columns=[f"c{i}" for i in range(5)])
        corr_denoised = DistancesUtils.pca_denoise_corr(
            df, explained_variance=0.5, min_periods=1
        )
        self.assertEqual(corr_denoised.shape, (5, 5))
        self.assertTrue(np.allclose(np.diag(corr_denoised.values), 1.0))

    def test_pca_n_components_bounded(self):
        df = pd.DataFrame(np.random.randn(10, 3), columns=["a", "b", "c"])
        # request more components than assets; should not crash and return 3x3
        out = DistancesUtils.pca_denoise_corr(df, n_components=10, min_periods=1)
        self.assertEqual(out.shape, (3, 3))

    def test_lw_shrink_with_all_nan_column(self):
        r = pd.DataFrame(np.random.randn(30, 3), columns=["X", "Y", "Z"])
        r["Z"] = np.nan
        corr_lw = DistancesUtils.lw_shrink_corr(r)
        self.assertEqual(corr_lw.shape, (3, 3))
        # column with all NaNs: ensure returned matrix is valid (finite/symmetric)
        val = corr_lw.loc["Z", "Z"]
        self.assertTrue(np.isfinite(val))
        self.assertAlmostEqual(val, 1.0, places=8)

    def test_price_to_distance_with_nan_column(self):
        prices = pd.DataFrame(
            {"A": [100.0, 101.0, 102.0], "B": [np.nan, np.nan, np.nan]}
        )
        dist = DistancesUtils.price_to_distance_df(prices)
        # distances involving the all-NaN column should be NaN
        self.assertTrue(np.isnan(dist.loc["A", "B"]) and np.isnan(dist.loc["B", "A"]))

    def test_windowed_price_to_distance_expanding_and_rolling(self):
        # expanding window vs rolling tail behavior
        idx = pd.date_range("2020-01-01", periods=6, freq="M")
        prices = pd.DataFrame(
            {"A": [100, 101, 102, 103, 104, 105], "B": [50, 51, 52, 53, 54, 55]},
            index=idx,
        )
        end = pd.Timestamp("2020-04-30")

        dist_exp = DistancesUtils.windowed_price_to_distance(prices, end, window=None)
        dist_roll = DistancesUtils.windowed_price_to_distance(prices, end, window=3)

        prices_exp = prices.loc[:end]
        prices_roll = prices_exp.tail(3)

        # compare shapes and symmetry
        self.assertEqual(
            dist_exp.shape, DistancesUtils.price_to_distance_df(prices_exp).shape
        )
        self.assertEqual(
            dist_roll.shape, DistancesUtils.price_to_distance_df(prices_roll).shape
        )
        # values should match manual pipeline
        np.testing.assert_allclose(
            dist_exp.values,
            DistancesUtils.price_to_distance_df(prices_exp).values,
            equal_nan=True,
        )
        np.testing.assert_allclose(
            dist_roll.values,
            DistancesUtils.price_to_distance_df(prices_roll).values,
            equal_nan=True,
        )

    def test_price_to_return_df_non_datetime_index(self):
        prices = pd.DataFrame(
            {"X": [10.0, 11.0, 12.1], "Y": [5.0, 5.5, 6.0]}, index=[0, 1, 2]
        )
        returns = DistancesUtils.price_to_return_df(prices)
        # first row is NaN even for non-datetime index
        self.assertTrue(returns.iloc[0].isna().all())
        self.assertAlmostEqual(returns.iloc[1]["X"], 0.1, places=6)

    def test_cov_to_corr_matrix_negative_diag_handling(self):
        # tiny negative diagonal should be treated as non-positive and result in NaN diagonal
        cov = np.array([[-1e-8, 0.0], [0.0, 4.0]])
        cols = ["A", "B"]
        corr_df = DistancesUtils.cov_to_corr_matrix(cov, cols)
        # first diagonal was non-positive -> should be NaN
        self.assertTrue(np.isnan(corr_df.loc["A", "A"]))
        # other diagonal should be 1.0
        self.assertAlmostEqual(corr_df.loc["B", "B"], 1.0, places=8)


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

    def test_mst_edge_count_and_total_weight(self):
        # same matrix as above; expected MST edges = n-1 = 3 and total weight = 3.5
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

        # Count unique edges and sum weights
        edges = set()
        total_weight = 0.0
        for u in nodes:
            for v, w in adj[u]:
                a, b = tuple(sorted((u, v)))
                edges.add((a, b))
                total_weight += w
        # each edge counted twice in adjacency, so divide
        total_weight = total_weight / 2.0
        self.assertEqual(len(edges), len(nodes) - 1)
        self.assertAlmostEqual(total_weight, 3.5, places=8)

    def test_mst_random_graph_properties(self):
        # ensure MST on a random complete graph yields n-1 edges and symmetric adjacency
        rng = np.random.default_rng(123)
        n = 6
        mat = rng.random((n, n))
        mat = (mat + mat.T) / 2.0
        np.fill_diagonal(mat, 0.0)
        nodes = [f"n{i}" for i in range(n)]
        mst = MST(mat, nodes)
        adj = mst.mst_adj

        edges = set()
        for u in nodes:
            for v, _ in adj[u]:
                a, b = tuple(sorted((u, v)))
                edges.add((a, b))
        self.assertEqual(len(edges), n - 1)

    def test_mst_rejects_non_square_and_empty_and_node_mismatch(self):
        with self.assertRaises(ValueError):
            MST([])

        with self.assertRaises(ValueError):
            MST([[0, 1, 2], [1, 0, 1]])

        mat = [[0.0, 1.0], [1.0, 0.0]]
        with self.assertRaises(ValueError):
            MST(mat, nodes=["only_one"])

    def test_mst_default_node_labels(self):
        mat = [[0.0, 1.0], [1.0, 0.0]]
        mst = MST(mat)
        adj = mst.get_adj_dict()
        self.assertEqual(set(adj.keys()), {0, 1})

    def test_mst_ties_consistent_edge_count(self):
        # complete graph with equal weights
        n = 5
        mat = np.ones((n, n)) - np.eye(n)
        nodes = [f"v{i}" for i in range(n)]
        mst = MST(mat, nodes)
        adj = mst.get_adj_dict()
        edges = set()
        for u in nodes:
            for v, _ in adj[u]:
                a, b = tuple(sorted((u, v)))
                edges.add((a, b))
        self.assertEqual(len(edges), n - 1)

    def test_mst_handles_negative_weights(self):
        # negative weights should not break Kruskal's algorithm
        mat = np.array([[0.0, -1.0], [-1.0, 0.0]])
        mst = MST(mat)
        adj = mst.get_adj_dict()
        # single edge connecting the two nodes
        edges = set()
        for u in adj:
            for v, _ in adj[u]:
                a, b = tuple(sorted((u, v)))
                edges.add((a, b))
        self.assertEqual(len(edges), 1)


class TestUnionFind(unittest.TestCase):
    def test_unionfind_basic_operations(self):
        uf = UnionFind(["a", "b", "c"])
        self.assertEqual(uf.find("a"), "a")
        self.assertEqual(uf.get_size("a"), 1)
        merged = uf.union("a", "b")
        self.assertTrue(merged)
        self.assertEqual(uf.find("a"), uf.find("b"))
        self.assertEqual(uf.get_size("a"), 2)
        uf.union("b", "c")
        self.assertEqual(uf.get_size("a"), 3)

    def test_path_compression_behavior(self):
        uf = UnionFind(["a", "b", "c", "d"])
        uf.union("a", "b")
        uf.union("b", "c")
        # After unions, root should be common
        root_a = uf.find("a")
        root_c = uf.find("c")
        self.assertEqual(root_a, root_c)
        # parents for a and c should point to same root
        self.assertEqual(uf.parents["a"], uf.parents["c"])

    def test_unionfind_union_idempotent_returns_false(self):
        uf = UnionFind(["x", "y"])
        self.assertTrue(uf.union("x", "y"))
        # second union between same components returns False
        self.assertFalse(uf.union("x", "y"))

    def test_unionfind_path_compression_sets_parents_to_root(self):
        uf = UnionFind(["p", "q", "r", "s"])
        uf.union("p", "q")
        uf.union("q", "r")
        root = uf.find("p")
        # force_find on others
        uf.find("q")
        uf.find("r")
        for n in ("p", "q", "r"):
            self.assertEqual(uf.parents[n], root)


class TestTreeUtils(unittest.TestCase):
    def test_nodes_from_adj_and_degrees(self):
        adj = {"A": [("B", 1.0)], "B": [("A", 1.0), ("C", 2.0)], "C": [("B", 2.0)]}
        nodes = TreeUtils.nodes_from_adj(adj)
        self.assertEqual(set(nodes), set(adj.keys()))
        degs = TreeUtils.degrees(adj)
        self.assertEqual(degs["A"], 1)
        self.assertEqual(degs["B"], 2)

    def test_all_pairs_distance_and_path_length(self):
        # simple chain A-B-C with weights 1,1
        adj = {"A": [("B", 1.0)], "B": [("A", 1.0), ("C", 1.0)], "C": [("B", 1.0)]}
        dist = TreeUtils.all_pairs_tree_distance(adj)
        self.assertAlmostEqual(dist["A"]["C"], 2.0)
        self.assertAlmostEqual(dist["C"]["A"], 2.0)

        # path length between
        pl = TreeUtils.path_length_between(adj, "A", "C")
        self.assertAlmostEqual(pl, 2.0)

    def test_remove_nodes_and_components(self):
        adj = {
            "A": [("B", 1.0)],
            "B": [("A", 1.0), ("C", 1.0), ("D", 1.0)],
            "C": [("B", 1.0)],
            "D": [("B", 1.0)],
        }
        comps = TreeUtils.remove_nodes_and_components(adj, {"B"})
        # removing B isolates A, C, and D (no direct edges between them),
        # so we expect three components {A}, {C}, {D}
        comps_sets = [set(c) for c in comps]
        self.assertIn({"A"}, comps_sets)
        self.assertIn({"C"}, comps_sets)
        self.assertIn({"D"}, comps_sets)

    def test_path_betweenness_counts_properties(self):
        # chain of 4 nodes: A-B-C-D
        adj = {
            "A": [("B", 1.0)],
            "B": [("A", 1.0), ("C", 1.0)],
            "C": [("B", 1.0), ("D", 1.0)],
            "D": [("C", 1.0)],
        }
        betw = TreeUtils.path_betweenness_counts(adj)
        # middle nodes B and C should have larger betweenness than leaves
        self.assertTrue(betw["B"] > betw["A"])
        self.assertTrue(betw["C"] > betw["D"])

    def test_star_graph_degrees(self):
        # star graph with center 'C' connected to 4 leaves
        adj = {"C": [(f"L{i}", 1.0) for i in range(4)]}
        for i in range(4):
            adj[f"L{i}"] = [("C", 1.0)]
        nodes = TreeUtils.nodes_from_adj(adj)
        degs = TreeUtils.degrees(adj)
        self.assertIn("C", nodes)
        self.assertEqual(degs["C"], 4)
        for i in range(4):
            self.assertEqual(degs[f"L{i}"], 1)

    def test_path_betweenness_small_trees(self):
        # single-node tree
        adj1 = {"A": []}
        betw1 = TreeUtils.path_betweenness_counts(adj1)
        # no paths exist in single-node tree
        self.assertEqual(betw1.get("A", 0), 0)

        # two-node tree
        adj2 = {"A": [("B", 1.0)], "B": [("A", 1.0)]}
        betw2 = TreeUtils.path_betweenness_counts(adj2)
        # leaves should have zero betweenness
        self.assertEqual(betw2["A"], 0)
        self.assertEqual(betw2["B"], 0)

    def test_remove_nodes_and_components_remove_nonexistent_noop(self):
        adj = {"A": [("B", 1.0)], "B": [("A", 1.0), ("C", 1.0)], "C": [("B", 1.0)]}
        comps_before = TreeUtils.remove_nodes_and_components(adj, set())
        comps_after = TreeUtils.remove_nodes_and_components(adj, {"Z"})
        # removing a non-existent node should be a no-op and return same component structure
        self.assertEqual(len(comps_before), len(comps_after))


if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestDistancesUtils))
    suite.addTests(loader.loadTestsFromTestCase(TestMST))
    suite.addTests(loader.loadTestsFromTestCase(TestUnionFind))
    suite.addTests(loader.loadTestsFromTestCase(TestTreeUtils))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    if not result.wasSuccessful():
        sys.exit(1)
