"""Test suite for spc/graph module."""

import unittest
import math
import numpy as np
import pandas as pd
from math import inf

from spc import graph
from spc.graph import DistancesUtils


class TestUnionFind(unittest.TestCase):
    def test_basic_find_union(self):
        uf = graph.UnionFind(["a", "b", "c"])
        self.assertEqual(uf.find("a"), "a")
        self.assertEqual(uf.get_size("a"), 1)
        self.assertTrue(uf.union("a", "b"))
        root_ab = uf.find("a")
        self.assertIn(root_ab, {"a", "b"})
        self.assertEqual(uf.get_size("a"), 2)
        self.assertFalse(uf.union("a", "b"))

    def test_union_by_size(self):
        uf = graph.UnionFind(["a", "b", "c", "d"])
        uf.union("a", "b")
        uf.union("c", "d")
        self.assertEqual(uf.get_size(uf.find("a")), 2)
        self.assertEqual(uf.get_size(uf.find("c")), 2)
        uf.union("a", "c")
        self.assertEqual(uf.get_size(uf.find("a")), 4)

    def test_path_compression(self):
        """Test that path compression flattens long chains."""
        uf = graph.UnionFind(["a", "b", "c", "d", "e"])
        # Create a chain: a -> b -> c -> d -> e (in terms of parent pointers)
        uf.union("a", "b")
        uf.union("b", "c")
        uf.union("c", "d")
        uf.union("d", "e")
        # Find from leaf should compress path
        root = uf.find("a")
        # After path compression, a should point directly to root
        self.assertEqual(uf.parents["a"], root)
        self.assertEqual(uf.get_size("a"), 5)

    def test_single_node(self):
        """Test UnionFind with a single node."""
        uf = graph.UnionFind(["x"])
        self.assertEqual(uf.find("x"), "x")
        self.assertEqual(uf.get_size("x"), 1)

    def test_empty_initialization(self):
        """Test UnionFind with no nodes."""
        uf = graph.UnionFind([])
        self.assertEqual(len(uf.parents), 0)


class TestMST(unittest.TestCase):
    def test_mst_small_triangle(self):
        adj_m = [
            [0.0, 1.0, 5.0],
            [1.0, 0.0, 2.0],
            [5.0, 2.0, 0.0],
        ]
        mst = graph.MST(adj_m)
        adj = mst.get_adj_dict()
        total_edges = sum(len(neigh) for neigh in adj.values()) // 2
        self.assertEqual(total_edges, 2)
        d01 = graph.TreeUtils.path_length_between(adj, 0, 1)
        self.assertTrue(math.isfinite(d01))

    def test_mst_non_square_raises(self):
        with self.assertRaises(ValueError):
            graph.MST([[1, 2, 3], [2, 1, 3]])

    def test_mst_empty_raises(self):
        """Test that empty matrix raises ValueError."""
        with self.assertRaises(ValueError):
            graph.MST([])

    def test_mst_single_node(self):
        """Test MST with a single node."""
        mst = graph.MST([[0.0]])
        adj = mst.get_adj_dict()
        self.assertEqual(len(adj), 1)
        self.assertEqual(adj[0], [])

    def test_mst_with_custom_labels(self):
        """Test MST construction with custom node labels."""
        adj_m = [
            [0.0, 1.0, 3.0],
            [1.0, 0.0, 2.0],
            [3.0, 2.0, 0.0],
        ]
        mst = graph.MST(adj_m, nodes=["A", "B", "C"])
        adj = mst.get_adj_dict()
        self.assertIn("A", adj)
        self.assertIn("B", adj)
        self.assertIn("C", adj)
        # MST should pick edges A-B (weight 1) and B-C (weight 2)
        total_edges = sum(len(neigh) for neigh in adj.values()) // 2
        self.assertEqual(total_edges, 2)

    def test_mst_nodes_mismatch_raises(self):
        """Test that mismatched node list raises ValueError."""
        adj_m = [[0.0, 1.0], [1.0, 0.0]]
        with self.assertRaises(ValueError):
            graph.MST(adj_m, nodes=["A", "B", "C"])

    def test_mst_complete_graph(self):
        """Test MST on a complete graph with 4 nodes."""
        adj_m = [
            [0.0, 1.0, 4.0, 5.0],
            [1.0, 0.0, 2.0, 3.0],
            [4.0, 2.0, 0.0, 1.5],
            [5.0, 3.0, 1.5, 0.0],
        ]
        mst = graph.MST(adj_m)
        adj = mst.get_adj_dict()
        total_edges = sum(len(neigh) for neigh in adj.values()) // 2
        self.assertEqual(total_edges, 3)  # n-1 edges for n nodes


class TestTreeUtils(unittest.TestCase):
    def test_bfs_and_path_length(self):
        adj = {0: [(1, 1.0)], 1: [(0, 1.0), (2, 2.0)], 2: [(1, 2.0)]}
        d_from_0 = graph.TreeUtils._bfs_from(adj, 0)
        self.assertAlmostEqual(d_from_0[0], 0.0)
        self.assertAlmostEqual(d_from_0[1], 1.0)
        self.assertAlmostEqual(d_from_0[2], 3.0)
        self.assertAlmostEqual(graph.TreeUtils.path_length_between(adj, 2, 0), 3.0)

    def test_all_pairs_tree_distance_simple(self):
        adj = {
            0: [(1, 1.0), (2, 1.0), (3, 1.0)],
            1: [(0, 1.0)],
            2: [(0, 1.0)],
            3: [(0, 1.0)],
        }
        ap = graph.TreeUtils.all_pairs_tree_distance(adj)
        self.assertEqual(ap[1][2], 2.0)
        self.assertEqual(ap[2][3], 2.0)
        self.assertEqual(ap[0][1], 1.0)

    def test_path_length_unreachable_returns_inf(self):
        adj = {0: [], 1: []}
        self.assertEqual(graph.TreeUtils.path_length_between(adj, 0, 1), inf)

    def test_remove_nodes_and_components(self):
        adj = {
            0: [(1, 1.0)],
            1: [(0, 1.0), (2, 1.0)],
            2: [(1, 1.0), (3, 1.0)],
            3: [(2, 1.0)],
        }
        comps = graph.TreeUtils.remove_nodes_and_components(adj, {1})
        comps_sorted = [set(c) for c in comps]
        self.assertIn({0}, comps_sorted)
        self.assertIn({2, 3}, comps_sorted)

    def test_remove_nodes_all_removed_returns_empty_list(self):
        adj = {0: [(1, 1)], 1: [(0, 1)]}
        comps = graph.TreeUtils.remove_nodes_and_components(adj, {0, 1})
        self.assertEqual(comps, [])

    def test_degrees(self):
        adj = {0: [(1, 1)], 1: [(0, 1), (2, 1)], 2: [(1, 1)]}
        deg = graph.TreeUtils.degrees(adj)
        self.assertEqual(deg[0], 1)
        self.assertEqual(deg[1], 2)
        self.assertEqual(deg[2], 1)

    def test_degrees_empty(self):
        self.assertEqual(graph.TreeUtils.degrees({}), {})

    def test_path_betweenness_chain(self):
        adj = {
            0: [(1, 1.0)],
            1: [(0, 1.0), (2, 1.0)],
            2: [(1, 1.0), (3, 1.0)],
            3: [(2, 1.0)],
        }
        betw = graph.TreeUtils.path_betweenness_counts(adj)
        expected = {0: 0, 1: 2, 2: 2, 3: 0}
        self.assertEqual(betw, expected)

    def test_nodes_from_adj(self):
        """Test extracting node list from adjacency dictionary."""
        adj = {0: [(1, 1.0)], 1: [(0, 1.0)], 2: [(3, 1.0)], 3: [(2, 1.0)]}
        nodes = graph.TreeUtils.nodes_from_adj(adj)
        self.assertEqual(set(nodes), {0, 1, 2, 3})

    def test_nodes_from_adj_empty(self):
        """Test nodes extraction from empty adjacency."""
        self.assertEqual(graph.TreeUtils.nodes_from_adj({}), [])

    def test_argmax_dict(self):
        """Test argmax helper function."""
        scores = {"a": 1.5, "b": 3.0, "c": 2.0}
        self.assertEqual(graph.TreeUtils.argmax_dict(scores), "b")

    def test_argmax_dict_empty(self):
        """Test argmax on empty dict returns None."""
        self.assertIsNone(graph.TreeUtils.argmax_dict({}))

    def test_argmax_dict_single(self):
        """Test argmax with single element."""
        self.assertEqual(graph.TreeUtils.argmax_dict({"x": 42}), "x")

    def test_path_length_same_node(self):
        """Test path length from node to itself."""
        adj = {0: [(1, 1.0)], 1: [(0, 1.0)]}
        self.assertEqual(graph.TreeUtils.path_length_between(adj, 0, 0), 0.0)

    def test_all_pairs_with_subset_nodes(self):
        """Test all_pairs_tree_distance with explicit node subset."""
        adj = {0: [(1, 1.0)], 1: [(0, 1.0), (2, 2.0)], 2: [(1, 2.0)]}
        ap = graph.TreeUtils.all_pairs_tree_distance(adj, nodes=[0, 2])
        self.assertEqual(ap[0][2], 3.0)
        self.assertNotIn(1, ap)  # node 1 not in requested subset

    def test_path_betweenness_star_graph(self):
        """Test betweenness on star graph (center should have high betweenness)."""
        # Star: center 0 connected to 1, 2, 3
        adj = {
            0: [(1, 1.0), (2, 1.0), (3, 1.0)],
            1: [(0, 1.0)],
            2: [(0, 1.0)],
            3: [(0, 1.0)],
        }
        betw = graph.TreeUtils.path_betweenness_counts(adj)
        # Center should have betweenness = 3 (all 3 choose 2 pairs pass through it)
        self.assertEqual(betw[0], 3)
        # Leaves should have 0
        self.assertEqual(betw[1], 0)
        self.assertEqual(betw[2], 0)
        self.assertEqual(betw[3], 0)

    def test_path_betweenness_empty(self):
        """Test betweenness on empty tree."""
        betw = graph.TreeUtils.path_betweenness_counts({})
        self.assertEqual(betw, {})

    def test_path_betweenness_single_node(self):
        """Test betweenness with single node."""
        betw = graph.TreeUtils.path_betweenness_counts({0: []})
        self.assertEqual(betw[0], 0)

    def test_remove_nodes_single_component_remains(self):
        """Test removing nodes leaves a single connected component."""
        adj = {
            0: [(1, 1.0), (2, 1.0)],
            1: [(0, 1.0)],
            2: [(0, 1.0)],
        }
        comps = graph.TreeUtils.remove_nodes_and_components(adj, {0})
        # Removing center leaves two isolated nodes
        self.assertEqual(len(comps), 2)
        self.assertIn({1}, [set(c) for c in comps])
        self.assertIn({2}, [set(c) for c in comps])


class TestDistancesUtils(unittest.TestCase):
    def test_corr_to_distance_and_returns(self):
        corr = pd.DataFrame(
            [[1.0, 0.5], [0.5, 1.0]], columns=["a", "b"], index=["a", "b"]
        )
        dist = DistancesUtils.corr_to_distance_df(corr)
        self.assertAlmostEqual(dist.loc["a", "b"], 1.0)
        self.assertAlmostEqual(dist.loc["a", "a"], 0.0)

        prices = pd.DataFrame({"a": [1.0, 2.0], "b": [2.0, 4.0]}, index=[0, 1])
        rets = DistancesUtils.price_to_return_df(prices)
        self.assertAlmostEqual(rets.iloc[1]["a"], 1.0)
        self.assertAlmostEqual(rets.iloc[1]["b"], 1.0)

    def test_return_to_corr_basic(self):
        rets = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [2.0, 4.0, 6.0]})
        corr = DistancesUtils.return_to_corr_df(
            rets, min_periods=1, corr_method="pearson"
        )
        self.assertAlmostEqual(corr.loc["x", "x"], 1.0)
        self.assertAlmostEqual(corr.loc["x", "y"], 1.0)

    def test_cov_to_corr_matrix(self):
        cov = np.array([[4.0, 2.0], [2.0, 9.0]])
        cols = ["a", "b"]
        corr_df = DistancesUtils.cov_to_corr_matrix(cov, cols)
        self.assertAlmostEqual(corr_df.loc["a", "b"], 1.0 / 3.0)
        self.assertAlmostEqual(corr_df.loc["a", "a"], 1.0)

    def test_pca_denoise_corr_perfect(self):
        rets = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [2.0, 4.0, 6.0]})
        corr_denoised = DistancesUtils.pca_denoise_corr(rets, n_components=1)
        self.assertAlmostEqual(corr_denoised.loc["x", "x"], 1.0)
        self.assertAlmostEqual(corr_denoised.loc["y", "y"], 1.0)
        self.assertAlmostEqual(corr_denoised.loc["x", "y"], corr_denoised.loc["y", "x"])

    def test_lw_shrink_corr_with_nans(self):
        rets = pd.DataFrame({"a": [np.nan, 0.01, 0.02], "b": [0.0, 0.01, np.nan]})
        corr = DistancesUtils.lw_shrink_corr(rets)
        self.assertIsInstance(corr, pd.DataFrame)
        self.assertListEqual(sorted(corr.columns.tolist()), ["a", "b"])
        self.assertTrue(np.isfinite(corr.loc["a", "a"]))

    def test_price_to_distance_df_perfect_corr(self):
        prices = pd.DataFrame(
            {"a": [1.0, 2.0, 3.0], "b": [2.0, 4.0, 6.0]},
            index=pd.date_range("2020-01-01", periods=3, freq="D"),
        )
        dist = DistancesUtils.price_to_distance_df(prices)
        self.assertAlmostEqual(dist.loc["a", "b"], 0.0)

    def test_windowed_price_to_distance_shapes(self):
        dates = pd.date_range("2020-01-01", periods=6, freq="D")
        prices = pd.DataFrame(
            {"a": np.arange(1, 7), "b": 2 * np.arange(1, 7)}, index=dates
        )
        D = DistancesUtils.windowed_price_to_distance(
            prices, end_date=dates[-1], window=3
        )
        self.assertEqual(list(D.index), ["a", "b"])
        self.assertEqual(D.shape, (2, 2))

    def test_corr_to_distance_type_error(self):
        """Test that non-DataFrame input raises TypeError."""
        with self.assertRaises(TypeError):
            DistancesUtils.corr_to_distance_df([[1.0, 0.5], [0.5, 1.0]])

    def test_price_to_return_df_type_error(self):
        """Test that non-DataFrame input raises TypeError."""
        with self.assertRaises(TypeError):
            DistancesUtils.price_to_return_df([[1.0, 2.0], [2.0, 4.0]])

    def test_return_to_corr_invalid_method(self):
        """Test that invalid corr_method raises ValueError."""
        rets = pd.DataFrame({"x": [1.0, 2.0], "y": [2.0, 4.0]})
        with self.assertRaises(ValueError):
            DistancesUtils.return_to_corr_df(rets, corr_method="invalid")

    def test_return_to_corr_type_error(self):
        """Test that non-DataFrame input raises TypeError."""
        with self.assertRaises(TypeError):
            DistancesUtils.return_to_corr_df([[1.0, 2.0]], corr_method="pearson")

    def test_price_to_distance_invalid_shrink_method(self):
        """Test that invalid shrink_method raises ValueError."""
        prices = pd.DataFrame({"a": [1.0, 2.0], "b": [2.0, 4.0]})
        with self.assertRaises(ValueError):
            DistancesUtils.price_to_distance_df(prices, shrink_method="invalid")

    def test_windowed_price_to_distance_invalid_shrink_method(self):
        """Test that invalid shrink_method raises ValueError in windowed version."""
        dates = pd.date_range("2020-01-01", periods=3, freq="D")
        prices = pd.DataFrame({"a": [1.0, 2.0, 3.0]}, index=dates)
        with self.assertRaises(ValueError):
            DistancesUtils.windowed_price_to_distance(
                prices, end_date=dates[-1], shrink_method="invalid"
            )

    def test_pca_denoise_corr_type_error(self):
        """Test that non-DataFrame input raises TypeError."""
        with self.assertRaises(TypeError):
            DistancesUtils.pca_denoise_corr([[1.0, 2.0]], n_components=1)

    def test_lw_shrink_corr_type_error(self):
        """Test that non-DataFrame input raises TypeError."""
        with self.assertRaises(TypeError):
            DistancesUtils.lw_shrink_corr([[0.01, 0.02]])

    def test_corr_to_distance_negative_corr(self):
        """Test distance calculation with negative correlations."""
        corr = pd.DataFrame(
            [[1.0, -0.5], [-0.5, 1.0]], columns=["a", "b"], index=["a", "b"]
        )
        dist = DistancesUtils.corr_to_distance_df(corr)
        expected_dist = np.sqrt(2.0 * (1.0 - (-0.5)))
        self.assertAlmostEqual(dist.loc["a", "b"], expected_dist)

    def test_windowed_price_to_distance_no_window(self):
        """Test windowed distance with no window (expanding)."""
        dates = pd.date_range("2020-01-01", periods=5, freq="D")
        prices = pd.DataFrame(
            {"a": [1.0, 2.0, 3.0, 4.0, 5.0], "b": [2.0, 4.0, 6.0, 8.0, 10.0]},
            index=dates,
        )
        D = DistancesUtils.windowed_price_to_distance(
            prices, end_date=dates[2], window=None
        )
        self.assertEqual(D.shape, (2, 2))
        # Perfect correlation should give distance near 0
        self.assertAlmostEqual(D.loc["a", "b"], 0.0, places=5)

    def test_return_to_corr_spearman(self):
        """Test correlation with Spearman method."""
        rets = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [1.0, 2.0, 3.0]})
        corr = DistancesUtils.return_to_corr_df(
            rets, min_periods=2, corr_method="spearman"
        )
        self.assertAlmostEqual(corr.loc["x", "y"], 1.0)

    def test_cov_to_corr_zero_variance(self):
        """Test cov_to_corr with zero variance column."""
        cov = np.array([[0.0, 0.0], [0.0, 4.0]])
        cols = ["a", "b"]
        corr = DistancesUtils.cov_to_corr_matrix(cov, cols)
        self.assertTrue(np.isnan(corr.loc["a", "a"]))
        self.assertEqual(corr.loc["b", "b"], 1.0)

    def test_pca_denoise_with_explained_variance(self):
        """Test PCA denoising using explained_variance threshold."""
        np.random.seed(42)
        # Create data with strong first component
        rets = pd.DataFrame(
            {
                "a": [1.0, 2.0, 3.0, 4.0],
                "b": [1.1, 2.1, 3.1, 4.1],
                "c": [0.9, 1.9, 2.9, 3.9],
            }
        )
        corr = DistancesUtils.pca_denoise_corr(rets, explained_variance=0.8)
        self.assertIsInstance(corr, pd.DataFrame)
        self.assertEqual(corr.shape, (3, 3))

    def test_price_to_distance_with_lw_shrinkage(self):
        """Test price_to_distance_df with Ledoit-Wolf shrinkage."""
        prices = pd.DataFrame(
            {"a": [1.0, 2.0, 3.0], "b": [2.0, 4.0, 6.0], "c": [1.5, 2.5, 3.5]},
            index=pd.date_range("2020-01-01", periods=3, freq="D"),
        )
        dist = DistancesUtils.price_to_distance_df(prices, shrink_method="lw")
        self.assertEqual(dist.shape, (3, 3))
        self.assertAlmostEqual(dist.loc["a", "a"], 0.0)

    def test_price_to_distance_with_pca(self):
        """Test price_to_distance_df with PCA denoising."""
        prices = pd.DataFrame(
            {"a": [1.0, 2.0, 3.0], "b": [2.0, 4.0, 6.0]},
            index=pd.date_range("2020-01-01", periods=3, freq="D"),
        )
        dist = DistancesUtils.price_to_distance_df(
            prices, shrink_method="pca", pca_n_components=1
        )
        self.assertEqual(dist.shape, (2, 2))


if __name__ == "__main__":
    unittest.main()
