"""Utilities for computing asset distances and tree topologies.

This module provides helpers to convert price/return series into correlation
and distance matrices (including Ledoit-Wolf shrinkage and PCA denoising),
and algorithms to build and analyze minimum spanning trees (MST).

Main components:
- DistancesUtils: price/return -> correlation -> distance pipeline.
- UnionFind: disjoint-set data structure for Kruskal's MST.
- MST: build minimum spanning tree from an adjacency matrix.
- TreeUtils: helpers for tree distances, components, degrees, and betweenness.
"""

import heapq
import logging
import numpy as np
import pandas as pd
from typing import Optional
from sklearn.decomposition import PCA
from sklearn.covariance import LedoitWolf
from typing import Dict, List, Optional, Sequence, Set, Tuple, Any
from collections import deque

# Module logger
logger = logging.getLogger(__name__)


class DistancesUtils:
    """Utilities to convert prices and returns into correlation and distance matrices.

    Provides static helpers for computing period returns, pairwise-overlap
    correlation matrices, and optional denoising using Ledoit-Wolf shrinkage
    or PCA-based reconstruction. Also exposes a mapping from correlation to
    metric distances usable for graph algorithms (e.g., MST construction).
    """

    @staticmethod
    def corr_to_distance_df(corr_df: pd.DataFrame) -> pd.DataFrame:
        """Convert a correlation DataFrame into a distance DataFrame.

        The transformation used is d = sqrt(2 * (1 - rho)), which maps
        correlations in [-1, 1] to non-negative distances. The returned
        DataFrame preserves the input index/columns and sets the diagonal
        to 0.0.

        Args:
            corr_df (pandas.DataFrame): Square correlation matrix.

        Returns:
            pandas.DataFrame: Pairwise distance matrix.
        """
        if not isinstance(corr_df, pd.DataFrame):
            raise TypeError("corr_df must be a pandas DataFrame.")
        corr_vals = corr_df.where(corr_df.isna(), corr_df.clip(lower=-1.0, upper=1.0))
        dist = np.sqrt(2.0 * (1.0 - corr_vals))
        np.fill_diagonal(dist.values, 0.0)
        return dist.astype(float)

    @staticmethod
    def windowed_price_to_distance(
        prices: pd.DataFrame,
        end_date,
        window: Optional[int] = None,
        min_periods: int = 1,
        corr_method: str = "pearson",
        shrink_method: Optional[str] = None,
        pca_n_components: Optional[int] = None,
        pca_explained_variance: Optional[float] = None,
    ) -> pd.DataFrame:
        """Compute distances using price history ending at ``end_date``.

        This helper slices the supplied ``prices`` DataFrame to the requested
        window (rolling or expanding) up to ``end_date``, invokes the price->
        distance pipeline, and caches results keyed by the window parameters
        and column ordering for reuse.

        Args:
            prices (pandas.DataFrame): Price level time series indexed by date.
            end_date (str | pandas.Timestamp): Inclusive end date for the window.
            window (int | None): Number of most recent rows to use; if ``None``
                an expanding window from the start is used.
            min_periods (int): Minimum observations required when computing correlations.
            corr_method (str): Correlation method, e.g. 'pearson' or 'spearman'.
            shrink_method (str | None): Optional denoising method: 'lw' or 'pca'.
            pca_n_components (int | None): Number of PCA components if using PCA.
            pca_explained_variance (float | None): Explained variance threshold if using PCA.

        Returns:
            pandas.DataFrame: Pairwise distance matrix for the requested window.
        """
        end_ts = pd.Timestamp(end_date)

        if window is None:
            prices_t = prices.loc[:end_ts]
        else:
            prices_t = prices.loc[:end_ts].tail(window)

        dist = DistancesUtils.price_to_distance_df(
            prices_t,
            min_periods=min_periods,
            corr_method=corr_method,
            shrink_method=shrink_method,
            pca_n_components=pca_n_components,
            pca_explained_variance=pca_explained_variance,
        )

        return dist

    @staticmethod
    def price_to_return_df(price_df: pd.DataFrame) -> pd.DataFrame:
        """Convert price levels to simple (period-over-period) returns.

        Args:
            price_df (pandas.DataFrame): Price levels with a DatetimeIndex.

        Returns:
            pandas.DataFrame: Period returns; the first row will be NaN.
        """
        if not isinstance(price_df, pd.DataFrame):
            raise TypeError("price_df must be a pandas DataFrame.")
        price_df = price_df.sort_index()
        return price_df.pct_change(fill_method=None)

    @staticmethod
    def return_to_corr_df(
        return_df: pd.DataFrame, min_periods: int = 1, corr_method: str = "pearson"
    ) -> pd.DataFrame:
        """Compute a pairwise correlation matrix handling NaNs via pairwise overlap.

        Uses ``DataFrame.corr`` with the provided ``corr_method`` and retains
        diagonal entries for columns that have at least ``min_periods``
        non-missing observations.

        Args:
            return_df (pandas.DataFrame): Returns DataFrame (columns = assets).
            min_periods (int): Minimum overlapping observations to compute a pair.
            corr_method (str): 'pearson' or 'spearman'.

        Returns:
            pandas.DataFrame: Correlation matrix with shape (n_assets, n_assets).
        """
        if not isinstance(return_df, pd.DataFrame):
            raise TypeError("return_df must be a pandas DataFrame.")
        if corr_method not in {"pearson", "spearman"}:
            raise ValueError("corr_method must be 'pearson' or 'spearman'.")

        corr = return_df.corr(method=corr_method, min_periods=min_periods)
        valid = return_df.notna().sum(axis=0) >= max(1, min_periods)
        for i, col in enumerate(corr.columns):
            if valid.get(col, False):
                corr.iat[i, i] = 1.0
        return corr

    @staticmethod
    def cov_to_corr_matrix(cov_matrix: np.ndarray, cols: list[str]):
        """Convert a covariance matrix to a correlation DataFrame.

        Args:
            cov_matrix (numpy.ndarray): Square covariance matrix.
            cols (list[str]): Column/row labels for the resulting DataFrame.

        Returns:
            pandas.DataFrame: Correlation matrix with labels ``cols``.
        """
        # Convert to correlation
        diag = np.diag(cov_matrix).copy()
        diag = np.where(diag <= 0.0, 0.0, diag)
        with np.errstate(divide="ignore", invalid="ignore"):
            inv_sd = 1.0 / np.sqrt(diag)
            inv_sd[~np.isfinite(inv_sd)] = 0.0
        D_inv = np.diag(inv_sd)
        Corr_hat = D_inv @ cov_matrix @ D_inv
        Corr_hat = np.clip(Corr_hat, -1.0, 1.0)
        for i in range(Corr_hat.shape[0]):
            Corr_hat[i, i] = 1.0 if diag[i] > 0 else np.nan

        return pd.DataFrame(Corr_hat, index=cols, columns=cols)

    @staticmethod
    def pca_denoise_corr(
        return_df: pd.DataFrame,
        n_components: Optional[int] = None,
        explained_variance: Optional[float] = None,
        min_periods: int = 1,
    ) -> pd.DataFrame:
        """Denoise a correlation matrix by truncating PCA components.

        The method computes a correlation matrix (pairwise overlap), performs
        PCA on that matrix and reconstructs it using a truncated number of
        components selected by either ``n_components`` or
        ``explained_variance``.

        Args:
            return_df (pandas.DataFrame): Returns DataFrame used to build the
                input correlation matrix.
            n_components (int | None): If provided, the exact number of PCA
                components to keep.
            explained_variance (float | None): If provided (0-1), choose the
                smallest k explaining at least this fraction of variance.
            min_periods (int): Minimum overlapping observations for computing
                the base correlation.

        Returns:
            pandas.DataFrame: Denoised correlation matrix (labels preserved).
        """
        if not isinstance(return_df, pd.DataFrame):
            raise TypeError("return_df must be a pandas DataFrame.")

        # Compute correlation matrix
        corr = DistancesUtils.return_to_corr_df(
            return_df, min_periods=min_periods, corr_method="pearson"
        )
        cols = corr.columns.tolist()
        corr_matrix = corr.values

        # Replace NaN correlations with 0
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

        n_assets = corr_matrix.shape[0]

        # Determine number of components
        if n_components is not None:
            k = min(n_components, n_assets)
        elif explained_variance is not None:
            # Use sklearn PCA to find k that explains the desired variance
            pca_temp = PCA()
            pca_temp.fit(corr_matrix)
            cumsum_var = np.cumsum(pca_temp.explained_variance_ratio_)
            k = np.argmax(cumsum_var >= explained_variance) + 1
            k = max(1, min(k, n_assets))
        else:
            # Default: use min(10, n_assets - 1)
            k = min(10, max(1, n_assets - 1))

        # Apply PCA
        pca = PCA(n_components=k)
        pca.fit(corr_matrix)

        # Reconstruct correlation matrix using truncated components
        corr_reconstructed = pca.inverse_transform(pca.transform(corr_matrix))
        logger.debug("PCA reconstructed correlation matrix:", corr_reconstructed)
        # Enforce symmetry
        corr_reconstructed = 0.5 * (corr_reconstructed + corr_reconstructed.T)

        # Clip to valid correlation range and fix diagonal
        corr_reconstructed = np.clip(corr_reconstructed, -1.0, 1.0)
        np.fill_diagonal(corr_reconstructed, 1.0)

        # Return as DataFrame with original index/columns
        return pd.DataFrame(corr_reconstructed, index=cols, columns=cols)

    @staticmethod
    def lw_shrink_corr(return_df: pd.DataFrame) -> pd.DataFrame:
        """Estimate a shrunk covariance (Ledoit-Wolf) and convert to correlation.

        Args:
            return_df (pandas.DataFrame): Returns DataFrame used to fit Ledoit-
                Wolf on filled data.

        Returns:
            pandas.DataFrame: Correlation matrix estimated from the shrunk covariance.
        """
        if not isinstance(return_df, pd.DataFrame):
            raise TypeError("return_df must be a pandas DataFrame.")

        cols = return_df.columns.tolist()

        # Fill NaNs with column medians, otherwise with zeros
        X = return_df.copy()
        col_medians = X.median(axis=0)
        X_filled = X.fillna(col_medians)
        X_filled = X_filled.fillna(0.0)

        # Fit Ledoit-Wolf estimator
        lw = LedoitWolf().fit(X_filled.values)
        cov_shrunk = lw.covariance_

        corr_df = DistancesUtils.cov_to_corr_matrix(cov_shrunk, cols)
        return corr_df

    @staticmethod
    def price_to_distance_df(
        price_df: pd.DataFrame,
        min_periods: int = 1,
        corr_method: str = "pearson",
        shrink_method: Optional[str] = None,
        pca_n_components: Optional[int] = None,
        pca_explained_variance: Optional[float] = None,
    ) -> pd.DataFrame:
        """Convert prices to distances by composing the pipeline.

        Steps: convert price levels to returns, compute pairwise correlation
        (or apply shrinkage / PCA), and map correlations into distances.

        Args:
            price_df (pandas.DataFrame): Price level DataFrame.
            min_periods (int): Minimum observations required for correlations.
            corr_method (str): Correlation method for base correlation.
            shrink_method (str | None): Optional 'lw' or 'pca' denoising method.
            pca_n_components (int | None): PCA components when using PCA.
            pca_explained_variance (float | None): Explained variance threshold.

        Returns:
            pandas.DataFrame: Pairwise distance matrix.
        """
        ret = DistancesUtils.price_to_return_df(price_df)

        if shrink_method is None:
            corr = DistancesUtils.return_to_corr_df(
                ret, min_periods=min_periods, corr_method=corr_method
            )
        elif shrink_method == "lw":
            corr = DistancesUtils.lw_shrink_corr(ret)
        elif shrink_method == "pca":
            corr = DistancesUtils.pca_denoise_corr(
                ret,
                n_components=pca_n_components,
                explained_variance=pca_explained_variance,
                min_periods=min_periods,
            )
        else:
            raise ValueError("shrink_method must be one of {None, 'lw', 'pca'}")

        dist = DistancesUtils.corr_to_distance_df(corr)
        return dist


class UnionFind:
    """Disjoint-set (union-find) data structure with path compression.

    Supports union-by-size and path compression. Designed for use with
    Kruskal's algorithm when building minimum spanning trees.
    """

    def __init__(self, nodes: list[str]) -> None:
        """Create a UnionFind structure for the provided nodes.

        Args:
            nodes (list[str]): List of node labels to initialize.
        """
        self.parents = {node: node for node in nodes}
        self.sizes = {node: 1 for node in nodes}

    def find(self, node: str) -> str:
        """Find and return the representative (root) of ``node``.

        Path compression is applied to flatten the tree for future queries.

        Args:
            node (str): Node label to find.

        Returns:
            str: Root representative of the set containing ``node``.
        """
        if self.parents[node] == node:
            return node

        parent = self.parents[node]
        self.parents[node] = self.find(parent)
        return self.parents[node]

    def union(self, node1: str, node2: str) -> bool:
        """Union the sets containing ``node1`` and ``node2``.

        Uses union-by-size heuristic and returns ``True`` if a union took
        place (the nodes were previously in different sets) or ``False`` if
        they were already connected.

        Args:
            node1 (str): First node.
            node2 (str): Second node.

        Returns:
            bool: ``True`` if the union merged two sets, ``False`` otherwise.
        """
        root1, root2 = self.find(node1), self.find(node2)
        if root1 == root2:
            return False

        size1, size2 = self.sizes[root1], self.sizes[root2]
        if size1 > size2:
            self.parents[root2] = root1
            self.sizes[root1] = size1 + size2
        else:
            self.parents[root1] = root2
            self.sizes[root2] = size1 + size2

        return True

    def get_size(self, node: str) -> int:
        """Return the size of the set containing ``node``.

        Args:
            node (str): Node label.

        Returns:
            int: Number of elements in the set containing ``node``.
        """
        return self.sizes[self.find(node)]


class MST:
    """Build a minimum spanning tree (MST) using Kruskal's algorithm.

    Constructed from a full adjacency (weight) matrix and optional node
    labels; the resulting MST is stored as an adjacency dictionary mapping
    each node to a list of (neighbor, weight) tuples.
    """

    def __init__(
        self, adj_matrix: list[list[float]], nodes: Optional[list[str]] = None
    ) -> None:
        """Initialize and compute the MST for the given adjacency matrix.

        Args:
            adj_matrix (list[list[float]]): Square adjacency/weight matrix.
            nodes (list[str] | None): Optional node labels corresponding to
                matrix rows/columns. If omitted, integer labels 0..n-1 are used.

        Raises:
            ValueError: If the adjacency matrix is empty or not square, or if
                the provided `nodes` length doesn't match the matrix size.
        """
        n = len(adj_matrix)
        if n == 0 or any(len(row) != n for row in adj_matrix):
            raise ValueError("Adjacency matrix must be non-empty and square.")
        self.adj_matrix = adj_matrix
        self.n = n

        # Default node labels to 0..n-1 if not provided
        self.nodes = list(range(n)) if nodes is None else nodes
        if len(self.nodes) != n:
            raise ValueError("nodes length must match matrix size.")
        self.node_index = {node: i for i, node in enumerate(self.nodes)}
        self.mst_adj = self._kruskal()

    def _make_edges_heap(self) -> list[tuple[float, int, int]]:
        """Create a min-heap of edges (weight, i, j) for Kruskal's algorithm."""
        edges_heap = []
        for i in range(self.n):
            for j in range(i + 1, self.n):
                w = self.adj_matrix[i][j]
                edges_heap.append((w, i, j))
        heapq.heapify(edges_heap)
        return edges_heap

    def _kruskal(self) -> dict[str, list[tuple[str, float]]]:
        """Run Kruskal's algorithm and return MST adjacency as a dict.

        Returns:
            dict: Mapping node -> list of (neighbor, weight) tuples representing
            the undirected MST adjacency.
        """
        uf = UnionFind(self.nodes)
        edges_heap = self._make_edges_heap()
        mst_adj_dict = {node: [] for node in self.nodes}
        edges_used = 0

        while edges_heap and edges_used < self.n - 1:
            w, i, j = heapq.heappop(edges_heap)
            node_u = self.nodes[i]
            node_v = self.nodes[j]
            if uf.union(node_u, node_v):
                mst_adj_dict[node_u].append((node_v, w))
                mst_adj_dict[node_v].append((node_u, w))
                edges_used += 1

        return mst_adj_dict

    def get_adj_dict(self) -> dict[str, list[tuple[str, float]]]:
        """Return the MST adjacency dictionary.

        Returns:
            dict: Node -> list of (neighbor, weight) tuples.
        """
        return self.mst_adj


class TreeUtils:
    """Utilities for working with weighted trees (MST).

    Functions operate on adjacency dictionaries mapping node -> list of
    (neighbor, weight) tuples and provide helpers for distances, degrees,
    components after node removal, and path-betweenness counts.
    """

    @staticmethod
    def nodes_from_adj(adj: Dict[Any, List[Tuple[Any, float]]]) -> List[Any]:
        """Return the list of nodes in the adjacency dictionary.

        Args:
            adj (dict): Adjacency mapping node -> list[(neighbor, weight)].

        Returns:
            list: Node labels.
        """
        return list(adj.keys())

    @staticmethod
    def all_pairs_tree_distance(
        adj: Dict[Any, List[Tuple[Any, float]]], nodes: Optional[Sequence[Any]] = None
    ) -> Dict[Any, Dict[Any, float]]:
        """Compute all-pairs path lengths in a tree using BFS per source.

        Args:
            adj (dict): Tree adjacency mapping.
            nodes (sequence | None): Subset (or ordering) of nodes to compute;
                defaults to ``adj.keys()``.

        Returns:
            dict: Nested mapping ``dist[u][v]`` -> path length (float).
        """
        if nodes is None:
            nodes = list(adj.keys())

        def bfs_from(src):
            d = {src: 0.0}
            q = deque([src])
            while q:
                u = q.popleft()
                for v, w in adj[u]:
                    if v not in d:
                        d[v] = d[u] + float(w)
                        q.append(v)
            return d

        return {s: {v: bfs_from(s).get(v, np.inf) for v in nodes} for s in nodes}

    @staticmethod
    def path_length_between(
        adj: Dict[Any, List[Tuple[Any, float]]], src: Any, dst: Any
    ) -> float:
        """Return the path length between ``src`` and ``dst`` in the tree.

        If nodes are disconnected (should not occur in a tree) returns ``inf``.

        Args:
            adj (dict): Tree adjacency mapping.
            src: Source node.
            dst: Destination node.

        Returns:
            float: Path length (sum of weights) or ``numpy.inf`` if unreachable.
        """
        if src == dst:
            return 0.0
        q = deque([src])
        dist = {src: 0.0}
        while q:
            u = q.popleft()
            for v, w in adj[u]:
                if v not in dist:
                    dist[v] = dist[u] + float(w)
                    if v == dst:
                        return dist[v]
                    q.append(v)
        return np.inf

    @staticmethod
    def argmax_dict(d: Dict[Any, float]) -> Any:
        """Return the key with the maximum value in mapping ``d`` or ``None``.

        Args:
            d (dict): Mapping keys -> numeric values.

        Returns:
            Any | None: Key with maximum value or ``None`` when ``d`` is empty.
        """
        if not d:
            return None

        return max(d, key=d.get)

    @staticmethod
    def remove_nodes_and_components(
        adj: Dict[Any, List[Tuple[Any, float]]], remove: Set[Any]
    ) -> List[Set[Any]]:
        """Remove a set of nodes and return connected components of the remainder.

        Args:
            adj (dict): Original adjacency mapping.
            remove (set): Nodes to remove.

        Returns:
            list[set]: List of connected component node sets after removal.
        """
        remaining = set(adj.keys()) - remove
        seen = set()
        comps = []
        for s in remaining:
            if s in seen:
                continue
            comp = set()
            q = deque([s])
            seen.add(s)
            while q:
                u = q.popleft()
                comp.add(u)
                for v, _ in adj[u]:
                    if v in remaining and v not in seen:
                        seen.add(v)
                        q.append(v)
            comps.append(comp)
        return comps

    @staticmethod
    def degrees(adj: Dict[Any, List[Tuple[Any, float]]]) -> Dict[Any, int]:
        """Return degree (number of neighbors) for each node in ``adj``.

        Args:
            adj (dict): Adjacency mapping.

        Returns:
            dict: Mapping node -> integer degree.
        """
        return {u: len(neigh) for u, neigh in adj.items()}

    @staticmethod
    def path_betweenness_counts(
        adj: Dict[Any, List[Tuple[Any, float]]],
    ) -> Dict[Any, int]:
        """Compute the path betweenness count for every node in a tree.

        For a tree, the number of simple source-destination paths that pass
        through node ``u`` can be derived from component sizes produced by
        removing ``u``. The formula used is
        ``betw(u) = sum_{i<j} s_i * s_j`` which is computed efficiently.

        Args:
            adj (dict): Tree adjacency mapping.

        Returns:
            dict: Mapping node -> integer betweenness count.
        """
        nodes = list(adj.keys())
        n = len(nodes)
        if n == 0:
            return {}

        # Root at arbitrary node and compute subtree sizes via DFS
        root = nodes[0]
        parent = {root: None}
        order = [root]
        stack = [root]
        while stack:
            u = stack.pop()
            for v, _ in adj[u]:
                if v == parent.get(u):
                    continue
                parent[v] = u
                stack.append(v)
                order.append(v)

        subtree = {u: 1 for u in nodes}
        for u in reversed(order):
            for v, _ in adj[u]:
                if parent.get(v) == u:
                    subtree[u] += subtree[v]

        betw = {}
        total = n - 1
        for u in nodes:
            comp_sizes = [subtree[v] for v, _ in adj[u] if parent.get(v) == u]
            if parent.get(u) is not None:
                comp_sizes.append(total - subtree[u])
            sum_sq = sum(s * s for s in comp_sizes)
            betw[u] = (total * total - sum_sq) // 2
        return betw
