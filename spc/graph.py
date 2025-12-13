"""Utilities for distances and tree topology helpers used by SPC.

This module provides helpers to convert price/return series into
correlation and distance matrices (with optional Ledoit-Wolf shrinkage
and PCA denoising), construct minimum spanning trees (MST) and inspect
tree properties such as node degrees, connected components and
path-betweenness counts.

Main components:
        - :class:`DistancesUtils`: price/return -> correlation -> distance pipeline.
        - :class:`UnionFind`: disjoint-set data structure used by Kruskal's MST.
        - :class:`MST`: build an MST from a full adjacency/weight matrix.
        - :class:`TreeUtils`: utilities for tree distances, components and
            centrality-like counts.
"""

from __future__ import annotations

import heapq
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.covariance import LedoitWolf
from typing import Dict, List, Optional, Sequence, Set, Tuple, Any
from collections import deque


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
        corr_vals = corr_df.clip(lower=-1.0, upper=1.0)
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
        distance pipeline, and returns the resulting distance matrix.

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

        prices_t = (
            prices.loc[:end_ts].tail(window)
            if window is not None
            else prices.loc[:end_ts]
        )

        ret = DistancesUtils.price_to_return_df(prices_t)

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

        return DistancesUtils.corr_to_distance_df(corr)

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
    def cov_to_corr_matrix(cov_matrix: np.ndarray, cols: List[str]) -> pd.DataFrame:
        """Convert a covariance matrix to a correlation DataFrame.

        Args:
            cov_matrix (numpy.ndarray): Square covariance matrix.
            cols (list[str]): Column/row labels for the resulting DataFrame.

        Returns:
            pd.DataFrame: Correlation matrix with labels ``cols``.
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
            # Use PCA to find k that explains the desired variance
            pca_temp = PCA()
            pca_temp.fit(corr_matrix)
            cumsum_var = np.cumsum(pca_temp.explained_variance_ratio_)
            k = np.argmax(cumsum_var >= explained_variance) + 1
            k = max(1, min(k, n_assets))
        else:
            k = max(1, n_assets // 2)

        # Apply PCA
        pca = PCA(n_components=k)
        pca.fit(corr_matrix)

        # Reconstruct correlation matrix using truncated components
        corr_reconstructed = pca.inverse_transform(pca.transform(corr_matrix))

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

        return DistancesUtils.corr_to_distance_df(corr)


class UnionFind:
    """Disjoint-set (union-find) data structure with path compression.

    Supports union-by-size and path compression. Designed for use with
    Kruskal's algorithm when building minimum spanning trees.
    """

    def __init__(self, nodes: List[str]) -> None:
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
        self, adj_matrix: List[List[float]], nodes: Optional[List[str]] = None
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

    def _make_edges_heap(self) -> List[Tuple[float, int, int]]:
        """Create a min-heap of edges (weight, i, j) for Kruskal's algorithm."""
        edges_heap = []
        for i in range(self.n):
            for j in range(i + 1, self.n):
                w = self.adj_matrix[i][j]
                edges_heap.append((w, i, j))
        heapq.heapify(edges_heap)
        return edges_heap

    def _kruskal(self) -> Dict[str, List[Tuple[str, float]]]:
        """Run Kruskal's algorithm and return MST adjacency as a dictionary.

        Returns:
            Dict[str, List[Tuple[str, float]]]: Mapping node -> list of
            (neighbor, weight) tuples representing the undirected MST
            adjacency.
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

    def get_adj_dict(self) -> Dict[str, List[Tuple[str, float]]]:
        """Return the MST adjacency dictionary.

        Returns:
            Dict[str, List[Tuple[str, float]]]: Node -> list of
            (neighbor, weight) tuples.
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
            adj (Dict[Any, List[Tuple[Any, float]]]): Adjacency mapping
                node -> list[(neighbor, weight)].

        Returns:
            List[Any]: Node labels.
        """
        return list(adj.keys())

    @staticmethod
    def argmax_dict(d: Dict[Any, float]) -> Optional[Any]:
        """Return the key with maximum value in mapping ``d`` or ``None``.

        Args:
            d (Dict[Any, float]): Mapping from keys to numeric scores.

        Returns:
            Any | None: Key with the maximum score or ``None`` when ``d`` is empty.

        This helper is used in selection routines to pick an initial seed
        when a mapping of per-node scores is available.
        """
        if not d:
            return None
        # Use max over keys with value as key function; return the argmax key
        return max(d.keys(), key=lambda k: d[k])

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

        return {
            s: {v: TreeUtils._bfs_from(adj, s).get(v, np.inf) for v in nodes}
            for s in nodes
        }

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
        dist = TreeUtils._bfs_from(adj, src)
        return dist.get(dst, np.inf)

    @staticmethod
    def _bfs_from(
        adj: Dict[Any, List[Tuple[Any, float]]], src: Any
    ) -> Dict[Any, float]:
        """Perform a breadth-first search from a source node and return distances.

        Distances are computed as the sum of edge weights along the unique path in
        the tree from ``src`` to each reachable node.

        Args:
            adj: Adjacency mapping where keys are node labels and values are
                lists of ``(neighbor, weight)`` tuples representing an
                undirected weighted tree.
            src: The source node label to start the traversal from.

        Returns:
            A dictionary mapping each reachable node label to the path length
            (float) from ``src``. Nodes that are not reachable from ``src``
            are not included in the returned mapping.
        """
        d: Dict[Any, float] = {src: 0.0}
        q = deque([src])
        while q:
            u = q.popleft()
            for v, w in adj[u]:
                if v not in d:
                    d[v] = d[u] + w
                    q.append(v)
        return d

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
        # Compute the set of nodes that remain after removing the requested
        # nodes. Using a set gives O(1) membership checks during BFS below.
        remaining = set(adj.keys()) - remove

        # `seen` keeps track of nodes we've already assigned to a component
        # so we never process a node twice. `comps` will accumulate the
        # discovered connected components (each as a set of node labels).
        seen = set()
        comps = []

        # Iterate over every remaining node. If the node is unvisited we
        # start a BFS to discover its entire connected component. If it's
        # already in `seen` it has been discovered as part of a previous BFS.
        for s in remaining:
            if s in seen:
                continue

            # Begin a new component discovery rooted at `s`.
            comp = set()
            q = deque([s])
            seen.add(s)

            # Standard BFS loop: pop a node, add to current component, and
            # enqueue its unvisited neighbors. Adjacency entries are
            # `(neighbor, weight)` pairs; weights are ignored because
            # connectivity (component membership) depends only on the
            # presence of an edge, not its weight.
            while q:
                u = q.popleft()
                comp.add(u)
                for v, _ in adj[u]:
                    # Only consider neighbors that still exist in the
                    # `remaining` set and haven't been seen yet.
                    if v in remaining and v not in seen:
                        seen.add(v)
                        q.append(v)

            # BFS finished: `comp` now contains one connected component of
            # the graph after removing the `remove` nodes. Append it to the
            # result list and continue scanning for other components.
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
            adj (Dict[Any, List[Tuple[Any, float]]]): Tree adjacency mapping.

        Returns:
            Dict[Any, int]: Mapping node -> integer betweenness count.
        """
        nodes = list(adj.keys())
        n = len(nodes)
        if n == 0:
            return {}

        # Overview of approach (optimized for trees):
        # 1. Root the tree at an arbitrary node and compute a parent map plus
        #    a DFS ordering (children discovered after parent). This lets us
        #    compute subtree sizes bottom-up.
        # 2. For each node u, removing u partitions the tree into several
        #    connected components (the subtrees rooted at u's children, plus
        #    the remainder that contains u's parent). The sizes of these
        #    components determine how many source-destination pairs have a
        #    path that goes through u.
        # 3. If the component sizes are s_1, s_2, ..., s_m (summing to
        #    total = n-1, excluding u itself), the number of unordered
        #    pairs whose path goes through u is sum_{i<j} s_i * s_j. This
        #    equals (total^2 - sum_i s_i^2) / 2 which we compute below.

        # Root at an arbitrary node and compute parent pointers using DFS.
        root = nodes[0]
        parent = {root: None}
        order: List[Any] = [root]
        stack = [root]
        while stack:
            u = stack.pop()
            for v, _ in adj[u]:
                # skip back-edge to parent
                if v == parent.get(u):
                    continue
                parent[v] = u
                stack.append(v)
                order.append(v)

        # Compute subtree sizes in reverse DFS order (children before parent).
        # subtree[u] counts the number of nodes in the subtree rooted at u
        # including u itself.
        subtree: Dict[Any, int] = {u: 1 for u in nodes}
        for u in reversed(order):
            for v, _ in adj[u]:
                # if v is a child of u, add its subtree size
                if parent.get(v) == u:
                    subtree[u] += subtree[v]

        # Now compute betweenness counts. For each node u, removing u
        # produces components whose sizes are the subtree sizes of its
        # children (those v with parent[v] == u). If u has a parent, there is
        # an additional component that consists of all nodes not in u's
        # subtree; its size equals (n - subtree[u]). We exclude u itself
        # from these component size calculations, so the sum of component
        # sizes equals total = n - 1.
        betw: Dict[Any, int] = {}
        total = n - 1
        for u in nodes:
            # collect sizes of components formed after removing u
            comp_sizes = [subtree[v] for v, _ in adj[u] if parent.get(v) == u]
            # include the remainder component (above u) when u is not root
            if parent.get(u) is not None:
                # remainder should be the nodes outside u's subtree (exclude u),
                # i.e. n - subtree[u]. Using `total - subtree[u]` was incorrect.
                comp_sizes.append(n - subtree[u])

            # Use algebraic identity: sum_{i<j} s_i*s_j = (total^2 - sum_i s_i^2)/2
            # This counts unordered pairs of nodes lying in different components,
            # which are exactly the node pairs whose unique path passes through u.
            sum_sq = sum(s**2 for s in comp_sizes)
            betw[u] = (total**2 - sum_sq) // 2

        return betw
