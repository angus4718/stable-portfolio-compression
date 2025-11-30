import heapq
import logging
import numpy as np
import pandas as pd
from typing import Optional
from sklearn.decomposition import PCA
from sklearn.covariance import LedoitWolf
from typing import Dict, List, Optional, Sequence, Set, Tuple, Any, Callable
from collections import deque

# Module logger
logger = logging.getLogger(__name__)


class DistancesUtils:
    # Simple in-memory cache for windowed computations keyed by window parameters and end date.
    _window_cache: Dict[tuple, pd.DataFrame] = {}

    @staticmethod
    def corr_to_distance_df(corr_df: pd.DataFrame) -> pd.DataFrame:
        """
        Map correlations in [-1, 1] to distances d = sqrt(2(1 - rho)).
        Preserves index/columns and sets diagonal to 0.
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
        """
        Compute a distance DataFrame using only price data up to `end_date`.

        - `window` (int) : if provided, use the last `window` rows up to `end_date` (rolling);
                           otherwise use all rows up to `end_date` (expanding).
        - Results are cached per-run keyed by (end_date, window, shrink/PCA params, columns).
        """
        end_ts = pd.Timestamp(end_date)
        key = (
            end_ts.isoformat(),
            int(window) if window is not None else None,
            int(min_periods),
            corr_method,
            shrink_method,
            pca_n_components,
            pca_explained_variance,
            tuple(prices.columns.tolist()),
        )
        if key in DistancesUtils._window_cache:
            return DistancesUtils._window_cache[key]

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

        DistancesUtils._window_cache[key] = dist
        return dist

    @staticmethod
    def price_to_return_df(price_df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert price levels to simple returns: r_t = P_t / P_{t-1} - 1
        The first row becomes NaN due to differencing.
        """
        if not isinstance(price_df, pd.DataFrame):
            raise TypeError("price_df must be a pandas DataFrame.")
        price_df = price_df.sort_index()
        return price_df.pct_change(fill_method=None)

    @staticmethod
    def return_to_corr_df(
        return_df: pd.DataFrame, min_periods: int = 1, corr_method: str = "pearson"
    ) -> pd.DataFrame:
        """
        Compute the correlation matrix across columns using pairwise overlapping samples.
        corr_method: 'pearson' (linear) or 'spearman' (rank).
        min_periods: minimum number of observations required per pair to compute correlation.
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
        """
        Denoise correlation via PCA (principal component truncation).

        - If `n_components` supplied, use that many components.
        - Else if `explained_variance` supplied (0-1), choose smallest k explaining that fraction.
        - Else default to min(10, N-1).

        Process:
        1. Compute correlation matrix (handles NaN via pairwise overlap).
        2. Apply PCA to correlation matrix.
        3. Reconstruct correlation using truncated components.
        4. Clip to valid range [-1, 1] and set diagonal to 1.
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
        """
        Estimate a shrunk covariance using Ledoit-Wolf and convert to a correlation DataFrame.
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
        """
        Pipeline: prices -> simple returns -> correlation -> optional denoising -> distance.
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
    def __init__(self, nodes: list[str]) -> None:
        self.parents = {node: node for node in nodes}
        self.sizes = {node: 1 for node in nodes}

    def find(self, node: str) -> str:
        if self.parents[node] == node:
            return node

        parent = self.parents[node]
        self.parents[node] = self.find(parent)
        return self.parents[node]

    def union(self, node1: str, node2: str) -> bool:
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
        return self.sizes[self.find(node)]


class MST:
    """
    Assume dense graph where every node is connected to every other node.
    """

    def __init__(
        self, adj_matrix: list[list[float]], nodes: Optional[list[str]] = None
    ) -> None:
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
        edges_heap = []
        for i in range(self.n):
            for j in range(i + 1, self.n):
                w = self.adj_matrix[i][j]
                edges_heap.append((w, i, j))
        heapq.heapify(edges_heap)
        return edges_heap

    def _kruskal(self) -> dict[str, list[tuple[str, float]]]:
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
        return self.mst_adj


class TreeUtils:
    """
    Utilities for working with a weighted tree (MST).
    Expects adjacency as dict: node -> list[(neighbor, weight)] with undirected edges.
    """

    @staticmethod
    def nodes_from_adj(adj: Dict[Any, List[Tuple[Any, float]]]) -> List[Any]:
        return list(adj.keys())

    @staticmethod
    def all_pairs_tree_distance(
        adj: Dict[Any, List[Tuple[Any, float]]], nodes: Optional[Sequence[Any]] = None
    ) -> Dict[Any, Dict[Any, float]]:
        """
        Compute all-pairs distances on a tree via repeated BFS/DFS.

        Returns nested dict dist[u][v] with path lengths (sum of edge weights).
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
        """
        Compute path length between two nodes on a tree via BFS.
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
        if not d:
            return None
        return max(d, key=d.get)

    @staticmethod
    def remove_nodes_and_components(
        adj: Dict[Any, List[Tuple[Any, float]]], remove: Set[Any]
    ) -> List[Set[Any]]:
        """
        Remove 'remove' nodes from the tree, return connected components of the remaining graph.
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
        return {u: len(neigh) for u, neigh in adj.items()}

    @staticmethod
    def path_betweenness_counts(
        adj: Dict[Any, List[Tuple[Any, float]]],
    ) -> Dict[Any, int]:
        """
        For a tree, the number of simple paths passing through a node equals:
        sum over all unordered pairs of components created by removing that node.
        If removing u yields component sizes s1, s2, ..., sd (d = degree(u)),
        then betw(u) = sum_{i<j} s_i * s_j = ( (sum s_i)^2 - sum s_i^2 ) / 2.
        Here sum s_i = n - 1 (excluding u).
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
