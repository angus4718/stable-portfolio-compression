import heapq
import numpy as np
import pandas as pd

class DistancesUtils:
    @staticmethod
    def _corr_to_distance(corr: float) -> float:
        """
        Map a single correlation rho in [-1, 1] to distance d = sqrt(2(1 - rho)).
        """
        if np.isnan(corr):
            return np.nan
        if corr < -1 or corr > 1:
            raise ValueError('Correlation must be between -1 and 1.')
        return float(np.sqrt(2.0 * (1.0 - corr)))

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
    def price_to_return_df(price_df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert price levels to simple returns: r_t = P_t / P_{t-1} - 1
        The first row becomes NaN due to differencing.
        """
        if not isinstance(price_df, pd.DataFrame):
            raise TypeError("price_df must be a pandas DataFrame.")
        price_df = price_df.sort_index()
        return price_df.pct_change()

    @staticmethod
    def return_to_corr_df(return_df: pd.DataFrame,
                          min_periods: int = 1,
                          corr_method: str = "pearson") -> pd.DataFrame:
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
    def price_to_distance_df(price_df: pd.DataFrame,
                             min_periods: int = 1,
                             corr_method: str = "pearson") -> pd.DataFrame:
        """
        Pipeline: prices -> simple returns -> correlation -> optional denoising -> distance.
        """
        ret = DistancesUtils.price_to_return_df(price_df)
        corr = DistancesUtils.return_to_corr_df(ret, min_periods=min_periods, corr_method=corr_method)

        dist = DistancesUtils.corr_to_distance_df(corr)
        return dist


class UnionFind:
    def __init__(self, nodes):
        self.parents = {node: node for node in nodes}
        self.sizes = {node: 1 for node in nodes}
    
    def find(self, node):
        if self.parents[node] == node:
            return node
        
        parent = self.parents[node]
        self.parents[node] = self.find(parent)
        return self.parents[node]
    
    def union(self, node1, node2):
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

    def get_size(self, node):
        return self.sizes[self.find(node)]

class MST:
    """
    Assume dense graph where every node is connected to every other node. 
    """
    def __init__(self, adj_matrix, nodes):
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
        self.mst_adj = self._prim()

    def _make_edges_heap(self):
        edges_heap = []
        for i in range(self.n):
            for j in range(i + 1, self.n):
                w = self.adj_matrix[i][j]
                edges_heap.append((w, i, j))
        heapq.heapify(edges_heap)
        return edges_heap
    
    def _prim(self):
        in_mst = [False] * self.n
        key = [float('inf')] * self.n
        parent = [-1] * self.n

        key[0] = 0.0
        for _ in range(self.n):
            u = min((i for i in range(self.n) if not in_mst[i]), key=lambda i: key[i])
            in_mst[u] = True
            row = self.adj_matrix[u]
            for v in range(self.n):
                if not in_mst[v]:
                    w = row[v]
                    if w < key[v]:
                        key[v] = w
                        parent[v] = u
        
        # Build adjacency dict with node names
        mst_adj_dict = {node: [] for node in self.nodes}
        for v in range(1, self.n):
            u = parent[v]
            w = self.adj_matrix[u][v]
            node_u = self.nodes[u]
            node_v = self.nodes[v]
            mst_adj_dict[node_u].append((node_v, w))
            mst_adj_dict[node_v].append((node_u, w))

        return mst_adj_dict