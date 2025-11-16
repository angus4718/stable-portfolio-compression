from collections import deque
from typing import Dict, List, Optional, Sequence, Set, Tuple, Any

import numpy as np


class TreeUtils:
    """
    Utilities for working with a weighted tree (MST).
    Expects adjacency as dict: node -> list[(neighbor, weight)] with undirected edges.
    """

    @staticmethod
    def nodes_from_adj(adj: Dict[Any, List[Tuple[Any, float]]]) -> List[Any]:
        nodes = list(adj.keys())
        return nodes

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
        dist = {u: {} for u in nodes}
        for s in nodes:
            # single-source distances via BFS on tree
            d = {s: 0.0}
            parent = {s: None}
            q = deque([s])
            while q:
                u = q.popleft()
                for v, w in adj[u]:
                    if v not in parent:
                        parent[v] = u
                        d[v] = d[u] + float(w)
                        q.append(v)
            # Fill in for s
            for v in nodes:
                dist[s][v] = d.get(v, np.inf)
        return dist

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
        parent = {src: None}
        while q:
            u = q.popleft()
            for v, w in adj[u]:
                if v not in parent:
                    parent[v] = u
                    dist[v] = dist[u] + float(w)
                    if v == dst:
                        return dist[v]
                    q.append(v)
        return np.inf

    @staticmethod
    def argmax_dict(d: Dict[Any, float]) -> Any:
        items = list(d.items())
        return max(items, key=lambda kv: kv[1])[0]


class BasisSelector:
    """
    Implements three basis selection strategies on a minimum spanning tree.

    Methods
    - select_max_spread(k)
    - select_hub_branch(k, h=None, alpha=0.5)
    - select_error_driven(k, E, B_prev=None, gamma=0.0, delta=None)

    Parameters
    - mst_adj: adjacency dict of the MST: node -> list of (neighbor, weight)
    - nodes: optional explicit node ordering; defaults to keys of mst_adj
    - precompute_distances: whether to precompute all-pairs tree distances for speed
    """

    def __init__(
        self,
        mst_adj: Dict[Any, List[Tuple[Any, float]]],
        nodes: Optional[Sequence[Any]] = None,
        precompute_distances: bool = True,
    ):
        self.adj = mst_adj
        self.nodes = (
            list(nodes) if nodes is not None else TreeUtils.nodes_from_adj(self.adj)
        )
        self.n = len(self.nodes)
        self._dist = TreeUtils.all_pairs_tree_distance(self.adj, self.nodes)

    def _dist_uv(self, u: Any, v: Any) -> float:
        if self._dist is not None:
            return self._dist[u][v]
        return TreeUtils.path_length_between(self.adj, u, v)

    def select_max_spread_weighted(
        self, k: int, weights: Dict[Any, float], alpha: float = 0.5
    ) -> List[Any]:
        """
        Weighted max-spread: balances spread (tree distance coverage) and weight importance.

        Iteratively selects nodes that maximize:
            score(v) = alpha * (distance to nearest basis node) + (1 - alpha) * (average weight of v)

        Parameters:
        - k: number of basis nodes to select
        - weights: dict mapping node names to weights (e.g., average market cap weight over time)
        - alpha: trade-off parameter (0 to 1)
                 alpha=1.0: pure spread (ignores weights, same as select_max_spread)
                 alpha=0.0: pure weight (just picks top-k by weight)
                 alpha=0.5: balanced between spread and weight

        Returns:
        - List of selected basis nodes
        """
        if k <= 0 or self.n == 0:
            return []
        if k >= self.n:
            return list(self.nodes)

        nodes = self.nodes

        # Normalize weights to [0, 1]
        w_min = min(weights.values()) if weights else 0
        w_max = max(weights.values()) if weights else 1
        w_range = w_max - w_min if w_max > w_min else 1
        normalized_weights = {
            u: (weights.get(u, 0) - w_min) / w_range if w_range > 0 else 0.5
            for u in nodes
        }

        # Initialize with node maximizing combined score of distance + weight
        avg_dist = {u: np.mean([self._dist_uv(u, v) for v in nodes]) for u in nodes}
        init_score = {
            u: alpha * avg_dist[u] + (1 - alpha) * normalized_weights[u] for u in nodes
        }
        b1 = TreeUtils.argmax_dict(init_score)
        B: List[Any] = [b1]

        # Maintain for each node v its distance to nearest basis node
        nearest_dist = {v: self._dist_uv(v, b1) for v in nodes}

        # Greedily add nodes
        while len(B) < k:
            # Score each remaining node by its combined contribution
            candidate_scores = {}
            for v in nodes:
                if v not in B:
                    dist_to_basis = nearest_dist[v]
                    w = normalized_weights[v]
                    score = alpha * dist_to_basis + (1 - alpha) * w
                    candidate_scores[v] = score

            v_star = TreeUtils.argmax_dict(candidate_scores)
            if v_star is None:
                break
            B.append(v_star)

            # Update nearest distances to basis
            for v in nodes:
                dvb = self._dist_uv(v, v_star)
                if dvb < nearest_dist[v]:
                    nearest_dist[v] = dvb

        return B
