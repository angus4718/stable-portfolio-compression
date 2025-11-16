from collections import deque
import numpy as np


class TreeUtils:
    """Small utilities for working with a tree represented as an adjacency dict.

    Adjacency format: dict[node] -> list of (neighbor, weight)
    """

    @staticmethod
    def nodes_from_adj(adj):
        return list(adj.keys())

    @staticmethod
    def all_pairs_tree_distance(adj, nodes=None):
        """Compute all-pairs distances on a tree via repeated BFS/DFS.

        Returns nested dict dist[u][v] with path lengths (sum of edge weights).
        """
        if nodes is None:
            nodes = list(adj.keys())
        dist = {u: {} for u in nodes}
        for s in nodes:
            # single-source distances via BFS-like traversal accumulating weights
            d = {s: 0.0}
            q = deque([s])
            while q:
                u = q.popleft()
                for v, w in adj.get(u, []):
                    if v not in d:
                        d[v] = d[u] + float(w)
                        q.append(v)
            # fill missing as inf for completeness
            for v in nodes:
                dist[s][v] = d.get(v, float("inf"))
        return dist

    @staticmethod
    def path_length_between(adj, src, dst):
        """Compute path length between src and dst via BFS; return inf if unreachable."""
        if src == dst:
            return 0.0
        q = deque([src])
        dist = {src: 0.0}
        seen = {src}
        while q:
            u = q.popleft()
            for v, w in adj.get(u, []):
                if v not in seen:
                    seen.add(v)
                    dist[v] = dist[u] + float(w)
                    if v == dst:
                        return dist[v]
                    q.append(v)
        return float("inf")

    @staticmethod
    def argmax_dict(d):
        items = list(d.items())
        return max(items, key=lambda kv: kv[1])[0]


class BasisSelector:
    def __init__(self, mst_adj, nodes=None):
        self.adj = mst_adj
        self.nodes = (
            list(nodes) if nodes is not None else TreeUtils.nodes_from_adj(self.adj)
        )
        self.n = len(self.nodes)
        self._dist = TreeUtils.all_pairs_tree_distance(self.adj, self.nodes)

    def _dist_uv(self, u, v):
        return self._dist[u][v]

    def select_max_spread(self, k):
        """Greedy 2-approximation for k-center on tree metric.

        Returns list of k selected nodes (or fewer if k > n).
        """
        if k <= 0 or self.n == 0:
            return []
        if k >= self.n:
            return list(self.nodes)

        nodes = self.nodes

        # Initialization: pick node with maximum average distance to others
        avg_dist = {u: float(np.mean([self._dist[u][v] for v in nodes])) for u in nodes}

        b1 = TreeUtils.argmax_dict(avg_dist)
        B = [b1]

        # nearest distance to B for each node
        nearest = {v: self._dist_uv(v, b1) for v in nodes}

        while len(B) < k:
            # choose node maximizing nearest[v] among nodes not yet in B
            v_star = max(nodes, key=lambda v: nearest[v] if v not in B else -1.0)
            B.append(v_star)
            # update nearest distances
            for v in nodes:
                dvb = self._dist_uv(v, v_star)
                if dvb < nearest[v]:
                    nearest[v] = dvb
        return B
