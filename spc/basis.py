from collections import deque
from typing import Dict, List, Optional, Sequence, Set, Tuple, Any, Callable
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

    @staticmethod
    def remove_nodes_and_components(
        adj: Dict[Any, List[Tuple[Any, float]]], remove: Set[Any]
    ) -> List[Set[Any]]:
        """
        Remove 'remove' nodes from the tree, return connected components of the remaining graph.
        """
        remaining = set(adj.keys()) - remove
        seen: Set[Any] = set()
        comps: List[Set[Any]] = []

        for s in list(remaining):
            if s in seen:
                continue
            comp = set()
            q = deque([s])
            seen.add(s)
            comp.add(s)
            while q:
                u = q.popleft()
                for v, _ in adj[u]:
                    if v in remaining and v not in seen:
                        seen.add(v)
                        comp.add(v)
                        q.append(v)
            comps.append(comp)
        return comps

    @staticmethod
    def degrees(adj: Dict[Any, List[Tuple[Any, float]]]) -> Dict[Any, int]:
        return {u: len(adj[u]) for u in adj}

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

        # Root the tree arbitrarily and compute subtree sizes
        root = nodes[0] if nodes else None

        # Build parent/children with DFS
        parent = {}
        order = []
        stack = [root] if root is not None else []
        parent[root] = None
        while stack:
            u = stack.pop()
            order.append(u)
            for v, _ in adj[u]:
                if v == parent.get(u):
                    continue
                parent[v] = u
                stack.append(v)

        # Post-order to compute subtree sizes
        subtree = {u: 1 for u in nodes}
        for u in reversed(order):
            for v, _ in adj[u]:
                if parent.get(v) == u:
                    subtree[u] += subtree[v]

        # For each node, component sizes when removing it:
        # - for each child v: size = subtree[v]
        # - for parent side: size = n - subtree[u]
        betw = {}
        for u in nodes:
            comp_sizes = []
            for v, _ in adj[u]:
                if parent.get(v) == u:
                    comp_sizes.append(subtree[v])
                elif parent.get(u) == v:
                    comp_sizes.append(n - subtree[u])
            total = n - 1
            sum_sq = sum(s * s for s in comp_sizes)
            betw[u] = (total * total - sum_sq) // 2
        return betw


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
    ):
        self.adj = mst_adj
        self.nodes = (
            list(nodes) if nodes is not None else TreeUtils.nodes_from_adj(self.adj)
        )
        self.n = len(self.nodes)
        self._dist = TreeUtils.all_pairs_tree_distance(self.adj, self.nodes)

        self._deg = TreeUtils.degrees(self.adj)
        self._betw = TreeUtils.path_betweenness_counts(self.adj)

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

    def select_hub_branch(
        self,
        k: int,
        h: Optional[int] = None,
        alpha: float = 0.5,
        weights: Optional[Dict[Any, float]] = None,
        weight_gamma: float = 0.0,
        rep_alpha: float = 0.5,
    ) -> List[Any]:
        """
        Choose h hubs by maximizing combined centrality C(u) = alpha*deg(u) + (1-alpha)*betweenness(u),
        then decompose tree minus hubs into branches (connected components) and add representatives
        from the largest remaining branches by path length until |B|=k.

        If h is None, we set h = max(1, min(k-1, round(0.2 * k))).
        """
        if k <= 0 or self.n == 0:
            return []
        if k >= self.n:
            return list(self.nodes)
        if h is None:
            h = max(1, min(k - 1, int(round(0.2 * k))))

        # Score for hubs: combine degree/betweenness centrality and optional weight importance
        centrality = {
            u: alpha * float(self._deg.get(u, 0))
            + (1.0 - alpha) * float(self._betw.get(u, 0))
            for u in self.nodes
        }

        # Normalize centrality and weights to [0,1] for blending
        cen_vals = list(centrality.values())
        cen_min = min(cen_vals) if cen_vals else 0.0
        cen_max = max(cen_vals) if cen_vals else 1.0
        cen_range = cen_max - cen_min if cen_max > cen_min else 1.0
        norm_centrality = {u: (centrality[u] - cen_min) / cen_range for u in self.nodes}

        if weights is None:
            norm_weights = {u: 0.0 for u in self.nodes}
        else:
            w_vals = [float(weights.get(u, 0.0)) for u in self.nodes]
            w_min = min(w_vals) if w_vals else 0.0
            w_max = max(w_vals) if w_vals else 1.0
            w_range = w_max - w_min if w_max > w_min else 1.0
            norm_weights = {
                u: (float(weights.get(u, 0.0)) - w_min) / w_range for u in self.nodes
            }

        # Blended score: low weight_gamma -> centrality-driven, high weight_gamma -> weight-driven
        score = {
            u: (1.0 - weight_gamma) * norm_centrality[u]
            + weight_gamma * norm_weights[u]
            for u in self.nodes
        }
        # Select top-h hubs
        hubs = sorted(self.nodes, key=lambda u: score[u], reverse=True)[:h]
        B = list(hubs)

        # Remove hubs and get connected components (branches)
        comps = TreeUtils.remove_nodes_and_components(self.adj, set(hubs))

        # Compute nearest hub and distance for nodes
        def nearest_hub(v: Any) -> Tuple[Any, float]:
            best = (None, float("inf"))
            for h_ in hubs:
                d = self._dist_uv(v, h_)
                if d < best[1]:
                    best = (h_, d)
            return best

        reps = []  # (component, rep, distance-to-nearest-hub)
        for comp in comps:
            rep = None
            best_score = -1.0
            for v in comp:
                _, d = nearest_hub(v)
                w = norm_weights.get(v, 0.0)
                # Combined representative score: rep_alpha * distance + (1-rep_alpha) * weight
                rep_score = rep_alpha * float(d) + (1.0 - rep_alpha) * float(w)
                if rep_score > best_score:
                    best_score = rep_score
                    rep = v
            if rep is not None:
                reps.append((comp, rep, best_score))

        # Sort components by best_d descending (farther peripheries first)
        reps.sort(key=lambda x: x[2], reverse=True)

        # Add representatives until we reach k
        i = 0
        while len(B) < k and i < len(reps):
            _, rep, _ = reps[i]
            if rep not in B:
                B.append(rep)
            i += 1

        # If still short, fill by farthest-point rule among remaining
        if len(B) < k:
            remaining = [v for v in self.nodes if v not in B]
            # Reuse nearest-distance maintenance
            nearest = {v: min(self._dist_uv(v, b) for b in B) for v in remaining}
            while len(B) < k and remaining:
                v_star = max(remaining, key=lambda v: nearest[v])
                B.append(v_star)
                remaining.remove(v_star)
                for v in remaining:
                    dvb = self._dist_uv(v, v_star)
                    if dvb < nearest[v]:
                        nearest[v] = dvb

        return B
