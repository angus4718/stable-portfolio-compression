from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple, Any, Callable
import numpy as np

import sys

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from spc.graph import TreeUtils


class BasisSelector:
    """Basis selection on a tree (max-spread, hub-branch, error-driven).

    Lightweight helpers and small rewrites keep logic identical while reducing
    boilerplate and repeated code paths.
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

    def _normalize_dict(
        self, values: Dict[Any, float], fallback=0.0
    ) -> Dict[Any, float]:
        nodes = self.nodes
        if not values:
            return {u: float(fallback) for u in nodes}
        vals = [float(values.get(u, 0.0)) for u in nodes]
        mn, mx = min(vals), max(vals)
        rng = mx - mn if mx > mn else 1.0
        return {u: (float(values.get(u, 0.0)) - mn) / rng for u in nodes}

    def _avg_dist(self) -> Dict[Any, float]:
        return {
            u: np.mean([self._dist_uv(u, v) for v in self.nodes]) for u in self.nodes
        }

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
        normalized_weights = self._normalize_dict(weights or {})
        avg_dist = self._avg_dist()
        init_score = {
            u: alpha * avg_dist[u] + (1 - alpha) * normalized_weights[u] for u in nodes
        }
        b1 = TreeUtils.argmax_dict(init_score)
        B: List[Any] = [b1]

        nearest_dist = {v: self._dist_uv(v, b1) for v in nodes}

        while len(B) < k:
            candidate_scores = {
                v: alpha * nearest_dist[v] + (1 - alpha) * normalized_weights[v]
                for v in nodes
                if v not in B
            }
            if not candidate_scores:
                break
            v_star = TreeUtils.argmax_dict(candidate_scores)
            if v_star is None:
                break
            B.append(v_star)
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

        centrality = {
            u: alpha * float(self._deg.get(u, 0))
            + (1.0 - alpha) * float(self._betw.get(u, 0))
            for u in self.nodes
        }
        norm_centrality = self._normalize_dict(centrality)
        norm_weights = self._normalize_dict(weights or {})

        score = {
            u: (1.0 - weight_gamma) * norm_centrality[u]
            + weight_gamma * norm_weights[u]
            for u in self.nodes
        }
        hubs = sorted(self.nodes, key=lambda u: score[u], reverse=True)[:h]
        B = list(hubs)

        comps = TreeUtils.remove_nodes_and_components(self.adj, set(hubs))

        # Choose representative per component using combined distance-to-nearest-hub and weight
        def rep_for_comp(comp):
            best = None
            best_score = -1.0
            for v in comp:
                d = min(self._dist_uv(v, hh) for hh in hubs)
                w = norm_weights.get(v, 0.0)
                rep_score = rep_alpha * float(d) + (1.0 - rep_alpha) * float(w)
                if rep_score > best_score:
                    best_score = rep_score
                    best = v
            return (comp, best, best_score) if best is not None else None

        reps = [r for r in (rep_for_comp(c) for c in comps) if r is not None]
        reps.sort(key=lambda x: x[2], reverse=True)

        for _, rep, _ in reps:
            if len(B) >= k:
                break
            if rep not in B:
                B.append(rep)

        if len(B) < k:
            remaining = [v for v in self.nodes if v not in B]
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
