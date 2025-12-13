"""Portfolio construction helpers and basis selection utilities.

This module implements utilities used by the project's prototype
portfolio-construction pipeline. It contains:

- ``BasisSelector``: algorithms that pick representative basis nodes from a
    tree (typically an MST) using either a max-spread greedy rule or a
    hub-and-branch decomposition.
- ``LocalRidgeRunner``: a small local Ridge regression pipeline used to
    produce coefficient estimates, reconstructed returns and regression
    diagnostics for non-basis assets.
- ``WeightMapper``: helpers that map index-level weights to basis weights
    using coefficient pivots and an optional turnover-penalized QP solver.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Sequence, Any, Union
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
import heapq
from scipy.optimize import minimize

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from spc.graph import DistancesUtils, TreeUtils
from spc.utils import (
    load_basis_list_from_path,
    load_prices_csv,
)


class BasisSelector:
    """Select representative basis nodes from a tree (MST).

    The selector extracts a compact set of representative "basis" nodes
    from a tree topology, typically the minimum spanning tree computed
    from pairwise asset distances. Two selection strategies are exposed:

    - ``select_max_spread``: a greedy farthest-first / max-spread rule that
      iteratively adds the node maximizing a combination of distance-to-
      nearest-basis and a normalized per-node weight.
    - ``select_hub_branch``: picks a small set of central "hubs" (by a
      mixture of degree and simple path betweenness) and then selects one
      representative from each remaining branch.

    Attributes:
        adj (dict): Tree adjacency mapping (node -> list[(neighbor, weight)]).
        nodes (list): Ordered list of nodes used by the selector.
        n (int): Number of nodes in ``nodes``.
        _dist (dict): All-pairs tree distances indexed by node pairs.
        _deg (dict): Node degree mapping.
        _betw (dict): Path-betweenness counts used as a simple centrality signal.

    Example:
        selector = BasisSelector(mst_adj)
        basis = selector.select_max_spread(k=10, weights=weight_map, alpha=0.5)
    """

    def __init__(
        self,
        mst_adj: Dict[Any, List[Tuple[Any, float]]],
        nodes: Optional[Sequence[Any]] = None,
    ) -> None:
        """Initialize the selector with a tree adjacency mapping.

        Args:
            mst_adj (dict): Adjacency mapping for the tree where keys are
                node labels and values are lists of ``(neighbor, weight)``
                pairs.
            nodes (Sequence, optional): Explicit ordering of nodes. If
                ``None``, the ordering is inferred from the adjacency mapping.
        """
        self.adj = mst_adj
        self.nodes = (
            list(nodes) if nodes is not None else TreeUtils.nodes_from_adj(self.adj)
        )
        self.n = len(self.nodes)
        self._dist = TreeUtils.all_pairs_tree_distance(self.adj, self.nodes)

        self._deg = TreeUtils.degrees(self.adj)
        self._betw = TreeUtils.path_betweenness_counts(self.adj)

    def _normalize_dict(
        self, values: Dict[Any, float], fallback: float = 0.0
    ) -> Dict[Any, float]:
        """Normalize a mapping of node -> numeric value into [0, 1].

        Args:
            values (dict): Mapping of node -> numeric value. Missing nodes are
                treated as ``fallback``.
            fallback (float): Value to use when ``values`` is empty.

        Returns:
            Dict: Normalized mapping for every node in ``self.nodes`` where
            values are scaled into the range ``[0, 1]``.
        """
        nodes = self.nodes
        if not values:
            return {u: fallback for u in nodes}
        vals = [values.get(u, fallback) for u in nodes]
        mn, mx = min(vals), max(vals)
        rng = mx - mn if mx > mn else 1.0
        return {u: (values.get(u, fallback) - mn) / rng for u in nodes}

    def _avg_dist(self) -> Dict[Any, float]:
        """Compute the mean path distance from each node to all others.

        Returns:
            Dict[Any, float]: Mapping from node -> average distance to every
                other node in ``self.nodes``.
        """
        avg: Dict[Any, float] = {}
        for u in self.nodes:
            dists = [self._dist_uv(u, v) for v in self.nodes]
            avg[u] = np.mean(dists)
        return avg

    def _dist_uv(self, u: Any, v: Any) -> float:
        """Return path length between two nodes in the tree.

        Args:
            u (Any): Node label for the first node.
            v (Any): Node label for the second node.

        Returns:
            float: Path length (distance) between ``u`` and ``v`` as stored
                in the precomputed ``_dist`` table.
        """
        return self._dist[u][v]

    def select_max_spread(
        self,
        k: int,
        weights: Dict[Any, float],
        alpha: float = 0.5,
        stickiness: float = 0.0,
        prev_basis: Optional[Sequence[Any]] = None,
    ) -> List[Any]:
        """Select basis nodes using a weighted max-spread greedy algorithm.

        The algorithm greedily adds nodes that maximize a combined score
        composed of the node's distance to the nearest already-selected
        basis (spread) and a normalized per-node importance weight. An
        optional ``stickiness`` parameter can bias selection toward items in
        ``prev_basis``.

        Args:
            k (int): Number of basis nodes to select.
            weights (Dict[Any, float]): Mapping from node -> importance
                weight (e.g. market cap). Missing nodes are treated as zero.
            alpha (float): Trade-off parameter between spread and weight in
                the interval ``[0, 1]``. ``alpha=1.0`` uses pure spread.
            stickiness (float): Value in ``[0, 1]`` controlling preference
                for nodes appearing in ``prev_basis``.
            prev_basis (Sequence, optional): Previously-selected basis nodes
                that may be pre-seeded and receive a small bonus.

        Returns:
            List[Any]: Ordered list of selected basis node labels (length <= k).
        """
        # This algo is modified from
        # T. F. Gonzalez, "Clustering to minimize the maximum intercluster distance", Theoretical Computer Science, 1985.
        if k <= 0 or self.n == 0:
            return []
        if k >= self.n:
            return list(self.nodes)

        # Normalize provided node weights into [0,1] to make them
        # comparable with distance-derived scores.
        normalized_weights = self._normalize_dict(weights or {})

        # Precompute the average distance from each node to all others; this
        # term acts as a proxy for how "far out" a node is in the tree.
        avg_dist = self._avg_dist()

        # Initial per-node score combines spread (avg distance) and the
        # normalized weight using parameter `alpha`.
        init_score: Dict[Any, float] = {}
        for u in self.nodes:
            init_score[u] = alpha * avg_dist[u] + (1 - alpha) * normalized_weights[u]

        # Prepare previous-basis pre-seed and stickiness bonus.
        # `prev_basis` entries are kept in order, filtered to the current
        # node set and deduplicated. We limit the pre-seed to at most `k`.
        seen = set()
        prev_list = []
        if prev_basis:
            for p in prev_basis:
                if p in self.nodes and p not in seen:
                    prev_list.append(p)
                    seen.add(p)
                    if len(prev_list) >= k:
                        break

        # Stickiness: if requested, compute a bonus that will be added to
        # candidate scores for nodes that appear in `prev_basis`. The bonus
        # is scaled by the maximum initial score so it is on a comparable scale.
        if stickiness and prev_list:
            scr_vals = list(init_score.values())
            max_scr = max(scr_vals) if scr_vals else 1.0
            bonus = stickiness * max_scr
        else:
            bonus = 0.0

        if stickiness and prev_list:
            for key in init_score.keys():
                if key in prev_list:
                    init_score[key] += bonus

        # Start building the chosen-basis list B. Pick
        # a single seed as the node with the highest initial score.
        b1 = TreeUtils.argmax_dict(init_score)
        B = [b1]

        # nearest_dist[v] stores distance from v to the nearest chosen basis
        # node (so far). It's used to compute the spread component of scores.
        nearest_dist = {v: self._dist_uv(v, B[0]) for v in self.nodes}

        # Greedily add nodes until we have k basis nodes. On each iteration
        # compute candidate scores for unchosen nodes using the nearest
        # distance term and the normalized weight, then pick the argmax.
        while len(B) < k:
            candidate_scores: Dict[Any, float] = {}
            for v in self.nodes:
                if v in B:
                    continue
                # score mixes distance-to-nearest-basis (spread) and node
                # importance (weight). `alpha` controls the trade-off.
                base = alpha * nearest_dist[v] + (1 - alpha) * normalized_weights[v]
                # apply stickiness bonus for nodes that were in prev_basis
                if bonus and prev_basis and v in prev_basis:
                    base += bonus
                candidate_scores[v] = base

            # If there are no candidates remaining, break early.
            if not candidate_scores:
                break

            # Choose the candidate with maximum score.
            v_star = TreeUtils.argmax_dict(candidate_scores)
            if v_star is None:
                break

            # Add chosen node and update nearest distances for all nodes.
            B.append(v_star)
            for v in self.nodes:
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
        stickiness: float = 0.0,
        prev_basis: Optional[Sequence[Any]] = None,
    ) -> List[Any]:
        """Select basis nodes using a hub-and-branch decomposition.

        The method first ranks nodes by a mixed centrality signal computed as
        ``alpha * degree + (1-alpha) * betweenness`` and selects the top ``h``
        nodes as hubs. It then removes hubs from the tree, finds connected
        components (branches) of the remaining graph, and picks one
        representative per component prioritized by a combination of
        distance-to-nearest-hub and normalized node weight. If fewer than
        ``k`` nodes are selected after this process, a farthest-first greedy
        fill is used to reach ``k`` nodes.

        Args:
            k (int): Total number of basis nodes to select.
            h (int, optional): Number of hubs to pick. If ``None``, a
                heuristic ``max(1, min(k-1, round(0.2*k)))`` is used.
            alpha (float): Trade-off between degree and betweenness for the
                hub-centrality measure.
            weights (dict, optional): External per-node importance signals.
            weight_gamma (float): Mixing parameter between centrality and
                external ``weights`` when ranking hubs.
            rep_alpha (float): Trade-off used when choosing a representative
                from a branch (distance-to-hub vs normalized node weight).
            stickiness (float): Bonus applied to previously-selected nodes
                appearing in ``prev_basis``.
            prev_basis (Sequence, optional): Previously-selected basis nodes
                used to bias selection via ``stickiness``.

        Returns:
            List[Any]: Ordered list of selected basis node labels (length <= k).
        """
        # Centrality and betweeness metrics are simplified from
        # L. C. Freeman, "A set of measures of centrality based on betweenness", Sociometry, 1977.
        if k <= 0 or self.n == 0:
            return []
        if k >= self.n:
            return list(self.nodes)
        if h is None:
            h = max(1, min(k - 1, int(round(0.2 * k))))

        # Compute a simple centrality measure per node mixing degree and
        # path-betweenness. Central hubs tend to have higher degree or lie on
        # many shortest paths; `alpha` controls the trade-off between degree
        # and betweenness for the centrality signal.
        centrality: Dict[Any, float] = {}
        for u in self.nodes:
            centrality[u] = alpha * self._deg.get(u, 0) + (
                1.0 - alpha
            ) * self._betw.get(u, 0)

        # Normalize centrality and optional external weights into [0,1] so
        # they are comparable. `weight_gamma` then mixes these normalized
        # signals to produce a final hub score used to rank candidate hubs.
        norm_centrality = self._normalize_dict(centrality)
        norm_weights = self._normalize_dict(weights or {})

        score: Dict[Any, float] = {}
        for u in self.nodes:
            score[u] = (1.0 - weight_gamma) * norm_centrality[
                u
            ] + weight_gamma * norm_weights[u]

        # Prepare previous-basis pre-seed and stickiness bonus.
        seen = set()
        prev_list = []
        if prev_basis:
            for p in prev_basis:
                if p in self.nodes and p not in seen:
                    prev_list.append(p)
                    seen.add(p)
                    if len(prev_list) >= k:
                        break

        # Stickiness: if requested, compute a bonus that will be added to
        # candidate scores for nodes that appear in `prev_basis`. The bonus
        # is scaled by the maximum initial score so it is on a comparable scale.
        if stickiness and prev_list:
            scr_vals = list(score.values())
            max_scr = max(scr_vals) if scr_vals else 1.0
            bonus = stickiness * max_scr
        else:
            bonus = 0.0

        if stickiness and prev_list:
            for k in score.keys():
                if k in prev_list:
                    score[k] += bonus

        # Pick the top-h nodes by the mixed score as hubs.
        hubs = sorted(self.nodes, key=lambda u: score[u], reverse=True)[:h]

        # Add the chosen hubs to the basis. Stop if we've reached k.
        B = []
        for hub in hubs:
            if len(B) >= k:
                break
            if hub not in B:
                B.append(hub)

        # Remove the hub nodes from the tree and compute connected components
        # of the remaining graph. Each component represents a "branch" that
        # is served by a nearby hub; we will pick one representative per
        # component prioritized by a combined distance-to-hub vs. node-weight
        # score.
        comps = TreeUtils.remove_nodes_and_components(self.adj, set(hubs))

        def rep_for_comp(comp: List[Any]) -> Optional[Tuple[List[Any], Any, float]]:
            # For a given component (list of nodes), choose the single best
            # representative `v` by maximizing `rep_score` which mixes the
            # path distance to the nearest hub and the (normalized) node
            # weight. `rep_alpha` controls the trade-off.
            best = None
            best_score = -1.0
            for v in comp:
                # distance to nearest hub (smaller => closer)
                d = min(self._dist_uv(v, h) for h in hubs)
                w = norm_weights.get(v, 0.0)
                rep_score = rep_alpha * d + (1.0 - rep_alpha) * w
                # apply stickiness bonus if requested to prefer previous basis
                if stickiness and prev_basis and v in prev_basis:
                    rep_score += stickiness
                if rep_score > best_score:
                    best_score = rep_score
                    best = v
            return (comp, best, best_score) if best is not None else None

        # Evaluate representatives for each component, sort by their score
        # (descending), and add them to the basis until we've selected k
        # nodes in total.
        reps = [r for r in (rep_for_comp(c) for c in comps) if r is not None]
        reps.sort(key=lambda x: x[2], reverse=True)

        for _, rep, _ in reps:
            if len(B) >= k:
                break
            if rep not in B:
                B.append(rep)

        if len(B) < k:
            # Use heap-based farthest-first greedy fill.
            self._greedy_farthest_fill(B, k)

        return B

    def _greedy_farthest_fill(self, B: List[Any], k: int) -> None:
        """Extend basis list ``B`` up to size ``k`` using a heap-optimized farthest-first rule.

        This method mutates ``B`` in-place. The algorithm:

        1. Computes the initial nearest distance from each remaining node to the
           currently selected basis in ``B``.
        2. Pushes ``(-distance, node)`` pairs into a max-heap (implemented via
           Python's min-heap with negated distances).
        3. Pops the heap top; if the popped distance is stale, re-push the
           updated value; otherwise select the node and update remaining
           distances.

        Args:
            B (List[Any]): Current basis list to extend.
            k (int): Target number of basis nodes.

        Returns:
            None: ``B`` is modified in-place.
        """
        if not B:
            return

        remaining = [v for v in self.nodes if v not in B]
        # nearest distance to current basis for each remaining node
        nearest = {v: min(self._dist_uv(v, b) for b in B) for v in remaining}

        # max-heap implemented as negative distances
        heap = [(-d, v) for v, d in nearest.items()]
        heapq.heapify(heap)

        while len(B) < k and heap:
            negd, v = heapq.heappop(heap)
            d = -negd
            # If this node already selected (removed from nearest), skip
            if v not in nearest:
                continue
            # If the popped distance is stale, push updated value
            if nearest[v] != d:
                heapq.heappush(heap, (-nearest[v], v))
                # Reselect since we have a more up-to-date distance
                continue

            # Select v
            B.append(v)
            # remove it from consideration
            del nearest[v]

            # Update nearest distances for remaining nodes
            for u in list(nearest.keys()):
                du = self._dist_uv(u, v)
                if du < nearest[u]:
                    nearest[u] = du
                    heapq.heappush(heap, (-du, u))


class LocalRidgeRunner:
    """Local neighbor-based Ridge regression pipeline.

    The runner fits one Ridge model per non-basis asset using a local set of
    basis neighbors derived from a static distance matrix. It produces:

    - ``coefficients_ridge.csv``: long-form coefficients (asset, basis, coef)
    - ``recon_returns.csv``: reconstructed return series for assets
    - ``regression_errors.csv``: per-asset RMSE and observation counts

    Args:
        prices_path (str | Path): Path to the price CSV used to compute
            returns and distances.
        basis_path (str | Path): Path to a one-column CSV listing basis
            tickers.

    Example:
        runner = LocalRidgeRunner(prices_path="synthetic_data/prices_monthly.csv",
                                  basis_path="outputs/basis_selected.csv")
        coef_df, recon_df, errors_df = runner.run()
    """

    def __init__(
        self,
        prices_path: Union[str, Path],
        basis_path: Union[str, Path],
        ridge_alpha: float = 1.0,
        q_neighbors: Optional[int] = None,
        min_periods: int = 1,
        corr_method: str = "pearson",
        shrink_method: Optional[str] = None,
        pca_n_components: Optional[int] = None,
        pca_explained_variance: Optional[float] = None,
        out_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        """Initialize the runner with explicit parameters (no config support).

        Args:
            prices_path (str | Path): Path to the CSV of price series.
            basis_path (str | Path): Path to a one-column CSV listing basis
                tickers.
            ridge_alpha (float): Regularization strength for Ridge.
            q_neighbors (int, optional): Maximum number of basis neighbors to
                use per asset. If ``None``, all basis neighbors are used.
            min_periods (int): Minimum overlapping observations when
                computing correlations/distances.
            corr_method (str): Correlation method used for distances
                (``'pearson'`` or ``'spearman'``).
            shrink_method (str, optional): Denoising method (e.g. ``'lw'``
                or ``'pca'``) applied when computing distance matrices.
            pca_n_components (int, optional): Number of PCA components if
                PCA denoising is used.
            pca_explained_variance (float, optional): Explained variance
                threshold for PCA denoising.
            out_dir (str | Path, optional): Directory to write CSV outputs.
        """
        self.prices_path = Path(prices_path)
        self.basis_path = Path(basis_path)
        self.ridge_alpha = ridge_alpha
        self.q_neighbors = q_neighbors

        self.min_periods = min_periods
        self.corr_method = corr_method
        self.shrink_method = shrink_method
        self.pca_n_components = pca_n_components
        self.pca_explained_variance = pca_explained_variance
        self.out_dir = Path(out_dir) if out_dir is not None else _ROOT / "outputs"

    def _compute_returns(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Convert price series to returns and drop empty rows.

        Args:
            prices (pd.DataFrame): Price DataFrame indexed by date with
                asset columns.

        Returns:
            pd.DataFrame: Returns DataFrame with the same columns as
                ``prices`` and rows with all-NaN dropped.
        """
        return DistancesUtils.price_to_return_df(prices).dropna(how="all")

    def _compute_distance_df(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Compute a full asset-to-asset distance DataFrame from prices.

        This wraps :class:`DistancesUtils.price_to_distance_df` and forwards
        the runner's configuration parameters.

        Args:
            prices (pd.DataFrame): Price DataFrame indexed by date.

        Returns:
            pd.DataFrame: Pairwise distance matrix (index and columns are
                asset labels).
        """
        return DistancesUtils.price_to_distance_df(
            prices,
            min_periods=self.min_periods,
            corr_method=self.corr_method,
            shrink_method=self.shrink_method,
            pca_n_components=self.pca_n_components,
            pca_explained_variance=self.pca_explained_variance,
        )

    def _filter_basis(self, basis_list: List[str], returns: pd.DataFrame) -> List[str]:
        """Filter a basis list to the subset present in the returns data.

        Args:
            basis_list (List[str]): Candidate basis tickers.
            returns (pd.DataFrame): Returns DataFrame whose columns define the
                available universe.

        Returns:
            List[str]: Filtered basis list containing only tickers present in
                ``returns``.

        Raises:
            ValueError: If no basis tickers are found in ``returns``.
        """
        filtered_basis = [b for b in basis_list if b in returns.columns]
        if not filtered_basis:
            raise ValueError("No basis tickers found in returns data.")
        return filtered_basis

    def _select_neighbors_for_asset(
        self,
        asset: str,
        basis_list: List[str],
        dist_df: pd.DataFrame,
        q_neighbors: Optional[int],
    ) -> List[str]:
        """Select nearest basis neighbors for a given asset.

        Args:
            asset (str): Asset label for which to select neighbors.
            basis_list (List[str]): Ordered candidate basis assets.
            dist_df (pd.DataFrame): Square pairwise distance matrix.
            q_neighbors (int, optional): Maximum number of neighbors to
                return. If ``None``, all neighbors are returned sorted by
                increasing distance.

        Returns:
            List[str]: Selected neighbor basis labels (may be empty).
        """
        if asset not in dist_df.index:
            return []
        dists = dist_df.loc[asset, basis_list].dropna()
        if dists.empty:
            return []
        dists = dists.sort_values()
        if q_neighbors is None:
            return dists.index.tolist()
        return dists.index.tolist()[:q_neighbors]

    def run_all_regressions(
        self,
        returns: pd.DataFrame,
        basis_list: List[str],
        prices: pd.DataFrame,
        ridge_alpha: float,
        q_neighbors: Optional[int],
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Run Ridge regressions over the full dataset with a static basis list.

        A single Ridge model is fit per non-basis asset using all available
        overlapping observations. The function returns a long-form
        coefficients DataFrame, reconstructed returns, and per-asset error
        metrics.

        Args:
            returns (pd.DataFrame): Returns DataFrame indexed by date.
            basis_list (List[str]): Ordered list of basis tickers used as
                regressors.
            prices (pd.DataFrame): Price DataFrame used to derive distances
                for neighbor selection.
            ridge_alpha (float): Regularization strength for Ridge.
            q_neighbors (int, optional): Max number of neighbors to use per
                asset; ``None`` means use all basis neighbors.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
                - coef_long_df: long-form DataFrame with columns
                  ``[asset, basis, coef]`` for nonzero/regressed coefficients.
                - recon_df: Reconstructed returns for assets (same index as
                    ``returns``).
                - errors_df: Per-asset error summary (RMSE and n_obs).

        Notes:
            The Ridge models are fit without an intercept term (``fit_intercept=False``)
            so regressions are performed through the origin to match the project's
            convention for local linear combinations.
        """
        # Prepare outputs
        recon_df = pd.DataFrame(
            index=returns.index, columns=returns.columns, dtype=float
        )
        coef_records: List[dict] = []

        # compute a single static distance matrix for the whole dataset
        dist_df = self._compute_distance_df(prices)

        # run a single regression per non-basis asset using all available observations
        for asset in returns.columns:
            if asset in basis_list:
                continue
            neighbors = self._select_neighbors_for_asset(
                asset, basis_list, dist_df, q_neighbors
            )
            if not neighbors:
                continue

            cols = [asset] + neighbors
            train = returns[cols].dropna()
            if train.shape[0] == 0:
                continue

            y = train[asset].values
            X = train[neighbors].values
            model = Ridge(alpha=ridge_alpha, fit_intercept=False)
            model.fit(X, y)

            for pos, b in enumerate(neighbors):
                coef_records.append(
                    {"asset": asset, "basis": b, "coef": float(model.coef_[pos])}
                )

            # predict for all timestamps where neighbor values are present
            X_all = returns[neighbors].dropna()
            if not X_all.empty:
                preds = model.predict(X_all.values)
                for idx, val in zip(X_all.index, preds):
                    recon_df.at[idx, asset] = float(val)

        coef_long_df = (
            pd.DataFrame(coef_records)
            if coef_records
            else pd.DataFrame(columns=["asset", "basis", "coef"])
        )

        errors = []
        for asset in returns.columns:
            preds = recon_df[asset].dropna()
            if preds.empty:
                continue
            true_vals = returns.loc[preds.index, asset]
            rmse = np.sqrt(np.mean((true_vals.values - preds.values) ** 2))
            errors.append({"asset": asset, "rmse": rmse, "n_obs": len(preds)})

        errors_df = (
            pd.DataFrame(errors).set_index("asset").sort_values("rmse", ascending=False)
            if errors
            else pd.DataFrame(columns=["rmse", "n_obs"])
        )

        return coef_long_df, recon_df, errors_df

    def _save_outputs(
        self,
        coef_df: pd.DataFrame,
        recon_df: pd.DataFrame,
        errors_df: pd.DataFrame,
        out_dir: Path,
    ) -> None:
        """Persist regression outputs to CSV files under ``out_dir``.

        Args:
            coef_df (pd.DataFrame): Long-form coefficients DataFrame.
            recon_df (pd.DataFrame): Reconstructed returns DataFrame.
            errors_df (pd.DataFrame): Per-asset error summary DataFrame.
            out_dir (Path): Directory to save CSV outputs into.
        """
        out_dir.mkdir(parents=True, exist_ok=True)
        coef_df.to_csv(out_dir / "coefficients_ridge.csv", index=False)
        recon_df.to_csv(out_dir / "recon_returns.csv")
        errors_df.to_csv(out_dir / "regression_errors.csv")

    def run(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Execute the regression pipeline and write outputs.

        Returns:
            Tuple of (coef_df, recon_df, errors_df) as produced by
            :py:meth:`_run_all_regressions`.
        """
        prices = load_prices_csv(self.prices_path)
        returns = self._compute_returns(prices)

        # Load a basis list from configured basis path.
        basis_list = load_basis_list_from_path(self.basis_path)
        basis_list = self._filter_basis(basis_list, returns)

        coef_df, recon_df, errors_df = self.run_all_regressions(
            returns=returns,
            basis_list=basis_list,
            prices=prices,
            ridge_alpha=self.ridge_alpha,
            q_neighbors=self.q_neighbors,
        )

        self._save_outputs(coef_df, recon_df, errors_df, self.out_dir)
        return coef_df, recon_df, errors_df


class WeightMapper:
    """Map index weights to basis weights using coefficient pivots.

    The mapper builds an asset-by-basis mapping matrix from a coefficient
    pivot (asset rows, basis columns) and applies it to index weights to
    produce basis-level weights. An optional turnover-penalized quadratic
    program can be used to trade off fit versus deviation from previous
    basis weights.

    Attributes:
        basis_path (Path): Path to the basis list CSV.
        coeffs_ts_path (Path): Path to the coefficient pivot CSV.
        weights_path (Path | None): Optional path to single-row index weights.
        out_dir (Path): Output directory for saved basis weights.
        turnover_lambda (float): Turnover penalty parameter for the QP.
        prev_basis_weights (pd.Series | None): Warm-start previous basis
            weights used by the QP solver.
    """

    def __init__(
        self,
        basis_path: Union[str, Path],
        coeffs_ts_path: Union[str, Path],
        weights_path: Optional[Union[str, Path]] = None,
        out_dir: Optional[Union[str, Path]] = None,
        weights_df: Optional[pd.DataFrame] = None,
        turnover_lambda: float = 0.0,
        prev_basis_weights: Optional[pd.Series] = None,
    ) -> None:
        """Initialize the mapper with explicit parameters.

        Args:
            basis_path: Path to basis list CSV (one-column list).
            coeffs_ts_path: Path to coefficient pivot CSV (asset x basis).
            weights_path: Optional path to single-row weights CSV. If not
                provided and `weights_df` is None, the mapper will error.
            out_dir: Directory to save outputs; defaults to repo `outputs/`.
            weights_df: Optional in-memory single-row weights DataFrame.
            turnover_lambda: Turnover penalty parameter for QP solver. If > 0,
                uses turnover-penalized optimization.
            prev_basis_weights: Optional previous basis weights Series for
                warm-starting QP solver.
        """
        self.basis_path = Path(basis_path)
        self.coeffs_ts_path = Path(coeffs_ts_path)
        self.weights_path = Path(weights_path) if weights_path is not None else None
        self.out_dir = Path(out_dir) if out_dir is not None else _ROOT / "outputs"
        # Optional in-memory single-row weights DataFrame. If provided,
        # `run()` will use this instead of reading a weights CSV from disk.
        self._weights_df = weights_df
        # Turnover penalty parameter (lambda).
        try:
            self.turnover_lambda = float(turnover_lambda)
        except Exception:
            self.turnover_lambda = 0.0
        # Optional previous basis weights (pd.Series indexed by basis labels).
        self.prev_basis_weights = prev_basis_weights

    def _compute_from_coeffs(
        self, coeffs_pivot: pd.DataFrame, w_spx_row: pd.Series, basis_list: List[str]
    ) -> pd.Series:
        """Compute basis weights from a coefficient pivot and a single index weight row.

        The method constructs an asset-by-basis matrix ``A_date`` aligned to
        the provided universe, optionally inserts identity rows for basis
        assets that appear in the universe, and either applies an analytic
        mapping ``w_row @ A_date`` or solves a turnover-penalized QP when
        ``turnover_lambda`` is set.

        Args:
            coeffs_pivot (pd.DataFrame): Coefficient pivot with asset index
                and basis columns.
            w_spx_row (pd.Series): Single-row index weight vector (indexed by
                asset labels) for a given date.
            basis_list (List[str]): Ordered list of basis labels corresponding
                to pivot columns.

        Returns:
            pd.Series: Basis weights indexed by basis labels.
        """
        all_bases = list(basis_list)
        universe = list(w_spx_row.index)

        A_date = pd.DataFrame(0.0, index=universe, columns=all_bases, dtype=float)

        # If a basis asset appears in the universe, set identity entry so that
        # basis assets can represent themselves even if missing in coeffs.
        for b in all_bases:
            if b in A_date.index and b in A_date.columns:
                A_date.at[b, b] = 1.0

        # Fill A_date from provided coefficient pivot (aligning indices/columns)
        for asset in coeffs_pivot.index.astype(str):
            if asset not in A_date.index:
                continue
            for b in coeffs_pivot.columns:
                if b in A_date.columns:
                    A_date.at[asset, b] = float(coeffs_pivot.at[asset, b])

        w_row = w_spx_row.values.reshape(1, -1)
        # If a turnover penalty was requested, solve the constrained QP to
        # obtain basis weights that trade off fit vs turnover
        if self.turnover_lambda and self.turnover_lambda > 0.0:
            # Prepare previous weights vector aligned to columns
            if self.prev_basis_weights is not None:
                w_prev_series = self.prev_basis_weights
            else:
                # try to load previous weights from disk if available
                prev_path = self.out_dir / "basis_weights.csv"
                if prev_path.exists():
                    try:
                        prev_df = pd.read_csv(prev_path, index_col=0)
                        # take the last row if multiple exist
                        w_prev_series = prev_df.iloc[-1]
                    except Exception:
                        w_prev_series = pd.Series(dtype=float)
                else:
                    w_prev_series = pd.Series(dtype=float)

            # If we have no previous weights at all, skip the turnover penalty this iteration
            if w_prev_series.empty:
                basis_w = pd.Series((w_row @ A_date.values).flatten(), index=all_bases)
                return basis_w

            # align w_prev to the current basis columns
            w_prev_arr = np.zeros(len(all_bases), dtype=float)
            for i, col in enumerate(all_bases):
                if col in w_prev_series.index:
                    try:
                        w_prev_arr[i] = float(w_prev_series.at[col])
                    except Exception:
                        w_prev_arr[i] = 0.0

            # solver expects numpy arrays
            b_vec = w_spx_row.values.astype(float)

            try:
                basis_w_series = self.solve_qp_basis_weights(
                    A_date, b_vec, all_bases, w_prev_arr, float(self.turnover_lambda)
                )
                return basis_w_series
            except Exception:
                # fallback to analytic mapping if solver fails
                basis_w = pd.Series((w_row @ A_date.values).flatten(), index=all_bases)
                return basis_w

        # default analytic mapping (no turnover control)
        basis_w = pd.Series((w_row @ A_date.values).flatten(), index=all_bases)
        return basis_w

    def solve_qp_basis_weights(
        self,
        A_date: pd.DataFrame,
        b_vec: np.ndarray,
        basis_cols: List[str],
        w_prev_arr: np.ndarray,
        lam: float,
    ) -> pd.Series:
        """Solve a small quadratic program for basis weights.

        Minimizes the objective::

            ||A w - b||^2 + lam * ||w - w_prev||^2

        The implementation uses SciPy's SLSQP via :func:`scipy.optimize.minimize`
        and returns a solution vector aligned to ``basis_cols``. The solver
        intentionally does not enforce a sum-to-one equality constraint so
        that any unmapped mass is left as cash.

        Args:
            A_date (pd.DataFrame): Asset-by-basis mapping matrix for the
                current date.
            b_vec (np.ndarray): Index weight vector (asset-ordered).
            basis_cols (List[str]): Ordered basis column labels.
            w_prev_arr (np.ndarray): Previous basis weights used as a warm
                start / penalty reference.
            lam (float): Turnover penalty parameter.

        Returns:
            pd.Series: Solved basis weights indexed by ``basis_cols``.
        """
        A = A_date.values.astype(float)
        b = b_vec.astype(float)
        M = A.shape[1]

        # Ensure w_prev_arr has the right length
        w_prev = np.asarray(w_prev_arr, dtype=float).reshape(-1)
        if w_prev.shape[0] != M:
            w_prev = np.zeros(M, dtype=float)

        # Objective function: computes the scalar cost combining fit error and turnover penalty.
        # The fit error term ||Aw - b||^2 ensures basis weights map back to index weights.
        # The turnover penalty lam * ||w - w_prev||^2 discourages large changes from previous weights.
        def obj(w: np.ndarray) -> float:
            r = A.dot(w) - b
            return float(r.dot(r) + lam * np.dot(w - w_prev, w - w_prev))

        # Jacobian (gradient) of the objective with respect to w.
        # Derived by differentiating: d/dw [||Aw - b||^2 + lam * ||w - w_prev||^2]
        # = 2 * A^T * (Aw - b) + 2 * lam * (w - w_prev)
        def jac(w: np.ndarray) -> np.ndarray:
            Aw_b = A.dot(w) - b
            return 2.0 * (A.T.dot(Aw_b) + lam * (w - w_prev))

        # No bound constraints; allow negative weights if the solver prefers them.
        bounds = None
        cons = ()

        # Initial guess: reuse previous weights if valid; otherwise start uniform
        x0 = np.array(w_prev, dtype=float)
        s = x0.sum()
        if s <= 0 or not np.isfinite(s):
            x0 = np.ones(M, dtype=float) / float(M)

        res = minimize(
            obj,
            x0,
            jac=jac,
            bounds=bounds,
            constraints=cons,
            method="SLSQP",
            options={"ftol": 1e-9, "maxiter": 200},
        )

        w_opt = res.x

        return pd.Series(w_opt, index=basis_cols)

    def _normalize_and_save_single_row(
        self, basis_weights: pd.DataFrame, out_dir: Path
    ) -> None:
        """Persist computed basis weights.

        The implementation writes the basis weights as-computed
        (no forced renormalization) to ``out_dir/basis_weights.csv``.

        Args:
            basis_weights (pd.DataFrame): DataFrame of basis weights
                (index = date, columns = basis labels).
            out_dir (Path): Directory where output CSV files will be saved.
        """
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "basis_weights.csv"
        basis_weights.to_csv(out_path, index=True)
        print(f"Saved basis weights to: {out_path}")

    def run(self) -> pd.DataFrame:
        """Compute and save basis weights according to configuration.

        The method loads a coefficient pivot, loads index weights (from
        provided path or in-memory DataFrame), computes basis weights for
        the first row, saves outputs and returns the final basis weights DataFrame.

        Returns:
            DataFrame: Final basis weights with a single row indexed by the date
                from the input weights.
        """
        # Load basis list and coefficient pivot from provided paths
        basis_list = load_basis_list_from_path(self.basis_path)

        coeffs_pivot = pd.read_csv(self.coeffs_ts_path, index_col=0)

        if self._weights_df is not None:
            w_spx = self._weights_df
        else:
            w_spx = pd.read_csv(self.weights_path, index_col=0)

        w_spx_row = w_spx.iloc[0]

        basis_weights_series = self._compute_from_coeffs(
            coeffs_pivot, w_spx_row, basis_list
        )

        # Wrap into DataFrame and save normalized output
        basis_weights = pd.DataFrame([basis_weights_series])
        basis_weights.index = [w_spx.index[0]]

        self._normalize_and_save_single_row(basis_weights, self.out_dir)

        return basis_weights
