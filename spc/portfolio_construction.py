from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Sequence, Any
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from spc.graph import DistancesUtils, TreeUtils
from spc.utils import (
    load_basis_list_from_path,
    resolve_output_and_input_paths,
    load_prices_csv,
    cfg_val,
)


class BasisSelector:
    """Select representative basis nodes from a tree (minimum spanning tree).

    The selector extracts a small set of representative "basis" nodes from a
    tree topology (typically an MST over asset distances). It provides two
    primary selection strategies:
        - ``select_max_spread_weighted``: iteratively picks nodes that maximize a
            combination of tree spread and per-node importance weights.
        - ``select_hub_branch``: selects central hubs (by degree/betweenness)
            then selects representatives from the remaining branches.

    Attributes:
            adj (dict): Adjacency dictionary for the tree (node -> list[(nbr, weight)]).
            nodes (list): Ordered list of nodes used by the selector.
            n (int): Number of nodes in ``nodes``.
            _dist (dict|None): Optional all-pairs tree distance table (node->node->dist).
            _deg (dict): Node degrees computed from adjacency.
            _betw (dict): Simple path-betweenness counts per node.

    Example:
            selector = BasisSelector(mst_adj)
            basis = selector.select_max_spread_weighted(k=10, weights=weight_map, alpha=0.5)
    """

    def __init__(
        self,
        mst_adj: Dict[Any, List[Tuple[Any, float]]],
        nodes: Optional[Sequence[Any]] = None,
    ):
        """Initialize the selector with a tree adjacency mapping.

        Args:
            mst_adj: Adjacency mapping for the tree where keys are node labels
                and values are lists of (neighbor, weight) pairs.
            nodes: Optional explicit ordering of nodes to use. If ``None``, the
                node ordering is inferred from the adjacency mapping.
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
        self, values: Dict[Any, float], fallback=0.0
    ) -> Dict[Any, float]:
        """Normalize a mapping of node -> numeric value into [0, 1].

        Args:
            values (dict): Mapping node -> numeric value. Missing nodes are
                treated as 0.0.
            fallback (float): Value to use when ``values`` is empty.

        Returns:
            dict: Normalized mapping for every node in ``self.nodes``.
        """
        nodes = self.nodes
        if not values:
            return {u: float(fallback) for u in nodes}
        vals = [float(values.get(u, 0.0)) for u in nodes]
        mn, mx = min(vals), max(vals)
        rng = mx - mn if mx > mn else 1.0
        return {u: (float(values.get(u, 0.0)) - mn) / rng for u in nodes}

    def _avg_dist(self) -> Dict[Any, float]:
        """Return average tree distance from each node to all others.

        Returns:
            dict: node -> mean distance to all nodes in ``self.nodes``.
        """
        return {
            u: np.mean([self._dist_uv(u, v) for v in self.nodes]) for u in self.nodes
        }

    def _dist_uv(self, u: Any, v: Any) -> float:
        """Return path length between nodes ``u`` and ``v``.

        Uses the precomputed all-pairs distance table if available, otherwise
        falls back to computing via :func:`TreeUtils.path_length_between`.
        """
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
            list: Selected basis node labels (length <= k).
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
        b1 = max(init_score, key=init_score.get)
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
            v_star = max(candidate_scores, key=candidate_scores.get)
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

        If ``h`` is ``None``, we set ``h = max(1, min(k-1, round(0.2 * k)))``.

        Args:
            k (int): Total number of basis nodes to select.
            h (int | None): Number of hubs to pick; if ``None`` a heuristic is used.
            alpha (float): Trade-off between degree and betweenness for centrality.
            weights (dict | None): Node importance weights that can influence hub selection.
            weight_gamma (float): Weighting parameter mixing centrality and ``weights``.
            rep_alpha (float): Trade-off when selecting a representative from a branch
                (distance-to-hub vs. node weight).

        Returns:
            list: Chosen basis node labels (length <= k).
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


class LocalRidgeRunner:
    """Local neighbor-based Ridge regression pipeline.

    This runner performs Ridge regressions for non-basis assets using a local
    neighborhood of basis assets determined from time-local distance matrices
    (to avoid look-ahead bias). Results (coefficients timeseries, coefficient
    snapshot, reconstructed returns and regression errors) are written to the
    outputs directory under the project root when `run()` is executed.

    Attributes:
        config (dict): Parsed configuration dictionary. Expected keys include
            entries under `input`, `output`, `distance_computation`,
            `pca_denoising`, and `regression` used by the runner.

    Example:
        runner = LocalRidgeRunner(config)
        coef_df, recon_df, errors_df = runner.run()
    """

    def __init__(self, config: Dict):
        """Initialize the runner with a configuration mapping.

        Args:
            config: Configuration dictionary for the runner.
        """
        self.config = config

    def _compute_returns(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Convert price series to returns and drop empty rows.

        Args:
            prices: DataFrame of prices indexed by date with columns as assets.

        Returns:
            DataFrame of returns with the same columns as `prices`.
        """
        return DistancesUtils.price_to_return_df(prices).dropna(how="all")

    def _filter_basis(self, basis_list: List[str], returns: pd.DataFrame) -> List[str]:
        """Filter the provided basis list to those present in `returns`.

        Args:
            basis_list: Sequence of basis asset labels.
            returns: Returns DataFrame whose columns define the available universe.

        Returns:
            A filtered list containing only basis tickers present in `returns`.

        Raises:
            ValueError: If no basis tickers are found in the returns data.
        """
        basis_present = [b for b in basis_list if b in returns.columns]
        if len(basis_present) != len(basis_list):
            missing = set(basis_list) - set(basis_present)
            print(f"Warning: {len(missing)} basis tickers not in price data: {missing}")
        if not basis_present:
            raise ValueError("No basis tickers found in returns data")
        return basis_present

    def _select_neighbors_for_asset(
        self,
        asset: str,
        basis_list: List[str],
        dist_df: pd.DataFrame,
        q_neighbors: Optional[int],
    ) -> List[str]:
        """Select nearest basis neighbors for `asset` using a distance matrix.

        Args:
            asset: Asset label for which to select neighbors.
            basis_list: Ordered list of candidate basis assets.
            dist_df: Square DataFrame of pairwise distances (index and columns are assets).
            q_neighbors: Optional maximum number of neighbors to return. If
                ``None``, return all available neighbors sorted by distance.

        Returns:
            List of selected neighbor basis labels (may be empty).
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

    def _run_all_regressions(
        self,
        returns: pd.DataFrame,
        basis_list: List[str],
        prices: pd.DataFrame,
        ridge_alpha: float,
        q_neighbors: Optional[int],
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Run rolling local Ridge regressions for non-basis assets.

        For each date, a time-local distance matrix is computed (or fetched from
        cache) and used to select nearby basis assets for each non-basis asset.
        A Ridge model is fit on historical returns up to that date and the
        time-series of coefficients, reconstructed returns and per-asset
        errors are returned.

        Args:
            returns: Returns DataFrame indexed by date with asset columns.
            basis_list: List of basis asset labels to consider as regressors.
            prices: Price DataFrame used when computing time-local distances.
            ridge_alpha: Regularization parameter for Ridge regression.
            q_neighbors: Optional maximum number of neighbors to use per asset.

        Returns:
            Tuple of (coef_long_df, recon_df, errors_df):
                - coef_long_df: long-form DataFrame with columns [date, asset, basis, coef]
                - recon_df: DataFrame of reconstructed returns for non-basis assets
                - errors_df: DataFrame indexed by asset with columns ['rmse','n_obs','n_basis_used']
        """
        non_basis = [c for c in returns.columns if c not in basis_list]
        if len(non_basis) == 0:
            return (
                pd.DataFrame(columns=["date", "asset", "basis", "coef"]),
                pd.DataFrame(index=returns.index),
                pd.DataFrame(columns=["rmse", "n_obs", "n_basis_used"]),
            )

        # track which neighbors were used for each asset (aggregate over time)
        neighbor_used_sets: Dict[str, set] = {asset: set() for asset in non_basis}

        coef_records: List[dict] = []
        recon_df = pd.DataFrame(index=returns.index, columns=non_basis, dtype=float)

        min_obs = 10

        # We'll compute distances in a time-local fashion to avoid look-ahead.
        dist_cache: Dict[tuple, pd.DataFrame] = {}

        # distance window and shrink settings from config
        dist_window = cfg_val(self.config, "distance_computation", "window_size", None)
        shrink_method = cfg_val(
            self.config, "distance_computation", "shrink_method", None
        )
        pca_n_components = cfg_val(
            self.config, "pca_denoising", "pca_n_components", None
        )
        pca_explained_variance = cfg_val(
            self.config, "pca_denoising", "pca_explained_variance", None
        )

        for t in returns.index:
            # build / fetch distance matrix up to time t (expanding or rolling)
            key = (
                pd.Timestamp(t),
                int(dist_window) if dist_window is not None else None,
                shrink_method,
                pca_n_components,
                pca_explained_variance,
            )
            if key in dist_cache:
                dist_t = dist_cache[key]
            else:
                dist_t = DistancesUtils.windowed_price_to_distance(
                    prices,
                    t,
                    window=dist_window,
                    min_periods=cfg_val(
                        self.config, "distance_computation", "min_periods", 1
                    ),
                    corr_method=cfg_val(
                        self.config, "distance_computation", "corr_method", "pearson"
                    ),
                    shrink_method=shrink_method,
                    pca_n_components=pca_n_components,
                    pca_explained_variance=pca_explained_variance,
                )
                dist_cache[key] = dist_t

            for asset in non_basis:
                # select neighbors using the time-local distance matrix
                neighbors = self._select_neighbors_for_asset(
                    asset, basis_list, dist_t, q_neighbors
                )
                if not neighbors:
                    continue
                neighbor_used_sets[asset].update(neighbors)
                cols = [asset] + neighbors
                train = returns.loc[:t, cols].dropna()
                if train.shape[0] < min_obs:
                    continue
                y = train[asset].values
                X = train[neighbors].values
                model = Ridge(alpha=ridge_alpha)
                model.fit(X, y)
                for b in basis_list:
                    coef_val = 0.0
                    if b in neighbors:
                        try:
                            pos = neighbors.index(b)
                            coef_val = float(model.coef_[pos])
                        except Exception:
                            coef_val = 0.0
                    coef_records.append(
                        {"date": t, "asset": asset, "basis": b, "coef": coef_val}
                    )
                x_t = returns.loc[t, neighbors]
                if x_t.isnull().any():
                    continue
                xvals = x_t.values.reshape(1, -1)
                try:
                    y_pred = float(model.predict(xvals)[0])
                    recon_df.at[t, asset] = y_pred
                except Exception:
                    continue

        coef_long_df = (
            pd.DataFrame(coef_records)
            if coef_records
            else pd.DataFrame(columns=["date", "asset", "basis", "coef"])
        )

        errors = []
        for asset in non_basis:
            preds = recon_df[asset].dropna()
            if preds.empty:
                continue
            true_vals = returns.loc[preds.index, asset]
            rmse = np.sqrt(np.mean((true_vals.values - preds.values) ** 2))
            errors.append(
                {
                    "asset": asset,
                    "rmse": float(rmse),
                    "n_obs": int(len(preds)),
                    "n_basis_used": len(neighbor_used_sets.get(asset, set())),
                }
            )

        errors_df = (
            pd.DataFrame(errors).set_index("asset").sort_values("rmse", ascending=False)
            if errors
            else pd.DataFrame(columns=["rmse", "n_obs", "n_basis_used"])
        )

        return coef_long_df, recon_df, errors_df

    def _save_outputs(
        self,
        coef_df: pd.DataFrame,
        recon_df: pd.DataFrame,
        errors_df: pd.DataFrame,
        out_dir: Path,
    ) -> None:
        """Persist outputs to CSV files under `out_dir`.

        Args:
            coef_df: Long-form coefficients timeseries DataFrame.
            recon_df: Reconstructed returns DataFrame.
            errors_df: Per-asset error summary DataFrame.
            out_dir: Directory to write CSV outputs into.
        """
        out_dir.mkdir(parents=True, exist_ok=True)
        coef_out_ts = out_dir / "coefficients_ridge_timeseries.csv"
        coef_df.to_csv(coef_out_ts, index=False)
        print(f"Saved time-series coefficients (long) to: {coef_out_ts}")
        try:
            last_date = coef_df["date"].max()
            pivot = (
                coef_df[coef_df["date"] == last_date]
                .pivot(index="asset", columns="basis", values="coef")
                .fillna(0.0)
            )
            pivot.to_csv(out_dir / "coefficients_ridge.csv")
            print(
                f"Saved latest-date coefficient snapshot to: {out_dir / 'coefficients_ridge.csv'}"
            )
        except Exception:
            pd.DataFrame().to_csv(out_dir / "coefficients_ridge.csv")
        recon_df.to_csv(out_dir / "recon_returns.csv")
        errors_df.to_csv(out_dir / "regression_errors.csv")

    def run(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Execute the regression pipeline and write outputs.

        Returns:
            Tuple of (coef_df, recon_df, errors_df) as produced by
            :py:meth:`_run_all_regressions`.
        """
        prices_path = Path(
            cfg_val(
                self.config, "input", "prices_path", "synthetic_data/prices_monthly.csv"
            )
        )
        basis_path = Path(
            cfg_val(self.config, "output", "basis_path", "outputs/basis_selected.csv")
        )
        ridge_alpha = cfg_val(self.config, "regression", "ridge_alpha", 1.0)
        q_neighbors = cfg_val(self.config, "regression", "q_neighbors", None)

        prices = load_prices_csv(prices_path)
        basis_list = load_basis_list_from_path(basis_path)
        returns = self._compute_returns(prices)

        basis_list = self._filter_basis(basis_list, returns)

        coef_df, recon_df, errors_df = self._run_all_regressions(
            returns=returns,
            basis_list=basis_list,
            prices=prices,
            ridge_alpha=ridge_alpha,
            q_neighbors=q_neighbors,
        )

        out_dir = _ROOT / "outputs"
        self._save_outputs(coef_df, recon_df, errors_df, out_dir)
        return coef_df, recon_df, errors_df


class WeightMapper:
    """Compute basis weights from index weights using Ridge coefficients.

    The mapper converts coefficient outputs (either timeseries or snapshot)
    into basis weights by constructing an asset->basis mapping matrix and
    applying it to index weights. Outputs are written under the project
    `outputs` directory.

    Attributes:
        config (dict): Parsed configuration dictionary with `input`/`output`
            locations used by :py:meth:`run`.
    """

    def __init__(self, config: Dict):
        """Initialize the mapper with configuration.

        Args:
            config: Configuration dictionary used to resolve input/output paths.
        """
        self.config = config

    def _compute_from_timeseries(
        self, coeffs_ts: pd.DataFrame, w_spx: pd.DataFrame, basis_list: list[str]
    ) -> pd.DataFrame:
        """Construct basis weight time-series from coefficient timeseries.

        The method pivots the coefficients timeseries into per-date A matrices
        (asset -> basis), aligns the universe with the index weight columns,
        normalizes asset rows, and multiplies index weights to produce basis
        weights per date.

        Args:
            coeffs_ts: Long-form DataFrame with columns ['date','asset','basis','coef'].
            w_spx: DataFrame of index weights indexed by date with asset columns.
            basis_list: Ordered list of basis asset labels.

        Returns:
            DataFrame of basis weights indexed by the dates from `w_spx`.
        """
        coeffs_ts["date"] = pd.to_datetime(coeffs_ts["date"])
        coeffs_ts = coeffs_ts.sort_values("date")
        coeffs_by_date = {}
        for d, grp in coeffs_ts.groupby("date"):
            pivot = grp.pivot(index="asset", columns="basis", values="coef").fillna(0.0)
            pivot = pivot.reindex(columns=basis_list, fill_value=0.0)
            coeffs_by_date[pd.Timestamp(d)] = pivot

        coeff_dates = sorted(coeffs_by_date.keys())

        universe = [str(c) for c in w_spx.columns]
        basis_rows = []
        for w_date in w_spx.index:
            sel_date = None
            for cd in coeff_dates:
                if cd <= pd.Timestamp(w_date):
                    sel_date = cd
                else:
                    break

            A_date = pd.DataFrame(0.0, index=universe, columns=basis_list, dtype=float)
            for b in basis_list:
                if b in A_date.index and b in A_date.columns:
                    A_date.at[b, b] = 1.0

            if sel_date is not None:
                pivot = coeffs_by_date[sel_date]
                for asset in pivot.index.astype(str):
                    if asset not in A_date.index:
                        continue
                    for b in basis_list:
                        A_date.at[asset, b] = (
                            float(pivot.at[asset, b]) if b in pivot.columns else 0.0
                        )

            row_sums_A = A_date.sum(axis=1)
            positive_rows = row_sums_A > 0.0
            if positive_rows.any():
                A_date.loc[positive_rows] = A_date.loc[positive_rows].div(
                    row_sums_A[positive_rows], axis=0
                )

            w_row = w_spx.loc[w_date].values.reshape(1, -1)
            basis_w_row = pd.Series(
                (w_row @ A_date.values).flatten(), index=basis_list, name=w_date
            )
            basis_rows.append(basis_w_row)

        basis_weights = pd.DataFrame(basis_rows)
        basis_weights.index = w_spx.index
        return basis_weights

    def _compute_from_snapshot(
        self, coeffs: pd.DataFrame, w_spx: pd.DataFrame, basis_list: list[str]
    ) -> pd.DataFrame:
        """Construct basis weights from a single coefficient snapshot.

        Args:
            coeffs: DataFrame indexed by asset with columns corresponding to basis labels (coefficients).
            w_spx: DataFrame of index weights (single snapshot or time-indexed; this method uses columns).
            basis_list: Ordered list of basis labels.

        Returns:
            DataFrame (or Series) of basis weights aligned with `w_spx` columns.
        """
        universe = [str(c) for c in w_spx.columns]
        A = pd.DataFrame(0.0, index=universe, columns=basis_list, dtype=float)
        for b in basis_list:
            if b in A.index and b in A.columns:
                A.at[b, b] = 1.0
        for asset in coeffs.index.astype(str):
            if asset not in A.index:
                continue
            for col in coeffs.columns:
                col_str = str(col)
                if col_str in A.columns:
                    A.at[asset, col_str] = float(coeffs.loc[asset, col])
        row_sums_A = A.sum(axis=1)
        positive_rows = row_sums_A != 0.0
        if positive_rows.any():
            A.loc[positive_rows] = A.loc[positive_rows].div(
                row_sums_A[positive_rows], axis=0
            )
        basis_weights = w_spx.reindex(columns=universe).fillna(0.0).dot(A)
        return basis_weights

    def _normalize_and_save_single_row(
        self, basis_weights: pd.DataFrame, out_dir: Path
    ) -> None:
        """Normalize computed basis weights and persist raw/normalized CSVs.

        This helper writes a raw CSV (`basis_weights_raw.csv`) if any rows do
        not sum to (approximately) 1, then writes a normalized CSV
        (`basis_weights.csv`) where rows with positive sums are scaled to sum=1.

        Args:
            basis_weights: DataFrame of basis weights (index = date, columns = basis labels).
            out_dir: Directory where output CSV files will be saved.
        """
        tol = 1e-6
        row_sums = basis_weights.sum(axis=1)
        is_unit = row_sums.apply(lambda x: np.isclose(x, 1.0, atol=tol))
        n_rows = len(row_sums)
        n_unit = int(is_unit.sum())
        n_zero = int((row_sums == 0.0).sum())

        if n_unit == n_rows and n_zero == 0:
            out_path = out_dir / "basis_weights.csv"
            basis_weights.to_csv(out_path, index=True)
            print(
                f"Saved basis weights to: {out_path} (all rows sum to 1 within tol={tol})"
            )
            return

        raw_out = out_dir / "basis_weights_raw.csv"
        basis_weights.to_csv(raw_out, index=True)
        print(f"Saved raw basis weights to: {raw_out}")

        normalized = basis_weights.copy()
        positive = row_sums > 0.0
        normalized.loc[positive] = normalized.loc[positive].div(
            row_sums[positive], axis=0
        )

        norm_out = out_dir / "basis_weights.csv"
        normalized.to_csv(norm_out, index=True)
        print(
            f"Saved normalized basis weights to: {norm_out} (rows with sum>0 scaled to sum=1)"
        )

    def run(self) -> pd.DataFrame:
        """Compute and save basis weights according to configuration.

        The method resolves input/output paths from the config, loads either
        a coefficient timeseries or snapshot, loads index weights, computes
        basis weights (timeseries preferred), normalizes/saves outputs and
        returns the final basis weights DataFrame.

        Returns:
            DataFrame: Final basis weights (indexed by date if timeseries).

        Raises:
            FileNotFoundError: If neither coefficients nor weights input files are found.
        """
        out_dir, basis_path, coeffs_path, coeffs_ts_path, weights_path = (
            resolve_output_and_input_paths(self.config, _ROOT)
        )

        print(f"Loading basis list from: {basis_path}")
        basis_list = load_basis_list_from_path(basis_path)

        coeffs = None
        coeffs_ts = None
        if coeffs_ts_path.exists():
            print(f"Loading time-series coefficients from: {coeffs_ts_path}")
            coeffs_ts = pd.read_csv(coeffs_ts_path, parse_dates=["date"])
        elif coeffs_path.exists():
            print(f"Loading coefficient snapshot from: {coeffs_path}")
            coeffs = pd.read_csv(coeffs_path, index_col=0)
        else:
            raise FileNotFoundError(
                f"Coefficients file not found: {coeffs_ts_path} or {coeffs_path}"
            )

        print(f"Loading index weights from: {weights_path}")
        if weights_path.exists():
            w_spx = pd.read_csv(weights_path, index_col=0, parse_dates=True)
        else:
            raise FileNotFoundError(f"Weights file not found: {weights_path}")

        universe = [str(c) for c in w_spx.columns.tolist()]
        w_spx_aligned = w_spx.reindex(columns=universe).fillna(0.0)

        if coeffs_ts is not None:
            basis_weights = self._compute_from_timeseries(
                coeffs_ts, w_spx_aligned, basis_list
            )
        else:
            basis_weights = self._compute_from_snapshot(
                coeffs, w_spx_aligned, basis_list
            )

        self._normalize_and_save_single_row(basis_weights, out_dir)

        # Return final basis_weights
        final_path = out_dir / "basis_weights.csv"
        if final_path.exists():
            return pd.read_csv(final_path, index_col=0)
        return basis_weights
