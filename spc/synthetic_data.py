"""Synthetic market simulator for experiments and backtests.

The module provides the :class:`DynamicIndexSimulator`, a small,
configurable market simulator that produces monthly time-series of
asset prices, shares outstanding, market capitalizations, and index
membership for a synthetic equity market.

Features:
        - Sector structure with per-sector volatility, correlation and
            optional drift.
        - Heavy-tailed shares-outstanding (Pareto-like) to produce skewed
            market capitalizations.
        - Asset births and removals (churn) controlled by Poisson rates and a
            bottom-cap removal policy.
        - Produces monthly prices, market-cap time series, normalized index
            weights, and a per-month record of active tickers.

Utilities:
        - ``generate_market_weights``: convert market caps to row-normalized
            index weights.
        - ``apply_max_stock_weight``: enforce a per-stock concentration cap
            while re-normalizing rows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Tuple, List, Dict, Set, Optional


class DynamicIndexSimulator:
    """Simulate a market index with dynamic membership and monthly prices.

    The simulator creates an initial universe organized into sectors and
    then advances month-by-month applying sector-correlated returns,
    occasional asset removals (bottom-cap policy) and asset additions
    (births). It tracks per-ticker price history, shares-outstanding and
    market-cap history and returns:

    - monthly ``prices_df`` (DataFrame indexed by date with tickers as
      columns, NaN when a ticker was inactive),
    - ``market_cap_df`` (per-ticker market-cap history) and
    - ``weights_df`` (row-normalized index weights derived from market caps).

    Attributes:
        n_sectors (int): Number of sectors used to seed the universe.
        assets_per_sector (int): Initial number of assets per sector.
        n_months (int): Number of monthly observations to simulate.
        removal_rate (float): Monthly removal Poisson intensity multiplier.
        addition_rate (float): Monthly addition Poisson intensity multiplier.
        random_seed (int): NumPy RNG seed for reproducibility.
    """

    def __init__(
        self,
        n_sectors: int = 20,
        assets_per_sector: int = 25,
        n_months: int = 12 * 15,
        removal_rate: float = 0.01,
        addition_rate: float = 0.01,
        random_seed: int = 42,
        index_size: Optional[int] = None,
        removal_bottom_pct: float = 0.2,
        sector_volatility: Optional[Dict[str, float]] = None,
        sector_correlation: Optional[Dict[str, float]] = None,
        sector_drift: Optional[Dict[str, float]] = None,
        cap_pareto_alpha: float = 1.2,
        cap_scale: float = 50,
    ) -> None:
        """Initialize the simulator configuration and seed the initial universe.

        Args:
            n_sectors (int): Number of sectors in the synthetic market.
            assets_per_sector (int): Initial assets per sector.
            n_months (int): Number of monthly observations to simulate.
            removal_rate (float): Monthly expected removal-rate multiplier
                used as the Poisson intensity factor for removals.
            addition_rate (float): Monthly expected addition-rate multiplier
                used as the Poisson intensity for additions.
            random_seed (int): RNG seed to ensure reproducibility.
            index_size (int | None): Target number of constituents in the
                index each month. If ``None``, defaults to
                ``n_sectors * assets_per_sector``.
            removal_bottom_pct (float): Fraction of the smallest-cap names
                eligible for removal under the bottom-cap policy.
            sector_volatility (dict | None): Optional mapping of sector ->
                annual volatility (overrides defaults).
            sector_correlation (dict | None): Optional mapping of sector ->
                within-sector correlation.
            sector_drift (dict | None): Optional sector drift configuration.
            cap_pareto_alpha (float): Pareto alpha for sampling shares-
                outstanding (heavy-tailed behavior).
            cap_scale (float): Scale factor for shares-outstanding sampling.

        The constructor creates the initial tickers, assigns sectors and
        initializes RNG and time-series containers used by the simulation.
        """
        self.n_sectors = n_sectors
        self.assets_per_sector = assets_per_sector
        self.n_months = n_months
        self.removal_rate = removal_rate
        self.addition_rate = addition_rate
        self.random_seed = random_seed
        self.index_size = (
            index_size
            if index_size is not None
            else (self.n_sectors * self.assets_per_sector)
        )

        np.random.seed(random_seed)

        # Keep market caps skewed (Pareto).
        self.cap_pareto_alpha = cap_pareto_alpha
        self.cap_scale = cap_scale
        self.removal_bottom_pct = removal_bottom_pct

        # Initialize sector properties
        self.sector_names = [
            "Technology",
            "Healthcare",
            "Finance",
            "Energy",
            "Consumer",
        ][:n_sectors]

        # Default sector volatility and correlation; can be overridden via init
        self.sector_volatility = (
            sector_volatility
            if sector_volatility is not None
            else {
                "Technology": 0.25,
                "Healthcare": 0.20,
                "Finance": 0.22,
                "Energy": 0.30,
                "Consumer": 0.18,
            }
        )

        self.sector_correlation = (
            sector_correlation
            if sector_correlation is not None
            else {
                "Technology": 0.60,
                "Healthcare": 0.50,
                "Finance": 0.70,
                "Energy": 0.65,
                "Consumer": 0.55,
            }
        )

        # Optional per-sector drift configuration (dict with 'low'/'high')
        self.sector_drift = sector_drift if sector_drift is not None else {}

        # Make the initial asset universe.
        self.total_assets_created = 0
        self.ticker_history = {}
        self._create_initial_assets()

        # Price and asset tracking state.
        self.prices = {}
        self.current_assets = set(self.initial_assets)
        self.asset_sectors = self.initial_asset_sectors.copy()

    def _create_initial_assets(self) -> None:
        """Construct the initial asset universe and record sector mappings.

        Populates:
            - ``self.initial_assets``: list of initial ticker identifiers.
            - ``self.initial_asset_sectors``: mapping ticker -> sector.
            - ``self.ticker_history``: metadata with created/removed months.
        """
        self.initial_assets = []
        self.initial_asset_sectors = {}

        for sector in self.sector_names:
            for _ in range(self.assets_per_sector):
                ticker = self._generate_ticker(sector)
                self.initial_assets.append(ticker)
                self.initial_asset_sectors[ticker] = sector
                self.ticker_history[ticker] = {
                    "created_month": 0,
                    "removed_month": None,
                    "sector": sector,
                }

    def _sample_shares_outstanding(self) -> float:
        """Sample a heavy-tailed shares-outstanding value.

        Uses a shifted Pareto distribution to produce skewed shares-
        outstanding values. Returned values are clipped to be at least 1.0.

        Returns:
            float: Sampled shares-outstanding (>= 1.0).
        """

        # We wanna generate a heavy-tailed distribution so we use pareto
        val = (np.random.pareto(self.cap_pareto_alpha) + 1.0) * self.cap_scale
        return max(val, 1.0)

    def _generate_ticker(self, sector: str) -> str:
        """Create a new, unique ticker identifier for the given sector.

        The ticker embeds the first three uppercase letters of the sector
        and an internal sequential id. The internal counter
        ``self.total_assets_created`` is incremented as a side-effect.

        Args:
            sector (str): Sector name for the new asset.

        Returns:
            str: Unique ticker identifier.
        """
        sector_prefix = sector[:3].upper()
        ticker = f"{sector_prefix}{self.total_assets_created}"
        self.total_assets_created += 1
        return ticker

    def _get_sector_correlation_matrix(self, sector: str, n_assets: int) -> np.ndarray:
        """Build a positive-semidefinite within-sector correlation matrix.

        The matrix has ones on the diagonal and a constant off-diagonal
        value equal to the configured sector correlation. If numerical
        issues produce negative eigenvalues, a small diagonal jitter is
        applied to restore positive-definiteness.

        Args:
            sector (str): Sector name.
            n_assets (int): Number of assets in the sector.

        Returns:
            np.ndarray: ``(n_assets, n_assets)`` correlation matrix.
        """
        base_corr = self.sector_correlation[sector]
        cov = np.ones((n_assets, n_assets)) * base_corr
        cov[np.diag_indices_from(cov)] = 1.0
        eigvals = np.linalg.eigvals(cov)
        if np.any(eigvals < 0):
            cov = cov + np.eye(n_assets) * (abs(min(eigvals)) + 0.01)
        return cov

    def _apply_removals(self, month: int, all_existing_tickers: Set[str]) -> None:
        """Apply the bottom-cap removal policy for the current month.

        Behavior:
            - Draw a Poisson count with intensity ``removal_rate * n`` where
              ``n`` is the current universe size.
            - From the subset of eligible tickers (not previously removed)
              select candidates from the bottom ``removal_bottom_pct`` by
              market-cap and remove up to the Poisson-drawn count.

        Args:
            month (int): Current month index (1-based after initialization).
            all_existing_tickers (Set[str]): Mutable set of currently active
                tickers; removed tickers are removed from this set.
        """
        # Determine expected number of removals this month using a Poisson
        # approximation where the intensity scales with the current universe
        # size.
        lam = self.removal_rate * len(all_existing_tickers)

        # Sample the integer count of removals for this month. A result of
        # zero means no removals - in that case we simply return early.
        n_remove = np.random.poisson(lam)
        if n_remove <= 0:
            return

        # Build the list of tickers that are eligible for removal. We exclude
        # any ticker that has already been marked as removed in
        # `self.ticker_history` so we never re-remove an already-removed
        # instrument.
        removable = [
            t
            for t in all_existing_tickers
            if self.ticker_history[t]["removed_month"] is None
        ]
        if len(removable) == 0:
            # Nothing eligible to remove.
            return

        # Determine current market caps for the removable subset. Some
        # tickers may not yet have market-cap history in `self.market_caps`
        # (rare), so we only include those present.
        current_caps = {
            t: self.market_caps[t][-1] for t in removable if t in self.market_caps
        }

        # Sort candidates by ascending market-cap so we can target the
        # 'smallest' names for removal (bottom-cap removal policy).
        sorted_by_cap = sorted(current_caps.items(), key=lambda x: x[1])

        # Compute the size of the bottom bucket from which we will sample
        # removals. We ensure at least one candidate using `max(1, ...)`.
        bottom_count = max(1, int(len(sorted_by_cap) * self.removal_bottom_pct))
        bottom_candidates = [t for t, _ in sorted_by_cap[:bottom_count]]

        # If the Poisson draw requests more removals than candidates, clamp
        # to the candidate count to avoid errors from sampling without
        # replacement below.
        n_remove = min(n_remove, len(bottom_candidates))

        # Randomly select the tickers to remove from the bottom candidates
        # without replacement and mark them as removed. We store the
        # removal month as `month + 1` to indicate the first month they are
        # absent in subsequent data construction; then remove them from the
        # `all_existing_tickers` set so future iterations no longer include
        # them in price updates.
        removed_samples = set(
            np.random.choice(list(bottom_candidates), size=n_remove, replace=False)
        )
        for rt in removed_samples:
            self.ticker_history[rt]["removed_month"] = month + 1
            all_existing_tickers.remove(rt)

    def _apply_additions(
        self, month: int, all_existing_tickers: Set[str], all_tickers_set: Set[str]
    ) -> None:
        """Create new assets for the month and initialize their histories.

        The number of additions is drawn from a Poisson distribution with
        intensity ``addition_rate * current_universe_size``. For each new
        asset the method assigns a sector, generates a ticker, samples
        shares-outstanding, initializes price and market-cap histories and
        records creation metadata.

        Args:
            month (int): Current month index.
            all_existing_tickers (Set[str]): Mutable set of currently active
                tickers; new tickers are added in-place.
            all_tickers_set (Set[str]): Mutable set of all tickers that have
                ever existed; new tickers are added in-place.
        """
        # Sample the number of new assets to create this month. The Poisson
        # intensity scales with the current universe size via
        # `self.addition_rate`
        n_new_assets = np.random.poisson(self.addition_rate * len(all_existing_tickers))

        # For each new asset, choose a sector, generate a unique ticker id,
        # initialize its shares outstanding, price and market-cap history,
        # record its sector mapping and creation metadata, and add it to the
        # sets that track active and historical tickers. We intentionally
        # initialize the first price to 100.0 and market cap to
        # `100.0 * shares_outstanding` as a deterministic starting point -
        # downstream code will evolve prices month-by-month.
        for asset_idx in range(n_new_assets):
            # Pick sector at random among simulator sector names.
            sector = np.random.choice(self.sector_names)

            # Create a unique ticker identifier.
            ticker = self._generate_ticker(sector)

            # Add to current active set and to the historical set of all
            # tickers that have ever existed.
            all_existing_tickers.add(ticker)
            all_tickers_set.add(ticker)

            # Initialize sample-level state for the new ticker: shares,
            # price history (single initial price), market-cap history,
            # sector assignment and creation metadata. These fields are
            # used by the rest of the simulator when updating prices and
            # computing top-index constituents.
            self.shares_outstanding[ticker] = self._sample_shares_outstanding()
            self.prices[ticker] = [100.0]
            self.market_caps[ticker] = [100.0 * self.shares_outstanding[ticker]]
            self.asset_sectors[ticker] = sector
            self.ticker_history[ticker] = {
                "created_month": month,
                "removed_month": None,
                "sector": sector,
            }

    def simulate_prices(self) -> Tuple[pd.DataFrame, List[str], List[Set[str]]]:
        """Run the month-by-month simulation and return generated price data.

        The simulator advances for ``n_months`` and for each month generates
        sector-correlated returns, updates prices and market caps, applies
        removals and additions, and records the target index membership.

        Returns:
            Tuple[pd.DataFrame, List[str], List[Set[str]]]:
                - prices_df: DataFrame of shape ``(n_months, n_tickers)`` with
                  NaN where a ticker was inactive.
                - all_tickers_sorted: Sorted list of all tickers that ever
                  existed in the simulation.
                - monthly_tickers: List of sets (length ``n_months``) with the
                  index constituents for each month.
        """

        # Generate monthly dates (first-of-month)
        start_date = datetime(2025, 12, 1) - pd.DateOffset(months=self.n_months - 1)
        dates = []
        current_date = start_date
        for _ in range(self.n_months):
            dates.append(current_date)
            if current_date.month == 12:
                current_date = datetime(current_date.year + 1, 1, 1)
            else:
                current_date = datetime(current_date.year, current_date.month + 1, 1)

        # Initialize shares, prices and market-cap containers for initial assets
        self.shares_outstanding = {}
        for ticker in self.initial_assets:
            self.shares_outstanding[ticker] = self._sample_shares_outstanding()

        for ticker in self.initial_assets:
            self.prices[ticker] = [100.0]

        all_tickers_set = set(self.initial_assets)
        all_existing_tickers = set(self.initial_assets)

        self.market_caps = {}
        for ticker in self.initial_assets:
            self.market_caps[ticker] = [
                self.prices[ticker][0] * self.shares_outstanding[ticker]
            ]

        target_index_size = self.index_size
        current_market_caps = {
            ticker: self.market_caps[ticker][-1] for ticker in all_existing_tickers
        }
        top_tickers = sorted(
            current_market_caps.items(), key=lambda x: x[1], reverse=True
        )[:target_index_size]
        self.current_assets = set(ticker for ticker, _ in top_tickers)
        monthly_tickers = [set(self.current_assets)]

        for month in range(1, self.n_months):
            returns = self._generate_monthly_returns(all_existing_tickers)

            for ticker in list(all_existing_tickers):
                ret = returns.get(ticker, 0.0)
                prev_price = self.prices[ticker][-1]
                if ret <= -0.999999:
                    ret = -0.999999
                new_price = prev_price * (1 + ret)
                if new_price <= 0.0:
                    new_price = 1e-8
                self.prices[ticker].append(new_price)
                market_cap = new_price * self.shares_outstanding[ticker]
                if market_cap < 0.0:
                    market_cap = 0.0
                self.market_caps[ticker].append(market_cap)

            # removals / additions
            self._apply_removals(month, all_existing_tickers)
            self._apply_additions(month, all_existing_tickers, all_tickers_set)

            current_market_caps = {
                ticker: self.market_caps[ticker][-1] for ticker in all_existing_tickers
            }
            top_tickers = sorted(
                current_market_caps.items(), key=lambda x: x[1], reverse=True
            )[:target_index_size]
            new_current_assets = set(ticker for ticker, _ in top_tickers)
            self.current_assets = new_current_assets
            monthly_tickers.append(set(self.current_assets))

        price_data = []
        for month in range(self.n_months):
            month_prices = {}
            for ticker in all_tickers_set:
                created_month = self.ticker_history[ticker]["created_month"]
                removed_month = self.ticker_history[ticker]["removed_month"]
                if created_month <= month and (
                    removed_month is None or month < removed_month
                ):
                    months_since_creation = month - created_month
                    month_prices[ticker] = self.prices[ticker][months_since_creation]
            price_data.append(month_prices)

        prices_df = pd.DataFrame(price_data, index=pd.DatetimeIndex(dates))
        prices_df.index.name = "date"
        prices_df = prices_df.sort_index(axis=1)

        return prices_df, sorted(list(all_tickers_set)), monthly_tickers

    def _generate_monthly_returns(self, tickers: Set[str]) -> Dict[str, float]:
        """Generate one-month returns for the provided tickers.

        For each sector the function constructs a correlated Gaussian draw
        using a Cholesky factorization of the within-sector correlation matrix
        and scales by the sector monthly volatility. A sector-level drift is
        sampled from the configured drift range and added to each asset's
        idiosyncratic return.

        Args:
            tickers (Set[str]): Active tickers to generate returns for.

        Returns:
            Dict[str, float]: Mapping ticker -> monthly return (decimal,
            e.g. 0.02 for +2%).
        """
        returns = {}
        sectors_assets = {}
        for ticker in tickers:
            sector = self.asset_sectors[ticker]
            sectors_assets.setdefault(sector, []).append(ticker)

        for sector, sector_tickers in sectors_assets.items():
            n_assets = len(sector_tickers)
            vol_annual = self.sector_volatility[sector]
            vol = vol_annual / np.sqrt(12.0)
            corr_matrix = self._get_sector_correlation_matrix(sector, n_assets)
            L = np.linalg.cholesky(corr_matrix)
            z = np.random.randn(n_assets)
            sector_returns = (L @ z) * vol

            low_default = -0.015
            high_default = 0.015
            drift_cfg = getattr(self, "sector_drift", {}) or {}
            if (
                isinstance(drift_cfg, dict)
                and "low" in drift_cfg
                and "high" in drift_cfg
            ):
                try:
                    low = drift_cfg.get("low", low_default)
                    high = drift_cfg.get("high", high_default)
                except Exception:
                    low, high = low_default, high_default
            else:
                low, high = low_default, high_default

            sector_drift = np.random.uniform(low, high)

            for i, ticker in enumerate(sector_tickers):
                returns[ticker] = sector_returns[i] + sector_drift

        return returns

    def generate_market_weights(
        self, prices_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Convert simulated prices into market-cap history and index weights.

        The function constructs a per-ticker market-cap DataFrame aligned to
        the input dates, clips negative values, assigns uniform weights on
        rows where total market cap is zero, and returns the row-normalized
        weights DataFrame together with the market-cap DataFrame.

        Args:
            prices_df (pd.DataFrame): Prices as produced by
                ``simulate_prices`` (dates x tickers).

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: ``(weights_df, market_cap_df)``
                where ``market_cap_df`` contains per-ticker market-cap values
                and ``weights_df`` contains row-normalized index weights.
        """
        # Build market-cap records for every date and ticker
        all_tickers = list(prices_df.columns)
        dates = prices_df.index
        market_cap_data = []

        for month, date in enumerate(dates):
            month_caps = {}
            for ticker in all_tickers:
                # Include only tickers active in this month
                created_month = self.ticker_history[ticker]["created_month"]
                removed_month = self.ticker_history[ticker]["removed_month"]

                if created_month <= month and (
                    removed_month is None or month < removed_month
                ):
                    # Ticker was alive this month
                    months_since_creation = month - created_month
                    market_cap = self.market_caps[ticker][months_since_creation]
                    month_caps[ticker] = market_cap
                # otherwise leave as NaN

            market_cap_data.append(month_caps)

        market_cap_df = pd.DataFrame(market_cap_data, index=dates)
        market_cap_df.index.name = "date"
        market_cap_df = market_cap_df.sort_index(axis=1)

        # Clip negative market-cap values (shouldn't happen often)
        if (market_cap_df < 0).any().any():
            print(
                "WARNING: negative market-cap values detected - clipping negatives to zero before computing weights."
            )
        market_cap_df = market_cap_df.clip(lower=0.0)

        # If a date has zero total market cap, give uniform weights to active
        # tickers for that row so things don't blow up.
        row_sums = market_cap_df.sum(axis=1)
        zero_rows = row_sums[row_sums == 0.0].index.tolist()
        if len(zero_rows) > 0:
            print(
                f"WARNING: {len(zero_rows)} rows have zero total market cap; assigning uniform weights among active tickers for those rows."
            )
            for date in zero_rows:
                active = market_cap_df.loc[date].dropna()
                if active.size > 0:
                    market_cap_df.loc[date, active.index] = 1.0 / active.size

        # Convert market caps to weights by normalizing across active names
        weights_df = market_cap_df.div(market_cap_df.sum(axis=1), axis=0)

        return weights_df, market_cap_df

    def apply_max_stock_weight(
        self, weights_df: pd.DataFrame, max_weight: Optional[float]
    ) -> pd.DataFrame:
        """Apply a per-stock cap and re-normalize weights across active tickers.

        The procedure iteratively caps overweight names and redistributes the
        excess proportionally to the remaining uncapped names until the row
        respects the cap or a maximum number of iterations is reached. Rows
        with no active tickers are left unchanged.

        Args:
            weights_df (pd.DataFrame): Row-normalized weight matrix where
                inactive tickers are NaN.
            max_weight (float | None): Per-stock cap to enforce. If ``None``,
                the input is returned unchanged.

        Returns:
            pd.DataFrame: Adjusted weights DataFrame with caps enforced and
                rows re-normalized across active names.

        Raises:
            ValueError: If the requested cap is infeasible for a row
                (``max_weight * n_active < 1``) for any row.

        Inline comments.
        """
        # If no cap is specified, return the original weights unchanged.
        if max_weight is None:
            return weights_df

        new_w = weights_df.copy()

        # Process each time period (row) independently. The index typically
        # represents dates/months, and each row contains weights across all tickers.
        for idx in new_w.index:
            # Extract the weight row for this time period.
            row = new_w.loc[idx].copy()

            # Identify which tickers are active (non-NaN) in this period.
            active = row.dropna()

            # If no tickers are active in this row (all NaN), there's nothing to
            # cap or redistribute. Skip to the next row.
            n_active = active.size
            if n_active == 0:
                continue

            # Feasibility check: If we cap every active ticker at `max_weight`,
            # the maximum possible sum is `cap * n_active`. If this product is
            # less than 1.0 (the required row sum for valid portfolio weights),
            # it's mathematically impossible to satisfy both the cap constraint
            # and the normalization constraint.
            if max_weight * n_active < 1.0 - 1e-12:
                raise ValueError(
                    f"max_weight={max_weight} is infeasible for {n_active} active tickers; "
                    "cap * n_active must be >= 1.0"
                )

            w = row.fillna(0.0)
            iteration = 0

            # Iterative cap-and-redistribute algorithm:
            # The goal is to enforce that no weight exceeds `max_weight` while maintaining
            # a row sum of 1.0. The algorithm repeatedly identifies overweight
            # positions, clips them to the max_weight, and redistributes the excess weight
            # proportionally to positions still below the max_weight. This continues until
            # no positions exceed the max_weight or we reach the iteration limit.
            while True:
                iteration += 1

                # Identify all positions that currently exceed the max_weight. These are
                # the "violators" that need to be clipped in this iteration.
                over = w[w > max_weight]

                # If no positions exceed the max_weight, we've converged to a valid solution.
                # Exit the loop and proceed to write the final weights back to the DataFrame.
                if over.empty:
                    break

                # Calculate the total excess weight that needs to be redistributed.
                # For each overweight position, the excess is (current_weight - max_weight).
                # Summing these gives the total amount to redistribute to other positions.
                excess = (over - max_weight).sum()

                # Clip all overweight positions to exactly the max_weight. These positions
                # are now "frozen" at the max_weight and won't receive any redistributed weight
                # in subsequent iterations (they're no longer in the "receivers" set).
                w.loc[over.index] = max_weight

                # Identify potential receivers: positions that are (1) below the max_weight
                # and (2) have positive weight. These are candidates to absorb the
                # redistributed excess. We exclude zero-weight positions to avoid
                # introducing new positions that weren't originally in the portfolio.
                receivers = w[(w < max_weight) & (w > 0.0)]

                # Redistribute the excess proportionally to the current
                # weights of the receiver positions. This preserves the relative
                # ordering of non-capped positions while respecting the max_weight constraint.
                total_receivers = receivers.sum()
                w.loc[receivers.index] = w.loc[receivers.index] + excess * (
                    w.loc[receivers.index] / total_receivers
                )

                # If the algorithm hasn't converged after 50 iterations,
                # force convergence by normalizing the current weights to sum to 1.0.
                if iteration > 50:
                    w = w / w.sum()
                    break

            for col in row.index:
                if pd.isna(row[col]):
                    # Ticker was inactive in this period; keep it as NaN.
                    new_w.at[idx, col] = np.nan
                else:
                    # Ticker was active; assign the adjusted weight.
                    new_w.at[idx, col] = w[col]

        return new_w
