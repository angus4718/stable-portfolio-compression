"""Synthetic market simulator for experiments and backtests.

This module provides the `DynamicIndexSimulator`, a configurable
simulator that generates a time-series of asset prices, market
capitalizations, and index membership for a synthetic equity market.

Features
- Sector structure with per-sector volatility, correlation, and optional
    drift.
- Heavy-tailed shares-outstanding (Pareto-like) to produce skewed market
    capitalizations.
- Asset births and removals (churn) controlled by Poisson rates and a
    bottom-cap removal policy.
- Generates monthly prices, market-cap time series, normalized index
    weights, and a record of which tickers are active each month.
- Utilities to convert market caps to normalized weights
    (`generate_market_weights`) and to enforce a per-stock
    concentration cap (`apply_max_stock_weight`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import json
from datetime import datetime
from typing import Tuple, List, Dict, Set
from pathlib import Path


class DynamicIndexSimulator:
    """Simulate a market index with dynamic asset membership and prices."""

    def __init__(
        self,
        n_sectors: int = 20,
        assets_per_sector: int = 25,
        n_months: int = 12 * 15,
        removal_rate: float = 0.01,
        addition_rate: float = 0.01,
        random_seed: int = 42,
        index_size: int = None,
        removal_bottom_pct: float = 0.2,
        sector_volatility: Dict[str, float] | None = None,
        sector_correlation: Dict[str, float] | None = None,
        sector_drift: Dict[str, float] | None = None,
        cap_pareto_alpha: float | None = None,
        cap_scale: float | None = None,
    ):
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

        # controls for skewed market cap (shares outstanding) distribution
        self.cap_pareto_alpha = (
            cap_pareto_alpha if cap_pareto_alpha is not None else 1.2
        )
        self.cap_scale = cap_scale if cap_scale is not None else 50.0
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
        self.sector_drift = sector_drift or {}

        # Create initial asset universe
        self.total_assets_created = 0
        self.ticker_history = {}
        self._create_initial_assets()

        # Price tracking
        self.prices = {}
        self.current_assets = set(self.initial_assets)
        self.asset_sectors = self.initial_asset_sectors.copy()

    def _create_initial_assets(self) -> None:
        self.initial_assets = []
        self.initial_asset_sectors = {}

        for sector_idx, sector in enumerate(self.sector_names):
            for asset_idx in range(self.assets_per_sector):
                ticker = self._generate_ticker(sector, asset_idx)
                self.initial_assets.append(ticker)
                self.initial_asset_sectors[ticker] = sector
                self.ticker_history[ticker] = {
                    "created_month": 0,
                    "removed_month": None,
                    "sector": sector,
                }

    def _sample_shares_outstanding(self) -> float:
        val = (np.random.pareto(self.cap_pareto_alpha) + 1.0) * self.cap_scale
        return float(max(val, 1.0))

    def _generate_ticker(self, sector: str, idx: int) -> str:
        sector_prefix = sector[:3].upper()
        ticker = f"{sector_prefix}{self.total_assets_created:04d}"
        self.total_assets_created += 1
        return ticker

    def _get_sector_correlation_matrix(self, sector: str, n_assets: int) -> np.ndarray:
        base_corr = self.sector_correlation[sector]
        cov = np.ones((n_assets, n_assets)) * base_corr
        cov[np.diag_indices_from(cov)] = 1.0
        eigvals = np.linalg.eigvals(cov)
        if np.any(eigvals < 0):
            cov = cov + np.eye(n_assets) * (abs(min(eigvals)) + 0.01)
        return cov

    def simulate_prices(self) -> Tuple[pd.DataFrame, List[str], List[Set[str]]]:
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

        # Generate shares outstanding
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

        target_index_size = int(self.index_size)
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

            # removals
            lam = max(0.0, self.removal_rate * len(all_existing_tickers))
            n_remove = np.random.poisson(lam)
            if n_remove > 0:
                removable = [
                    t
                    for t in all_existing_tickers
                    if self.ticker_history[t]["removed_month"] is None
                ]
                if len(removable) > 0:
                    current_caps = {
                        t: self.market_caps[t][-1]
                        for t in removable
                        if t in self.market_caps
                    }
                    sorted_by_cap = sorted(current_caps.items(), key=lambda x: x[1])
                    bottom_count = max(
                        1, int(len(sorted_by_cap) * float(self.removal_bottom_pct))
                    )
                    bottom_candidates = [t for t, _ in sorted_by_cap[:bottom_count]]
                    if len(bottom_candidates) > 0:
                        n_remove = min(n_remove, len(bottom_candidates))
                        removed_samples = set(
                            np.random.choice(
                                list(bottom_candidates), size=n_remove, replace=False
                            )
                        )
                        for rt in removed_samples:
                            self.ticker_history[rt]["removed_month"] = month + 1
                            all_existing_tickers.remove(rt)

            # additions
            n_new_assets = np.random.poisson(
                max(0.0, self.addition_rate * len(all_existing_tickers))
            )
            for asset_idx in range(n_new_assets):
                sector = np.random.choice(self.sector_names)
                ticker = self._generate_ticker(sector, asset_idx)
                all_existing_tickers.add(ticker)
                all_tickers_set.add(ticker)
                self.shares_outstanding[ticker] = self._sample_shares_outstanding()
                self.prices[ticker] = [100.0]
                self.market_caps[ticker] = [100.0 * self.shares_outstanding[ticker]]
                self.asset_sectors[ticker] = sector
                self.ticker_history[ticker] = {
                    "created_month": month,
                    "removed_month": None,
                    "sector": sector,
                }

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
        returns = {}
        sectors_assets = {}
        for ticker in tickers:
            sector = self.asset_sectors[ticker]
            sectors_assets.setdefault(sector, []).append(ticker)

        for sector, sector_tickers in sectors_assets.items():
            n_assets = len(sector_tickers)
            vol_annual = float(self.sector_volatility[sector])
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
                    low = float(drift_cfg.get("low", low_default))
                    high = float(drift_cfg.get("high", high_default))
                except Exception:
                    low, high = low_default, high_default
            else:
                low, high = low_default, high_default

            sector_drift = np.random.uniform(low, high)

            for i, ticker in enumerate(sector_tickers):
                returns[ticker] = sector_returns[i] + sector_drift

        return returns

    def generate_market_weights(
        self, prices_df: pd.DataFrame, monthly_tickers: List[Set[str]]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Generate market-cap time series and normalized index weights.

        The output includes all tickers that ever existed; months when a ticker
        was not active are represented with NaN.
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
                    # Ticker existed in this month
                    months_since_creation = month - created_month
                    market_cap = self.market_caps[ticker][months_since_creation]
                    month_caps[ticker] = market_cap
                # otherwise leave as NaN

            market_cap_data.append(month_caps)

        market_cap_df = pd.DataFrame(market_cap_data, index=dates)
        market_cap_df.index.name = "date"
        market_cap_df = market_cap_df.sort_index(axis=1)

        # Clip any negative market-cap values (safety) and warn
        if (market_cap_df < 0).any().any():
            print(
                "WARNING: negative market-cap values detected — clipping negatives to zero before computing weights."
            )
        market_cap_df = market_cap_df.clip(lower=0.0)

        # Handle rows where total market cap is zero (e.g., all NaN or clipped to zero)
        row_sums = market_cap_df.sum(axis=1)
        zero_rows = row_sums[row_sums == 0.0].index.tolist()
        if len(zero_rows) > 0:
            print(
                f"WARNING: {len(zero_rows)} rows have zero total market cap; assigning uniform weights among active tickers for those rows."
            )
            for date in zero_rows:
                active = market_cap_df.loc[date].dropna()
                if active.size > 0:
                    market_cap_df.loc[date, active.index] = 1.0 / float(active.size)

        # Convert market caps to weights by normalizing across active names
        weights_df = market_cap_df.div(market_cap_df.sum(axis=1), axis=0)

        return weights_df, market_cap_df

    def apply_max_stock_weight(
        self, weights_df: pd.DataFrame, max_weight: float
    ) -> pd.DataFrame:
        if max_weight is None:
            return weights_df

        new_w = weights_df.copy()
        for idx in new_w.index:
            row = new_w.loc[idx].copy()
            active = row.dropna()
            if active.size == 0:
                continue

            cap = float(max_weight)
            n_active = active.size
            if cap * n_active < 1.0 - 1e-12:
                continue

            w = row.fillna(0.0)
            iteration = 0
            while True:
                iteration += 1
                over = w[w > cap]
                if over.empty:
                    break

                excess = (over - cap).sum()
                w.loc[over.index] = cap
                receivers = w[(w < cap) & (w > 0.0)]
                if receivers.sum() <= 0:
                    uncapped = w[w < cap]
                    if uncapped.empty:
                        w.loc[active.index] = 1.0 / n_active
                        break
                    add = float(excess) / float(uncapped.size)
                    w.loc[uncapped.index] = w.loc[uncapped.index] + add
                else:
                    total_receivers = receivers.sum()
                    w.loc[receivers.index] = w.loc[receivers.index] + excess * (
                        w.loc[receivers.index] / total_receivers
                    )

                if iteration > 50:
                    w = w / w.sum()
                    break

            for col in row.index:
                if pd.isna(row[col]):
                    new_w.at[idx, col] = float("nan")
                else:
                    new_w.at[idx, col] = float(w[col])

        return new_w
