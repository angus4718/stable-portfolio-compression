"""
Generate synthetic price and market-index weight data with dynamic asset membership.

This module produces CSV files in the `synthetic_data` output directory. The outputs are:

- `prices_monthly.csv`: monthly prices (index: start-of-month dates; columns: tickers)
- `market_index_weights.csv`: normalized market-cap weights (index: dates; columns: tickers)
- `market_cap_values.csv`: raw market-cap values used to compute weights

The simulator models a dynamic universe with heavy-tailed market caps (Pareto-like
shares-outstanding) and Poisson-distributed additions/removals. Removals are drawn
from the smallest fraction of names by market cap (configurable).
"""

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
    ):
        """
        Initialize the simulator.

        Args:
            n_sectors: Number of asset sectors.
            assets_per_sector: Initial assets per sector.
            n_months: Number of months to simulate.
            removal_rate: Per-stock expected removal rate per month.
            addition_rate: Per-stock expected addition rate per month.
            random_seed: Random seed for reproducibility.
            index_size: Number of constituents to keep in the index (optional).
            removal_bottom_pct: Fraction of smallest names eligible for removal.

        Returns:
            None
        """
        self.n_sectors = n_sectors
        self.assets_per_sector = assets_per_sector
        self.n_months = n_months
        self.removal_rate = removal_rate
        self.addition_rate = addition_rate
        self.random_seed = random_seed
        # Fixed index size (number of constituents to keep each month).
        # If None, use the full initial universe size.
        self.index_size = (
            index_size
            if index_size is not None
            else (self.n_sectors * self.assets_per_sector)
        )

        np.random.seed(random_seed)

        # controls for skewed market cap (shares outstanding) distribution
        # lower alpha -> heavier tail (more skew). Tune these to adjust concentration.
        self.cap_pareto_alpha = 1.2
        self.cap_scale = 50.0
        # Fraction (0-1) of the smallest market-cap names eligible for removal.
        self.removal_bottom_pct = removal_bottom_pct

        # Initialize sector properties
        self.sector_names = [
            "Technology",
            "Healthcare",
            "Finance",
            "Energy",
            "Consumer",
        ][:n_sectors]

        self.sector_volatility = {
            "Technology": 0.25,
            "Healthcare": 0.20,
            "Finance": 0.22,
            "Energy": 0.30,
            "Consumer": 0.18,
        }

        self.sector_correlation = {
            "Technology": 0.60,
            "Healthcare": 0.50,
            "Finance": 0.70,
            "Energy": 0.65,
            "Consumer": 0.55,
        }

        # Create initial asset universe
        self.total_assets_created = 0
        self.ticker_history = {}  # Track all tickers created
        self._create_initial_assets()

        # Price tracking
        self.prices = {}
        self.current_assets = set(self.initial_assets)
        self.asset_sectors = self.initial_asset_sectors.copy()

    def _create_initial_assets(self) -> None:
        """Create the initial set of assets and record their sectors.

        Args:
            None

        Returns:
            None
        """
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
        """Sample a skewed shares-outstanding value so market caps are heavy-tailed.

        Args:
            None

        Returns:
            A positive float representing shares outstanding for a ticker.
        """
        val = (np.random.pareto(self.cap_pareto_alpha) + 1.0) * self.cap_scale
        return float(max(val, 1.0))

    def _generate_ticker(self, sector: str, idx: int) -> str:
        """Generate a unique ticker symbol for a sector and index.

        Args:
            sector: Sector name used as a prefix.
            idx: Integer index (for uniqueness).

        Returns:
            A string representing the new ticker.
        """
        sector_prefix = sector[:3].upper()
        ticker = f"{sector_prefix}{self.total_assets_created:04d}"
        self.total_assets_created += 1
        return ticker

    def _get_sector_correlation_matrix(self, sector: str, n_assets: int) -> np.ndarray:
        """
        Generate correlation matrix for assets in a sector.

        Args:
            sector: Sector name
            n_assets: Number of assets

        Returns:
            Correlation matrix
        """
        base_corr = self.sector_correlation[sector]

        # Create correlation matrix with within-sector correlation
        cov = np.ones((n_assets, n_assets)) * base_corr
        cov[np.diag_indices_from(cov)] = 1.0

        # Ensure positive semi-definite
        eigvals = np.linalg.eigvals(cov)
        if np.any(eigvals < 0):
            cov = cov + np.eye(n_assets) * (abs(min(eigvals)) + 0.01)

        return cov

    def simulate_prices(self) -> Tuple[pd.DataFrame, List[str]]:
        """Simulate monthly prices for a dynamic universe.

        Constituents are chosen each month by market-cap ranking.

        Args:
            None

        Returns:
            prices_df: DataFrame of monthly prices (index: date, columns: tickers).
            all_tickers: Sorted list of all tickers that ever existed.
        """
        # Generate monthly dates
        start_date = datetime(2025, 12, 1) - pd.DateOffset(months=self.n_months - 1)
        dates = []
        current_date = start_date
        for _ in range(self.n_months):
            dates.append(current_date)
            # Move to first day of next month
            if current_date.month == 12:
                current_date = datetime(current_date.year + 1, 1, 1)
            else:
                current_date = datetime(current_date.year, current_date.month + 1, 1)

        # Generate shares outstanding for all assets (this doesn't change)
        self.shares_outstanding = {}
        for ticker in self.initial_assets:
            # use skewed distribution so a small set of names dominate market cap
            self.shares_outstanding[ticker] = self._sample_shares_outstanding()

        # Initialize prices for all assets
        for ticker in self.initial_assets:
            self.prices[ticker] = [100.0]

        # Track the full universe and the currently existing tickers
        all_tickers_set = set(self.initial_assets)
        all_existing_tickers = set(self.initial_assets)

        # Initialize market-cap time series for each ticker
        self.market_caps = {}
        for ticker in self.initial_assets:
            self.market_caps[ticker] = [
                self.prices[ticker][0] * self.shares_outstanding[ticker]
            ]

        # Target number of index constituents
        target_index_size = int(self.index_size)
        # Choose initial constituents by top market cap
        current_market_caps = {
            ticker: self.market_caps[ticker][-1] for ticker in all_existing_tickers
        }
        top_tickers = sorted(
            current_market_caps.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:target_index_size]
        self.current_assets = set(ticker for ticker, _ in top_tickers)
        monthly_tickers = [set(self.current_assets)]

        for month in range(1, self.n_months):
            # Sample returns for the current universe and update histories
            returns = self._generate_monthly_returns(all_existing_tickers)

            for ticker in all_existing_tickers:
                ret = returns.get(ticker, 0.0)
                prev_price = self.prices[ticker][-1]
                new_price = prev_price * (1 + ret)
                self.prices[ticker].append(new_price)

                market_cap = new_price * self.shares_outstanding[ticker]
                self.market_caps[ticker].append(market_cap)

            # Decide how many stocks to remove this month (Poisson draw).
            # Removals are selected at random from the bottom fraction by market cap
            # and take effect in the following month.
            lam = max(0.0, self.removal_rate * len(all_existing_tickers))
            n_remove = np.random.poisson(lam)
            if n_remove > 0:
                # Candidates not already scheduled for removal
                removable = [
                    t
                    for t in all_existing_tickers
                    if self.ticker_history[t]["removed_month"] is None
                ]
                if len(removable) > 0:
                    # Obtain current market caps for candidates
                    current_caps = {
                        t: self.market_caps[t][-1]
                        for t in removable
                        if t in self.market_caps
                    }
                    # sort assets by market cap ascending and take bottom pct
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
                            # mark removal to take effect next month
                            self.ticker_history[rt]["removed_month"] = month + 1
                            all_existing_tickers.remove(rt)

            # Add new assets: draw a Poisson count and create that many new tickers.
            n_new_assets = np.random.poisson(
                max(0.0, self.addition_rate * len(all_existing_tickers))
            )
            for asset_idx in range(n_new_assets):
                sector = np.random.choice(self.sector_names)
                ticker = self._generate_ticker(sector, asset_idx)
                all_existing_tickers.add(ticker)
                all_tickers_set.add(ticker)

                # new assets also draw from the skewed distribution
                self.shares_outstanding[ticker] = self._sample_shares_outstanding()
                self.prices[ticker] = [100.0]
                self.market_caps[ticker] = [100.0 * self.shares_outstanding[ticker]]

                self.asset_sectors[ticker] = sector
                self.ticker_history[ticker] = {
                    "created_month": month,
                    "removed_month": None,
                    "sector": sector,
                }

            # Select top X by market cap to be in the index
            current_market_caps = {
                ticker: self.market_caps[ticker][-1] for ticker in all_existing_tickers
            }
            top_tickers = sorted(
                current_market_caps.items(), key=lambda x: x[1], reverse=True
            )[:target_index_size]

            # Update current assets (index constituents)
            new_current_assets = set(ticker for ticker, _ in top_tickers)

            # Update current index constituents (this does not remove a ticker from the universe)
            self.current_assets = new_current_assets
            monthly_tickers.append(set(self.current_assets))

        # Assemble a prices DataFrame covering all tickers.
        # Months where a ticker did not yet exist (or was removed) are left as NaN.
        price_data = []
        for month in range(self.n_months):
            month_prices = {}
            for ticker in all_tickers_set:
                # Check if ticker existed at this month
                created_month = self.ticker_history[ticker]["created_month"]
                removed_month = self.ticker_history[ticker]["removed_month"]

                if created_month <= month and (
                    removed_month is None or month < removed_month
                ):
                    # Ticker existed in this month
                    months_since_creation = month - created_month
                    month_prices[ticker] = self.prices[ticker][months_since_creation]
                # else: leave as NaN (ticker didn't exist yet or was removed)

            price_data.append(month_prices)

        # Create DataFrame
        prices_df = pd.DataFrame(price_data, index=pd.DatetimeIndex(dates))
        prices_df.index.name = "date"
        prices_df = prices_df.sort_index(axis=1)  # Sort columns by ticker

        return prices_df, sorted(list(all_tickers_set)), monthly_tickers

    def _generate_monthly_returns(self, tickers: Set[str]) -> Dict[str, float]:
        """Draw correlated monthly returns by sector for the given tickers.

        Args:
            tickers: Set of active tickers for which to draw returns.

        Returns:
            A dict mapping ticker -> monthly return (float).
        """
        returns = {}

        # Group by sector
        sectors_assets = {}
        for ticker in tickers:
            sector = self.asset_sectors[ticker]
            if sector not in sectors_assets:
                sectors_assets[sector] = []
            sectors_assets[sector].append(ticker)

        # For each sector, draw correlated shocks and apply a small drift
        for sector, sector_tickers in sectors_assets.items():
            n_assets = len(sector_tickers)
            vol = self.sector_volatility[sector]

            # Build correlated shocks using Cholesky on sector-level correlation
            corr_matrix = self._get_sector_correlation_matrix(sector, n_assets)
            L = np.linalg.cholesky(corr_matrix)
            z = np.random.randn(n_assets)
            sector_returns = (L @ z) * vol

            # Add a small random sector drift
            sector_drift = np.random.uniform(-0.015, 0.015)

            for i, ticker in enumerate(sector_tickers):
                returns[ticker] = sector_returns[i] + sector_drift

        return returns

    def generate_market_weights(
        self, prices_df: pd.DataFrame, monthly_tickers: List[Set[str]]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Generate market-cap time series and normalized index weights.

        The output includes all tickers that ever existed; months when a ticker
        was not active are represented with NaN.

        Args:
            prices_df: DataFrame of prices produced by `simulate_prices`.
            monthly_tickers: List of sets containing index constituents each month.

        Returns:
            weights_df: DataFrame of market-cap weights (index: date, columns: tickers).
            market_cap_df: DataFrame of raw market-cap values (same shape as weights_df).
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

        # Convert market caps to weights by normalizing across active names
        weights_df = market_cap_df.div(market_cap_df.sum(axis=1), axis=0)

        return weights_df, market_cap_df


def main():
    """Main execution function.

    Args:
        None

    Returns:
        None
    """

    print("\n" + "=" * 70)
    print("SYNTHETIC MARKET DATA GENERATION")
    print("=" * 70)

    # Load configuration
    config_path = Path(__file__).parent / "synthetic_data_config.json"
    if not config_path.exists():
        print(f"ERROR: Configuration file not found at {config_path}")
        print(
            "Please create synthetic_data_config.json in the synthetic_data directory."
        )
        return

    with open(config_path, "r") as f:
        config = json.load(f)

    # Extract simulation parameters
    sim_params = config["simulation_parameters"]
    n_sectors = sim_params["n_sectors"]["value"]
    assets_per_sector = sim_params["assets_per_sector"]["value"]
    n_months = sim_params["n_months"]["value"]
    removal_rate = sim_params["removal_rate"]["value"]
    addition_rate = sim_params["addition_rate"]["value"]
    random_seed = sim_params["random_seed"]["value"]
    index_size = sim_params.get("index_size", {}).get("value", None)

    # Extract sector parameters
    sector_volatility = {
        k: v["value"]
        for k, v in config["sector_volatility"].items()
        if k != "description"
    }
    sector_correlation = {
        k: v["value"]
        for k, v in config["sector_correlation"].items()
        if k != "description"
    }

    output_dir = config["output_directory"]["value"]

    # Display configuration
    print("\nConfiguration:")
    print(f"Sectors: {n_sectors}")
    print(f"Assets per sector: {assets_per_sector}")
    print(f"Total months: {n_months} ({n_months/12:.1f} years)")
    print(f"Monthly removal rate: {removal_rate*100:.1f}%")
    print(f"Monthly addition rate: {addition_rate*100:.1f}%")
    print(f"Random seed: {random_seed}")
    print(f"Output directory: {output_dir}")

    # Generate data
    print("\nGenerating synthetic data...")
    simulator = DynamicIndexSimulator(
        n_sectors=n_sectors,
        assets_per_sector=assets_per_sector,
        n_months=n_months,
        removal_rate=removal_rate,
        addition_rate=addition_rate,
        random_seed=random_seed,
        index_size=index_size,
    )

    # Override sector parameters from config
    simulator.sector_volatility = sector_volatility
    simulator.sector_correlation = sector_correlation

    market_cap_params = config.get("market_cap_parameters", {})
    simulator.cap_pareto_alpha = float(market_cap_params["cap_pareto_alpha"]["value"])
    simulator.cap_scale = float(market_cap_params["cap_scale"]["value"])
    simulator.removal_bottom_pct = float(
        market_cap_params["removal_bottom_pct"]["value"]
    )

    prices_df, all_tickers, monthly_tickers = simulator.simulate_prices()
    weights_df, market_cap_df = simulator.generate_market_weights(
        prices_df, monthly_tickers
    )

    # Save to CSV
    print("\nSaving to CSV files...")
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    prices_path = output_path / config["output_files"]["prices"]["filename"]
    weights_path = output_path / config["output_files"]["weights"]["filename"]
    market_cap_path = output_path / config["output_files"]["market_caps"]["filename"]

    prices_df.to_csv(prices_path)
    weights_df.to_csv(weights_path)
    market_cap_df.to_csv(market_cap_path)

    print(f"Saved: {prices_path}")
    print(f"Saved: {weights_path}")
    print(f"Saved: {market_cap_path}")


if __name__ == "__main__":
    main()
