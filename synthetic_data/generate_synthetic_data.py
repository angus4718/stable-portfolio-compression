"""
Generate synthetic price and market index weight data with dynamic asset membership.

This script creates two CSV files:
1. prices_monthly.csv: Monthly prices for assets (rows: first day of each month, cols: tickers)
2. market_index_weights.csv: Market cap weights for assets in the index
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
    ):
        """
        Initialize the simulator.

        Args:
            n_sectors: Number of asset sectors
            assets_per_sector: Initial assets per sector
            n_months: Number of months to simulate
            removal_rate: Fraction of assets removed each month
            addition_rate: Fraction of new assets added each month
            random_seed: Random seed for reproducibility
        """
        self.n_sectors = n_sectors
        self.assets_per_sector = assets_per_sector
        self.n_months = n_months
        self.removal_rate = removal_rate
        self.addition_rate = addition_rate
        self.random_seed = random_seed
        # Fixed index size (number of constituents to keep each month)
        # If None, default to full initial universe size
        self.index_size = (
            index_size
            if index_size is not None
            else (self.n_sectors * self.assets_per_sector)
        )

        np.random.seed(random_seed)

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
        """Create initial set of assets."""
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

    def _generate_ticker(self, sector: str, idx: int) -> str:
        """Generate a unique ticker symbol."""
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
        """
        Simulate monthly prices with dynamic asset membership.
        Assets are selected based on market cap ranking each month.

        Returns:
            prices_df: DataFrame with monthly prices (index: date, columns: tickers)
            all_tickers: List of all tickers that existed at any point
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
            self.shares_outstanding[ticker] = np.random.uniform(10, 500)

        # Initialize prices for all assets
        for ticker in self.initial_assets:
            self.prices[ticker] = [100.0]  # Starting price

        # Track all tickers ever created
        all_tickers_set = set(self.initial_assets)
        all_existing_tickers = set(self.initial_assets)

        # Initialize market cap tracking
        self.market_caps = {}
        for ticker in self.initial_assets:
            self.market_caps[ticker] = [
                self.prices[ticker][0] * self.shares_outstanding[ticker]
            ]

        # Target number of assets in index (fixed k)
        target_index_size = int(self.index_size)
        # Determine initial month constituents by top market cap after initialization
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
            # Generate price returns for ALL existing assets (universe)
            returns = self._generate_monthly_returns(all_existing_tickers)

            # Update prices and market caps for all existing tickers
            for ticker in all_existing_tickers:
                ret = returns.get(ticker, 0.0)
                prev_price = self.prices[ticker][-1]
                new_price = prev_price * (1 + ret)
                self.prices[ticker].append(new_price)

                market_cap = new_price * self.shares_outstanding[ticker]
                self.market_caps[ticker].append(market_cap)

            # Randomly remove some assets from the universe based on removal_rate.
            # Removals take effect from the next month (so current month's prices remain).
            n_remove = int(len(all_existing_tickers) * self.removal_rate)
            if n_remove > 0:
                removable = [
                    t
                    for t in all_existing_tickers
                    if self.ticker_history[t]["removed_month"] is None
                ]
                n_remove = min(n_remove, len(removable))
                if n_remove > 0:
                    removed_samples = set(
                        np.random.choice(list(removable), size=n_remove, replace=False)
                    )
                    for rt in removed_samples:
                        # mark removal to take effect next month
                        self.ticker_history[rt]["removed_month"] = month + 1
                        all_existing_tickers.remove(rt)

            # Add new assets occasionally based on addition_rate (fraction of current universe)
            n_new_assets = max(0, int(len(all_existing_tickers) * self.addition_rate))
            for asset_idx in range(n_new_assets):
                sector = np.random.choice(self.sector_names)
                ticker = self._generate_ticker(sector, asset_idx)
                all_existing_tickers.add(ticker)
                all_tickers_set.add(ticker)

                self.shares_outstanding[ticker] = np.random.uniform(10, 500)
                self.prices[ticker] = [100.0]  # Start new assets at 100
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

        # Create DataFrame with aligned indices/columns
        # Include ALL tickers ever created, with NaN for months they didn't exist
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
        """
        Generate correlated monthly returns by sector.

        Args:
            tickers: Set of active tickers

        Returns:
            Dictionary of ticker -> return
        """
        returns = {}

        # Group by sector
        sectors_assets = {}
        for ticker in tickers:
            sector = self.asset_sectors[ticker]
            if sector not in sectors_assets:
                sectors_assets[sector] = []
            sectors_assets[sector].append(ticker)

        # Generate returns per sector
        for sector, sector_tickers in sectors_assets.items():
            n_assets = len(sector_tickers)
            vol = self.sector_volatility[sector]

            # Generate correlated returns
            corr_matrix = self._get_sector_correlation_matrix(sector, n_assets)
            L = np.linalg.cholesky(corr_matrix)
            z = np.random.randn(n_assets)
            sector_returns = (L @ z) * vol

            # Sector drift
            sector_drift = np.random.uniform(-0.015, 0.015)

            for i, ticker in enumerate(sector_tickers):
                returns[ticker] = sector_returns[i] + sector_drift

        return returns

    def generate_market_weights(
        self, prices_df: pd.DataFrame, monthly_tickers: List[Set[str]]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate market cap weights for index constituents.
        Includes all tickers ever created, with NaN for months they didn't exist.

        Args:
            prices_df: DataFrame of prices
            monthly_tickers: List of active tickers per month

        Returns:
            weights_df: DataFrame of market cap weights (index: date, columns: tickers)
            market_cap_df: DataFrame of market cap values
        """
        # Get all tickers from prices_df columns
        all_tickers = list(prices_df.columns)
        dates = prices_df.index
        market_cap_data = []

        for month, date in enumerate(dates):
            month_caps = {}
            for ticker in all_tickers:
                # Check if ticker existed at this month
                created_month = self.ticker_history[ticker]["created_month"]
                removed_month = self.ticker_history[ticker]["removed_month"]

                if created_month <= month and (
                    removed_month is None or month < removed_month
                ):
                    # Ticker existed in this month
                    months_since_creation = month - created_month
                    market_cap = self.market_caps[ticker][months_since_creation]
                    month_caps[ticker] = market_cap
                # else: leave as NaN

            market_cap_data.append(month_caps)

        market_cap_df = pd.DataFrame(market_cap_data, index=dates)
        market_cap_df.index.name = "date"
        market_cap_df = market_cap_df.sort_index(axis=1)

        # Calculate weights (normalize by total market cap of active assets only)
        weights_df = market_cap_df.div(market_cap_df.sum(axis=1), axis=0)

        return weights_df, market_cap_df


def main():
    """Main execution function."""

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
    print(f"  - Sectors: {n_sectors}")
    print(f"  - Assets per sector: {assets_per_sector}")
    print(f"  - Total months: {n_months} ({n_months/12:.1f} years)")
    print(f"  - Monthly removal rate: {removal_rate*100:.1f}%")
    print(f"  - Monthly addition rate: {addition_rate*100:.1f}%")
    print(f"  - Random seed: {random_seed}")
    print(f"  - Output directory: {output_dir}")

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

    # Show asset membership changes
    print("\n" + "=" * 70)
    print("ASSET MEMBERSHIP DYNAMICS")
    print("=" * 70)

    n_assets_per_month = prices_df.notna().sum(axis=1)
    print("\nNumber of active assets per month (first 12 months):")
    for month in range(min(12, len(prices_df))):
        date = prices_df.index[month]
        n_assets = n_assets_per_month.iloc[month]
        print(f"  {date.strftime('%Y-%m')}: {n_assets} assets")

    print("\n" + "=" * 70)
    print("Data generation complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
