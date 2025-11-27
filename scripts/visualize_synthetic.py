"""Visualize synthetic market data.

Reads CSV outputs from the synthetic-data generator and produces PNG plots
into a `plots` subdirectory. Plots include:
- Sample asset prices time series (top-N by market cap at final date)
- Market index weights over time (stacked area for top-N)
- Market-cap time series for top-N assets
- Concentration (top-k weight share) over time
- Number of active constituents over time
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def load_data(data_dir: Path):
    prices_path = data_dir / "prices_monthly.csv"
    weights_path = data_dir / "market_index_weights.csv"
    caps_path = data_dir / "market_cap_values.csv"

    missing = [p for p in (prices_path, weights_path, caps_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing files: {missing}")

    prices = pd.read_csv(prices_path, parse_dates=[0], index_col=0)
    weights = pd.read_csv(weights_path, parse_dates=[0], index_col=0)
    caps = pd.read_csv(caps_path, parse_dates=[0], index_col=0)

    prices.index = pd.to_datetime(prices.index)
    weights.index = pd.to_datetime(weights.index)
    caps.index = pd.to_datetime(caps.index)

    return prices, weights, caps


def ensure_plot_dir(data_dir: Path):
    out = data_dir / "plots"
    out.mkdir(parents=True, exist_ok=True)
    return out


def plot_price_samples(
    prices: pd.DataFrame, market_caps: pd.DataFrame, out_dir: Path, top_n: int = 12
):
    # Choose top_n by final market cap
    final_caps = market_caps.iloc[-1].dropna()
    top = final_caps.sort_values(ascending=False).head(top_n).index.tolist()
    # plotting style is set centrally (see `main`); fall back if missing
    fig, ax = plt.subplots(figsize=(12, 6))
    prices[top].plot(ax=ax, linewidth=1.5)
    ax.set_title(f"Sample Prices — Top {top_n} by Final Market Cap")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.tight_layout()
    p = out_dir / f"prices_top_{top_n}.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)


def plot_market_weights(weights: pd.DataFrame, out_dir: Path, top_n: int = 12):
    # Plot stacked area chart for top_n names by average weight
    avg_w = weights.mean(axis=0).dropna()
    top = avg_w.sort_values(ascending=False).head(top_n).index.tolist()

    fig, ax = plt.subplots(figsize=(12, 6))
    stacked = weights[top].fillna(0)
    # If any small negative rounding errors exist, clip them to zero for plotting.
    if (stacked < 0).any().any():
        print(
            "Note: negative weight values detected — clipping negatives to zero for plotting."
        )
    stacked = stacked.clip(lower=0.0)

    # Sanity check: top-N contributions should be <= 1. Warn if not (numeric issues).
    top_sums = stacked.sum(axis=1)
    if (top_sums > 1.000001).any():
        print(
            "WARNING: top-N weights exceed 1.0 for some rows — this indicates inconsistent weights in the dataset."
        )

    stacked.plot.area(ax=ax, linewidth=0)
    ax.set_title(f"Market Weights (Stacked) — Top {top_n} by Average Weight")
    ax.set_xlabel("Date")
    ax.set_ylabel("Weight")
    ax.set_ylim(0, 1)
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.tight_layout()
    p = out_dir / f"weights_stacked_top_{top_n}.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)


def plot_market_caps(caps: pd.DataFrame, out_dir: Path, top_n: int = 12):
    final_caps = caps.iloc[-1].dropna()
    top = final_caps.sort_values(ascending=False).head(top_n).index.tolist()

    fig, ax = plt.subplots(figsize=(12, 6))
    (caps[top] / 1e6).plot(ax=ax)  # show millions
    ax.set_title(f"Market Caps (Millions) — Top {top_n} by Final Cap")
    ax.set_xlabel("Date")
    ax.set_ylabel("Market Cap (M)")
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.tight_layout()
    p = out_dir / f"market_caps_top_{top_n}.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)


def plot_initial_marketcap_distribution(caps: pd.DataFrame, out_dir: Path):
    """Plot the distribution of market caps at the first date of the simulation.

    The histogram uses a log x-axis to display heavy-tailed behavior more clearly.
    """
    if caps.shape[0] == 0:
        return

    initial_caps = caps.iloc[0].dropna()
    if initial_caps.empty:
        return

    # Prepare data in millions
    vals = initial_caps.values.astype(float) / 1e6
    # Single-panel histogram (log-x) showing initial market-cap distribution
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(vals, bins=60, color="C2", alpha=0.85)
    ax.set_xscale("log")
    ax.set_title("Initial Market-Cap Distribution (log scale, Millions)")
    ax.set_xlabel("Market Cap (M, log scale)")
    ax.set_ylabel("Count")

    fig.tight_layout()
    p = out_dir / "initial_marketcap_distribution.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)


def plot_concentration(weights: pd.DataFrame, out_dir: Path, top_k: int = 5):
    # Sum of top_k weights over time
    topk_share = weights.apply(
        lambda row: row.dropna().sort_values(ascending=False).head(top_k).sum(), axis=1
    )

    fig, ax = plt.subplots(figsize=(10, 4))
    topk_share.plot(ax=ax)
    ax.set_title(f"Concentration: Sum of Top {top_k} Weights Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Top-k Weight Share")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    p = out_dir / f"concentration_top_{top_k}.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)


def plot_constituent_counts(weights: pd.DataFrame, out_dir: Path):
    active_counts = weights.notna().sum(axis=1)

    fig, ax = plt.subplots(figsize=(10, 4))
    active_counts.plot(ax=ax)
    ax.set_title("Number of Active Constituents Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Active Count")
    fig.tight_layout()
    p = out_dir / "active_constituents_over_time.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)


def plot_correlation_heatmap(prices: pd.DataFrame, out_dir: Path, top_n: int = 12):
    """Compute monthly returns and plot correlation heatmap for top-N tickers."""
    returns = prices.pct_change(fill_method=None).dropna(how="all")
    # choose top_n by final price availability (or by non-NaN count)
    avail_counts = prices.notna().sum(axis=0)
    top = avail_counts.sort_values(ascending=False).head(top_n).index.tolist()

    # pick returns for top tickers and drop columns with all NaN
    top_returns = returns[top].dropna(how="all", axis=1).fillna(0)
    if top_returns.shape[1] < 2:
        return

    corr = top_returns.corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr, ax=ax, cmap="vlag", center=0, square=True, cbar_kws={"shrink": 0.8}
    )
    ax.set_title(f"Return Correlation Heatmap — Top {len(corr)} Assets")
    fig.tight_layout()
    p = out_dir / f"correlation_heatmap_top_{len(corr)}.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)


def plot_returns_distribution(prices: pd.DataFrame, out_dir: Path, top_n: int = 12):
    """Plot histogram + KDE of monthly returns for the universe and top-N."""
    returns = prices.pct_change(fill_method=None).dropna(how="all")
    all_returns = returns.stack().dropna()

    # Top-N by availability
    avail_counts = prices.notna().sum(axis=0)
    top = avail_counts.sort_values(ascending=False).head(top_n).index.tolist()
    top_returns = returns[top].stack().dropna()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(all_returns, bins=80, kde=True, ax=axes[0], color="C0")
    axes[0].set_title("All Assets: Monthly Return Distribution")
    axes[0].set_xlabel("Monthly Return")

    sns.histplot(top_returns, bins=60, kde=True, ax=axes[1], color="C1")
    axes[1].set_title(f"Top {top_n} Assets: Monthly Return Distribution")
    axes[1].set_xlabel("Monthly Return")

    fig.tight_layout()
    p = out_dir / f"returns_distribution_top_{top_n}.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)


def plot_cumulative_returns(
    market_caps: pd.DataFrame, prices: pd.DataFrame, out_dir: Path, top_n: int = 12
):
    """Plot cumulative index value (market-cap weighted total) and top-N cumulative series."""
    # Index total market cap over time
    total_mc = market_caps.sum(axis=1)
    total_mc = total_mc.ffill().bfill()
    index_level = total_mc / total_mc.iloc[0]

    # Top-N by final cap
    final_caps = market_caps.iloc[-1].dropna()
    top = final_caps.sort_values(ascending=False).head(top_n).index.tolist()

    # For top assets compute cumulative returns from prices
    prices_top = prices[top].copy()
    prices_top = prices_top.ffill()
    cum_top = prices_top.divide(prices_top.iloc[0]).ffill()

    fig, ax = plt.subplots(figsize=(12, 6))
    index_level.plot(ax=ax, label="Index (MC-weighted)", linewidth=2, color="black")
    # plot top series using pandas to avoid mixed converters
    cum_top.plot(ax=ax, alpha=0.9, linewidth=1)

    ax.set_title("Cumulative Growth: Index and Top Assets")
    ax.set_xlabel("Date")
    ax.set_ylabel("Index / Price (Normalized)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    p = out_dir / "cumulative_returns_index_top.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)


def plot_turnover(weights: pd.DataFrame, out_dir: Path):
    """Compute simple monthly turnover (fraction of constituents replaced)."""
    dates = weights.index
    # Count-based (set membership) turnover
    active_sets = [set(weights.columns[weights.loc[d].notna()]) for d in dates]
    count_turnovers = []
    for i in range(1, len(active_sets)):
        prev = active_sets[i - 1]
        cur = active_sets[i]
        if len(prev) == 0:
            count_turnovers.append(np.nan)
        else:
            kept = len(prev.intersection(cur))
            turnover = 1.0 - (kept / float(len(prev)))
            count_turnovers.append(turnover)

    turnover_series = pd.Series([np.nan] + count_turnovers, index=dates)

    fig, ax = plt.subplots(figsize=(10, 4))
    turnover_series.plot(ax=ax, label="Count-based Turnover", color="C0")
    ax.set_title("Monthly Turnover (fraction replaced)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Turnover")

    # Scale y-axis based on the plotted line, always starting at 0 and capped at 1.0
    try:
        max_val = float(turnover_series.max(skipna=True))
        upper = min(1.0, max(0.01, max_val * 1.05))
    except Exception:
        upper = 1.0

    ax.set_ylim(0, upper)
    ax.legend()
    fig.tight_layout()
    p = out_dir / "monthly_turnover.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)


def main():
    """Run visualizations."""
    data_dir = Path("synthetic_data")
    top_n = 12
    top_k = 5

    prices, weights, caps = load_data(data_dir)
    out_dir = ensure_plot_dir(data_dir)

    sns.set_theme(style="darkgrid")

    print(f"Saving plots to: {out_dir}")

    # Initial diagnostics: market-cap distribution at simulation start
    plot_initial_marketcap_distribution(caps, out_dir)

    plot_price_samples(prices, caps, out_dir, top_n=top_n)
    plot_market_weights(weights, out_dir, top_n=top_n)
    plot_market_caps(caps, out_dir, top_n=top_n)
    plot_concentration(weights, out_dir, top_k=top_k)
    plot_constituent_counts(weights, out_dir)
    # Additional diagnostics
    plot_correlation_heatmap(prices, out_dir, top_n=top_n)
    plot_returns_distribution(prices, out_dir, top_n=top_n)
    plot_cumulative_returns(caps, prices, out_dir, top_n=top_n)
    plot_turnover(weights, out_dir)

    print("Plots saved.")


if __name__ == "__main__":
    main()
