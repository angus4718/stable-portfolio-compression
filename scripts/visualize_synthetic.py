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


def main():
    """Run visualizations using simple defaults.

    Configuration precedence (highest -> lowest):
      1. Command-line args via `sys.argv` (positional: data_dir top_n top_k_conc show)
      2. Environment variables: `SYNTH_DATA_DIR`, `SYNTH_TOP_N`, `SYNTH_TOP_K_CONC`, `SYNTH_SHOW`
      3. Defaults hard-coded below.
    """
    data_dir = Path("synthetic_data")
    top_n = 12
    top_k = 5

    prices, weights, caps = load_data(data_dir)
    out_dir = ensure_plot_dir(data_dir)

    sns.set_theme(style="darkgrid")

    print(f"Saving plots to: {out_dir}")

    plot_price_samples(prices, caps, out_dir, top_n=top_n)
    plot_market_weights(weights, out_dir, top_n=top_n)
    plot_market_caps(caps, out_dir, top_n=top_n)
    plot_concentration(weights, out_dir, top_k=top_k)
    plot_constituent_counts(weights, out_dir)

    print("Plots saved.")


if __name__ == "__main__":
    main()
