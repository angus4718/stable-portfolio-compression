# Stable Portfolio Compression (SPC)

Project implementing a tree-based basis selection and a local Ridge-regression pipeline for compressing/approximating index exposures into a small set of representative basis assets. The code provides utilities to compute asset distances from price data, build a minimum-spanning tree (MST), select a compact basis, run expanding-window Ridge regressions to map assets -> basis, and map index weights into basis weights for replication/analysis.

## Project Summary

- **Goal:** Reduce a large index (many assets) to a small number of representative basis assets while preserving exposure and replicating returns as well as possible.
- **Approach:** Compute pairwise correlations and convert them to metric distances; build a tree (MST) and select basis nodes; for each non-basis asset, fit a Ridge regression on local (nearby) basis assets using an expanding historical window; use the fitted coefficients to reconstruct returns and to map index weights into basis weights.
- **Key outputs:** coefficient timeseries, reconstructed returns, per-asset regression diagnostics, and final basis weights.

## Finance Background

This section explains the minimal finance concepts needed to understand what the repository does.

- **Prices and Returns:** Asset price series (e.g., monthly closing prices) are converted to simple period returns via pct_change. Returns are the primary input for computing correlations and fitting regressions.
- **Correlation & Distance:** Correlation measures how two asset returns move together. We convert correlations to distances using d = sqrt(2*(1-rho)), which gives a non-negative metric useful for building graphs and trees.
- **Minimum Spanning Tree (MST):** A tree that connects all assets with the smallest total distance. The MST helps identify the topology of the asset universe and informs selection of diverse basis assets.
- **Basis Assets:** A small set of representative assets selected from the MST (or by other heuristics) intended to span the risk/return space of the full universe.
- **Local Ridge Regression:** For each non-basis asset, we regress its historical returns on a small set of nearby basis asset returns using Ridge (L2-regularized) regression. Coefficients map each asset to the basis and are used to reconstruct returns or translate index weights into basis weights.
- **Replication / Basis Weights:** Given index weights (e.g., market-cap weights), and the asset->basis mapping from the regressions, we compute basis-level weights that approximate the original index exposure.

We evaluate compression using the following metrics:
- **Small tracking error.** Tracking error is the difference between the
  returns of the compressed/replicated portfolio and the target index. We
  aim to minimize tracking error so the basis-weighted portfolio behaves
  similarly to the original index.

- **Low turnover.** Turnover measures how much portfolio weights change
  between periods. Lower turnover reduces transaction costs and operational
  complexity, making the replication more practical for trading.

## Repository structure

This project is organized by clear functional layers: core library code in `spc/`, small runnable helpers in `scripts/`, example inputs in `synthetic_data/`, and generated outputs in `outputs/`. Below are the key folders and the most important files you will use.

- `spc/` - core Python package (library code)
  - `graph.py` : distance/correlation pipeline, PCA denoising, Ledoit-Wolf shrinkage, MST construction, and tree utilities.
  - `portfolio_construction.py` : `BasisSelector`, `LocalRidgeRunner`, and `WeightMapper` classes that implement the main pipeline.
  - `backtest.py` : backtesting utilities used by strategy and baseline backtests.
  - `utils.py` : small helpers for config values, loading CSV inputs, and resolving input/output paths.

- `scripts/` - runnable scripts / small CLIs
  - `generate_synthetic_data.py` : create example price, cap, and weight CSVs under `synthetic_data/` using `scripts/synthetic_data_config.json`.
  - `generate_basis.py` : compute distances and select a small set of basis assets (writes `outputs/basis_selected.csv`).
  - `local_ridge.py` : runs the `LocalRidgeRunner` pipeline to produce `coefficients_ridge_timeseries.csv`, `coefficients_ridge.csv`, `recon_returns.csv`, and `regression_errors.csv` in `outputs/`.
  - `weight_mapping.py` : converts index weights into basis weights using coefficient outputs (writes `basis_weights.csv` / `basis_weights_raw.csv`).
  - `strategy_backtest.py` : run strategy-level backtests that use the computed basis weights to simulate portfolio performance.
  - `baseline_backtest.py` : simple replication backtest that replicates the index using the top-N weighted stocks (baseline comparison).
  - `visualize_synthetic.py` : optional plotting/diagnostics for synthetic data (useful immediately after data generation).

- `synthetic_data/` - example inputs (CSV files)
  - `prices_monthly.csv` : monthly prices used by pipelines and tests.
  - `market_cap_values.csv`, `market_index_weights.csv` : example cap and index weight files used by backtests and mapping.

- `outputs/` - pipeline outputs (created by scripts)
  - `basis_selected.csv` : selected basis asset list.
  - `coefficients_ridge_timeseries.csv` : long-form coefficients (`date, asset, basis, coef`).
  - `coefficients_ridge.csv` : pivoted snapshot of the latest-date coefficients (assets x basis).
  - `recon_returns.csv` : reconstructed returns produced by regressions.
  - `regression_errors.csv` : per-asset RMSE and diagnostics.
  - `basis_weights.csv` / `basis_weights_raw.csv` : mapped basis weights (normalized and raw versions).

- `backtest/` - backtest outputs and saved summaries (CSV/plots)

- `tests/` - unit tests

Notes:
- Configuration: `scripts/basis_config.json` controls algorithmic params (distance window, shrink method, regression settings, paths). The synthetic-data generator reads `scripts/synthetic_data_config.json` for options controlling generated sizes and paths.
- Typical workflow: generate data -> generate basis -> run regressions -> map weights -> run backtests. See the "Running the main scripts" section for the recommended order and commands.

## Setup and dependencies

Install dependencies using the provided `requirements.txt`:

```powershell
python -m pip install --upgrade pip
python -m pip install -r .\requirements.txt
```

## Running the main scripts

All scripts assume you run them from the project root.

Typical (recommended) workflow:

1. Generate synthetic data (or provide your own data in the same format):
```powershell
python .\scripts\generate_synthetic_data.py
```
2. Generate basis assets from the synthetic data:
```powershell
python .\scripts\generate_basis.py
```
3. Run local Ridge regressions to compute coefficients:
```powershell
python .\scripts\local_ridge.py
```
4. Map weights from assets to basis assets:
```powershell
python .\scripts\weight_mapping.py
```
5. Run backtests using the computed basis weights:
```powershell
python .\scripts\strategy_backtest.py
python .\scripts\baseline_backtest.py
```

- Optionally run `scripts/visualize_synthetic.py` after data generation to inspect synthetic inputs and basic diagnostics.

## Configuration

 - Scripts that use configuration read `scripts/basis_config.json` (see that file for parameters like `distance_computation.window_size`, `regression.ridge_alpha`, and I/O file locations) and `scripts/synthetic_data_config.json` (controls options and paths used by the synthetic data generator). If these files are not present, some scripts fall back to sensible defaults and to files under `synthetic_data/` or `outputs/`.