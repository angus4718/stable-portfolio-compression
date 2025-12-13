# Stable Portfolio Compression (SPC)

Stable Portfolio Compression (SPC) is a Python library and pipeline for reducing a large financial index to a small, representative set of basis assets while preserving index returns and exposure characteristics. This is useful for index tracking funds, portfolio replication, and reducing trading costs.

## Project Summary

**Goal:** Compress a large asset universe into a small number of representative basis assets that capture the behavior of the full index.

**Core Problem:** Many institutional investors hold indices containing hundreds or thousands of assets. Managing this many positions is expensive due to trading costs, data maintenance, and operational overhead. SPC solves this by:
1. Identifying which assets are most representative of the index behavior
2. Computing regression coefficients that map each asset to a small set of basis assets
3. Translating index weights into basis weights that approximate the original exposure
4. Measuring replication quality through tracking error and turnover metrics

**Approach:**
- Compute pairwise correlation distances between all assets using historical returns
- Build a Minimum Spanning Tree (MST) to understand asset relationships and diversity
- Select a compact basis of representative assets (diverse nodes on the tree)
- For each non-basis asset, run expanding-window Ridge regressions using nearby basis assets
- Use fitted coefficients to reconstruct returns and map index weights to basis weights

**Key Outputs:**
- `basis_selected.csv` : List of selected basis assets
- `coefficients_ridge_panel.csv` : Regression coefficients mapping assets → basis
- `recon_returns.csv` : Portfolio returns reconstructed using basis weights
- `regression_errors.csv` : Per-asset fit quality (RMSE, $R^2$)
- `basis_weights.csv` : Final basis weights replicating the index
- Backtest summaries showing tracking error, turnover, and performance

## Finance Background

This section explains the financial concepts required to understand the repository. If you are familiar with portfolio management, correlation, and regression, you can skip this section.

### Asset Returns and Correlations

- **Prices and Returns:** Asset prices are typically measured as closing prices at regular intervals (e.g., monthly). A return is the percentage change in price between two periods. For example, if a stock closes at \$100 one month and \$105 the next, the return is (105-100)/100 = 5%.

- **Correlation:** Correlation measures how strongly two assets' returns move together. A correlation of +1 means they always move together perfectly; 0 means no relationship; -1 means they move in opposite directions. Correlation is useful because assets that are less correlated provide **diversification** - they reduce portfolio risk by not all declining at the same time.

- **Distance Metrics:** SPC converts correlation into a distance metric using the formula: $d = \sqrt{2(1 - \text{correlation})}$. High correlation becomes low distance (assets are "close"); low correlation becomes high distance. This allows us to use graph algorithms to find representative assets.

### Index and Basis Assets

- **Index:** A collection of assets with assigned weights (often market-cap based). For example, the S&P 500 contains 500 stocks with weights proportional to their market capitalization. An index weight tells you what fraction of the portfolio should be held in each asset.

- **Basis Assets:** A small subset of the full index selected to represent the entire index. Instead of holding all 500 S&P stocks, you might hold 50 basis assets chosen to capture the overall behavior. This reduces trading costs and operational complexity.

- **Replication:** Given basis assets and their correlations with non-basis assets, we compute weights for the basis assets that approximate the returns of the original index. If done well, replication is nearly indistinguishable from holding the full index.

### Tree Structure and Asset Selection

- **Minimum Spanning Tree (MST):** A graph connecting all assets with the smallest total distance. The MST structure reveals which assets are most similar and which are most different (outliers). Basis assets are typically selected as diverse nodes on the tree to maximize representation.

- **Local Neighborhoods:** Once the tree is built, nearby assets on the tree have low distance (high correlation). SPC uses these neighborhoods when running regressions - each asset is regressed on nearby basis assets, making the regression more stable and interpretable.

### Ridge Regression and Coefficients

- **Regression:** We model each non-basis asset's returns as a linear combination of basis asset returns plus a residual (error term): $\text{Asset Return} = c_1 \times \text{Basis 1 Return} + c_2 \times \text{Basis 2 Return} + \ldots + \text{error}$. The coefficients ($c_1, c_2, \ldots$) measure how much each basis asset contributes to the non-basis asset's behavior.

- **Ridge Regression:** Standard regression can be unstable when data is noisy or assets are highly correlated. Ridge regression adds a penalty term to make coefficients smaller and more stable. The penalty strength is controlled by `ridge_alpha` parameter.

- **Expanding Window:** SPC uses an expanding-window approach: coefficients are recomputed at each point in time using all historical data up to that point. This captures how relationships between assets evolve over time.

### Performance Metrics

- **Tracking Error:** The difference between the returns of the basis-weighted portfolio and the target index. We measure this as the standard deviation of the differences. Lower tracking error means the basis portfolio replicates the index more faithfully.

- **Turnover:** The amount of rebalancing required to maintain basis weights. High turnover increases trading costs and operational burden. The pipeline measures turnover and aims to keep it low through careful basis selection and weight mapping.

- **Reconstruction $R^2$:** A measure of how well the regression model explains each asset's returns. Higher values (closer to 1) indicate better fit; lower values indicate the basis assets don't fully explain the asset behavior.

## Repository Structure

The project is organized into clear functional layers to separate concerns:

### Core Library (`spc/`)
The main Python package containing reusable classes and functions:

- **`graph.py`** : Asset distance and tree construction
  - Compute pairwise correlation distances
  - PCA denoising and Ledoit-Wolf shrinkage (noise-robust covariance estimation)
  - Minimum Spanning Tree (MST) construction using Kruskal's algorithm
  - Tree utilities (adjacency extraction, neighbor queries)

- **`portfolio_construction.py`** : Main compression pipeline
  - `BasisSelector` : Selects basis assets from the MST using configurable heuristics
  - `LocalRidgeRunner` : Fits Ridge regressions for each asset using its tree neighbors
  - `WeightMapper` : Converts index weights to basis weights using regression coefficients
  
- **`backtest.py`** : Backtesting and performance evaluation
  - `Backtester` : Base class for backtesting logic
  - `StrategyBacktester` : Simulates portfolio performance using basis weights
  - `BaselineBacktester` : Baseline comparison using top-N weighted assets
  
- **`utils.py`** : Utility functions
  - Configuration management and parameter loading
  - CSV file I/O and path resolution
  - Data validation helpers

- **`synthetic_data.py`** : Synthetic data generation (for testing and examples)

### Scripts (`scripts/`)
Runnable entry points for the pipeline. All scripts should be run from the project root directory.

**Pipeline Scripts (in order):**
1. **`generate_synthetic_data.py`** : Creates example price and market-cap data
   - Reads configuration from `synthetic_data_config.json`
   - Outputs: `synthetic_data/prices_monthly.csv`, `market_cap_values.csv`, `market_index_weights.csv`
   
2. **`generate_basis.py`** : Computes distances and selects basis assets
   - Reads: `synthetic_data/prices_monthly.csv`
   - Configuration: `basis_config.json` (distance window, shrinkage method)
   - Outputs: `outputs/basis_selected.csv`
   
3. **`run_local_ridge.py`** : Runs Ridge regression for all assets
   - Reads: `basis_selected.csv`, price data
   - Configuration: `local_ridge_config.json` (ridge_alpha, q_neighbors)
   - Outputs: `coefficients_ridge_panel.csv`, `recon_returns.csv`, `regression_errors.csv`
   
4. **`weight_mapping.py`** : Maps index weights to basis weights
   - Reads: Coefficients, index weights, basis list
   - Outputs: `basis_weights.csv`

**Diagnostic Scripts:**
- **`visualize_synthetic.py`** : Plots synthetic data generation results (price paths, correlations, basic diagnostics)
- **`visualize_hub_branch.py`** : Visualizes the hub-and-branch basis selection on the MST
- **`visualize_max_spread.py`** : Visualizes max-spread basis selection on the MST
- **`visualize_path_betweeness.py`** : Visualizes path betweenness on the MST

### Analysis Scripts (`analysis_scripts/`)
Advanced analysis scripts for time-series basis selection and backtesting. Unlike the main pipeline in `scripts/`, which computes a single basis set using all historical data, these scripts run the full compression workflow **at each time period** to analyze how basis selection and portfolio performance evolve over time.

**Time-Series Pipeline Scripts:**

1. **`generate_basis_timeseries.py`** : Generates basis assets for each date
   - Configuration: `basis_timeseries_config.json`
   - For each date in the price history, computes distances using data up to that date and selects k basis assets
   - Outputs: `analysis_outputs/basis_timeseries.csv` (columns: `date`, `basis`)
   - Key parameters:
     - `k` : Number of basis assets per date (e.g., 10)
     - `method` : Selection method (`max_spread` or `hub_branch`)
     - Distance computation settings (window size, shrinkage method)

2. **`run_local_ridge_timeseries.py`** : Computes regression coefficients over time
   - Configuration: `local_ridge_timeseries_config.json`
   - Reads the per-date basis from step 1
   - For each date, fits Ridge regressions mapping all assets to that date's basis
   - Outputs: `analysis_outputs/coefficients_ridge_timeseries_by_date.csv`, `regression_errors_by_date.csv`
   - Key parameters:
     - `ridge_alpha` : Ridge penalty strength
     - `q_neighbors` : Number of nearby basis assets to use per regression

3. **`weight_mapping_timeseries.py`** : Maps index weights to basis weights over time
   - Configuration: `weight_mapping_timeseries_config.json`
   - Reads coefficients from step 2 and index weights
   - For each date, converts index weights to basis weights using that date's coefficients
   - Outputs: `analysis_outputs/basis_weights_timeseries.csv`
   - Key parameters:
     - `turnover_lambda` : Penalty for portfolio turnover (reduces rebalancing frequency)

4. **`strategy_backtest.py`** : Backtests the time-series basis portfolio
   - Configuration: `strategy_backtest_config.json`
   - Reads basis weights from step 3 and simulates portfolio returns
   - Computes tracking error, turnover, and performance metrics
   - Outputs: `backtest/analysis_backtest_summary.json`, `analysis_backtest_timeseries.csv`

5. **`baseline_backtest.py`** : Backtests a baseline comparison strategy
   - Configuration: `baseline_backtest_config.json`
   - Simple top-N asset strategy (no compression or regression)
   - Used to validate that time-series SPC provides better replication
   - Outputs: `backtest/baseline_analysis_summary.json`, `baseline_analysis_timeseries.csv`

**When to Use Analysis Scripts:**

Use `analysis_scripts/` instead of `scripts/` when you need to:
- Analyze how basis composition changes over time
- Measure out-of-sample performance (each date uses only past data)
- Evaluate portfolio turnover and rebalancing costs realistically
- Study the stability of basis selection across different market regimes
- Compare time-varying vs static basis approaches

The main `scripts/` pipeline uses all available data to select a single basis for the next period. The `analysis_scripts/` pipeline is more realistic for backtesting live trading scenarios where you can only use past data.

**Running the Analysis Pipeline:**

```powershell
# 1. Generate time-series basis (one basis set per date)
python .\analysis_scripts\generate_basis_timeseries.py

# 2. Compute regression coefficients over time
python .\analysis_scripts\run_local_ridge_timeseries.py

# 3. Map weights using time-varying coefficients
python .\analysis_scripts\weight_mapping_timeseries.py

# 4. Run backtests
python .\analysis_scripts\strategy_backtest.py
python .\analysis_scripts\baseline_backtest.py
```

### Input Data (`synthetic_data/`)
Example input files (created by `generate_synthetic_data.py` or provided by users):

- **`prices_monthly.csv`** : Monthly asset prices (rows = time periods, columns = assets). Headers: `date`, `asset_0`, `asset_1`, ..., `asset_N`.
- **`market_cap_values.csv`** : Market capitalization for each asset (used for market-cap weighting).
- **`market_index_weights.csv`** : Index weights at each time period (the target weights to replicate).

### Outputs (`outputs/`)
Pipeline-generated results:

- **`basis_selected.csv`** : List of selected basis assets (Asset IDs).
- **`coefficients_ridge_panel.csv`** : Regression coefficients (rows = time, columns = asset->basis coefficient triplets). Typically shaped as `date`, `asset_id`, `basis_asset_id`, `coefficient`.
- **`recon_returns.csv`** : Reconstructed portfolio returns using basis weights.
- **`regression_errors.csv`** : Per-asset fit diagnostics (Asset ID, RMSE, $R^2$, average coefficient magnitude).
- **`basis_weights.csv`** : Basis weights at each period.

### Configuration (`scripts/`)
- **`basis_config.json`** : Main configuration for distance computation, basis selection, regression, and I/O paths. Parameters:
  - `distance_computation.window_size` : Rolling window for correlation (in months)
  - `distance_computation.shrink_method` : Shrinkage method for covariance (`ledoit_wolf` or `pca`)
  - `basis_selection.target_basis_size` : Desired number of basis assets
  - `regression.ridge_alpha` : Ridge penalty strength
  - `regression.local_window_size` : How many tree neighbors to use in regression
  - File paths: `price_file`, `cap_file`, `weight_file`, `basis_output_file`, etc.

- **`synthetic_data_config.json`** : Controls synthetic data generation (number of sectors, assets per sector, time periods, random seed, etc.).

### Tests (`tests/`)
Comprehensive unit test suite (237 tests total) covering:
- Graph construction and tree utilities
- Basis selection and regression logic
- Weight mapping and backtest computations
- Utility functions and data I/O

All tests pass.

## Installation and Setup

### Prerequisites
- Python 3.8 or later
- pip (Python package manager)

### Installing Dependencies

All required packages are listed in `requirements.txt`. Install them using:

```powershell
python -m pip install --upgrade pip
python -m pip install -r .\requirements.txt
```

## Running the Pipeline

All scripts should be run from the project root directory. The recommended workflow executes the pipeline in order:

### Step 1: Generate Synthetic Data (Optional)

If you don't have your own data, create example prices and weights:

```powershell
python .\scripts\generate_synthetic_data.py
```

**What it does:**
- Creates `synthetic_data/prices_monthly.csv` : 120 months of price data for 100 synthetic assets organized into 10 sectors
- Creates `synthetic_data/market_cap_values.csv` : Market capitalization for each asset
- Creates `synthetic_data/market_index_weights.csv` : Index weights (market-cap normalized) at each period
- Uses seed from `scripts/synthetic_data_config.json` for reproducibility

**To customize:** Edit `scripts/synthetic_data_config.json` to adjust number of assets, sectors, time periods, or random seed.

### Step 2: Generate Basis Assets

Select a representative set of basis assets from all assets:

```powershell
python .\scripts\generate_basis.py
```

**What it does:**
- Reads `synthetic_data/prices_monthly.csv`
- Computes rolling correlation and converts to distance metrics
- Builds a Minimum Spanning Tree (MST) of all assets
- Selects diverse basis assets using tree structure
- Outputs `outputs/basis_selected.csv` (list of selected asset IDs)

**Configuration:** Edit `scripts/basis_config.json`:
- `distance_computation.window_size` : Correlation window in months (e.g., 24)
- `distance_computation.shrink_method` : Use `ledoit_wolf` or `pca` for noise-robust estimates
- `basis_selection.target_basis_size` : Number of basis assets to select (e.g., 20)

**Understanding the output:**
- `basis_selected.csv` lists asset IDs (e.g., `asset_0`, `asset_15`, ...) that form the basis
- These are the only assets you will hold going forward
- All other assets are represented through regression mappings

### Step 3: Run Ridge Regressions

Compute regression coefficients mapping each asset to the basis:

```powershell
python .\scripts\run_local_ridge.py
```

**What it does:**
- Reads basis list and price data
- For each non-basis asset, finds nearby basis assets on the MST
- Fits an expanding-window Ridge regression: Asset returns ~ $c_1 \times$ Basis1 returns + $c_2 \times$ Basis2 returns + ...
- Outputs three files in `outputs/`:
  - `coefficients_ridge_panel.csv` : Coefficients over time (each row = period, columns = coefficients)
  - `recon_returns.csv` : Reconstructed returns using basis weights
  - `regression_errors.csv` : Per-asset fit quality (RMSE, $R^2$, etc.)

**Configuration:** `scripts/local_ridge_config.json`:
- `regression.ridge_alpha` : Ridge penalty strength (higher = more shrinkage, more stable)
- `regression.q_neighbors` : Number of nearby basis assets to use per regression

**Understanding the output:**
- High $R^2$ values (close to 1) indicate the basis explains the asset well
- Low RMSE indicates small residuals (good fit)
- If $R^2$ is low for some assets, they may not be well-explained by the basis

### Step 4: Map Weights

Convert index weights to basis weights:

```powershell
python .\scripts\weight_mapping.py
```

**What it does:**
- Reads: basis list, index weights, regression coefficients
- Computes: basis-level weights that replicate the original index exposure
- Outputs:
  - `basis_weights.csv` : Final basis weights.

**Understanding the output:**
- Each row represents one time period
- Each column is a basis asset
- Values don't necessarily sum to 1; the remaining weight is in cash
- High weight on a basis asset means it represents significant index exposure

**Note:** The main `scripts/` pipeline focuses on basis selection and weight mapping. For backtesting and performance evaluation, see the `analysis_scripts/` folder which includes time-series backtesting with `strategy_backtest.py` and `baseline_backtest.py`.

## Example: Quick Start

Here's a complete example from start to finish (all commands run from project root):

```powershell
# 1. Create synthetic data
$env:PYTHONPATH = "." ; python .\scripts\generate_synthetic_data.py

# 2. Generate basis
python .\scripts\generate_basis.py

# 3. Run regressions to compute coefficients
python .\scripts\run_local_ridge.py

# 4. Map weights
python .\scripts\weight_mapping.py
```

### Using Different Data

Replace the CSVs in `synthetic_data/` with your own:
1. Ensure same column structure
2. Ensure date is parseable
3. Run from Step 2 onwards (no need to re-generate synthetic data)

## Testing and Validation

The project includes a comprehensive test suite to ensure correctness and catch regressions:

### Running Tests

```powershell
# Run all tests
python -m pytest tests/ -v

# Run tests for a specific module
python -m pytest tests/test_graph_all.py -v

# Run a specific test
python -m pytest tests/test_graph_all.py::TestDistanceUtils::test_distance_metric -v

# Show test coverage
python -m pytest tests/ --cov=spc --cov-report=html
```

### Test Organization

- **`test_graph_all.py`** (107 tests) : Tests for distance computation, MST construction, tree utilities
- **`test_portfolio_construction.py`** (56 tests) : Tests for basis selection, regression, weight mapping
- **`test_synthetic_data.py`** (28 tests) : Tests for synthetic data generation
- **`test_utils.py`** (21 tests) : Tests for utilities and I/O
- **`test_backtest.py`** (25 tests) : Tests for backtesting logic

**Total: 237 tests, all passing**

### What Is Tested

- **Graph algorithms:** Correctness of MST, distance metrics, neighbor queries
- **Regression:** Ridge coefficients, expanding window logic, numeric stability
- **Weight mapping:** Correct aggregation and normalization
- **Backtesting:** Return calculations, performance metrics
- **I/O:** CSV parsing, config loading, path resolution
- **Edge cases:** Empty inputs, missing data, single-asset portfolios, numerical boundaries

## Class Hierarchy

```
Backtester (base class)
├── StrategyBacktester (basis-weighted portfolio)
└── BaselineBacktester (top-N asset comparison)

DistancesUtils (static distance/correlation methods)
UnionFind (union-find data structure for Kruskal's algorithm)
MST (minimum spanning tree construction)
TreeUtils (tree traversal and neighbor queries)

BasisSelector (selects basis assets from MST)
LocalRidgeRunner (fits Ridge regressions)
WeightMapper (maps index weights to basis weights)
```