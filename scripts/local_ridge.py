"""
Local Ridge regression using basis assets.

This script reads configuration from `scripts/basis_config.json` and a selected
basis list from `outputs/basis_selected.csv` (or an explicit path set in the
config under `output.basis_path`). It loads monthly prices (default
`synthetic_data/prices_monthly.csv`), computes simple returns, and performs
expanding-window (monthly) Ridge regressions for each non-basis asset using
nearest basis assets (by precomputed distance). The regressions record a
time-series of fitted coefficients (per date) and produce reconstructed
returns and per-asset fit diagnostics.

Key outputs (written to the `outputs/` directory by default):
- `coefficients_ridge_timeseries.csv`: long-format coefficients (date, asset, basis, coef)
- `coefficients_ridge.csv`: latest-date coefficient snapshot (asset x basis)
- `recon_returns.csv`: predicted returns (index = dates, columns = assets)
- `regression_errors.csv`: per-asset RMSE and diagnostics
"""

from __future__ import annotations

from pathlib import Path
import json
import sys

# Ensure project root is on sys.path so local package imports work when running script
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from spc.portfolio_construction import LocalRidgeRunner


def load_config() -> dict:
    cfg_path = _ROOT / "scripts" / "basis_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main():
    config = load_config()
    runner = LocalRidgeRunner(config=config)
    runner.run()


if __name__ == "__main__":
    main()
