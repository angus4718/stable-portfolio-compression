"""Compute basis weights from index weights using Ridge coefficients.

Reads:
- Basis list: `outputs/basis_selected.csv` (or path set in `scripts/basis_config.json`).
- Coefficients: `outputs/coefficients_ridge_timeseries.csv` (preferred) or
  `outputs/coefficients_ridge.csv` (snapshot).
- Index weights: `synthetic_data/market_index_weights.csv` (or configured path).

Behavior:
- If time-series coefficients are present, the script carries forward the most
  recent coefficient date for each weights date and computes date-varying
  basis weights by applying the asset->basis mapping for that date.
- If only a coefficient snapshot is present, the script applies the snapshot
  mapping to the index weights to produce basis weights.
- Rows are normalized so each date's basis weights sum to 1 where possible.

Outputs (written to `outputs/`):
- `basis_weights.csv` — final output (by default a single-row latest-month
  basis-weight snapshot). When needed, a normalized `basis_weights.csv` and a
  raw `basis_weights_raw.csv` are also written for inspection.
"""

from pathlib import Path
import sys
import json

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from spc.portfolio_construction import WeightMapper


def load_config() -> dict:
    cfg_path = _ROOT / "scripts" / "basis_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    cfg = load_config()
    mapper = WeightMapper(cfg)
    mapper.run()


if __name__ == "__main__":
    main()
