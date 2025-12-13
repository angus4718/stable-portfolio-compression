from pathlib import Path
import json
import pandas as pd
import sys

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from spc.portfolio_construction import WeightMapper
from spc.utils import cfg_val, load_weights_csv


def main():
    out_dir = _ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    # Load configuration
    config_path = _ROOT / "scripts" / "weight_mapping_config.json"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    coeffs_cfg = cfg_val(
        cfg,
        "input",
        "coefficients_pivot_path",
        out_dir / "coefficients_ridge_panel.csv",
    )
    coeffs_pivot_path = Path(coeffs_cfg)

    weights_cfg = cfg_val(
        cfg, "input", "weights_path", "synthetic_data/market_index_weights.csv"
    )
    weights_path = Path(weights_cfg)

    basis_cfg = cfg_val(cfg, "output", "basis_path", "outputs/basis_selected.csv")
    basis_path = Path(basis_cfg)
    if not basis_path.is_absolute():
        basis_path = (_ROOT / basis_path).resolve()

    turnover_lambda = float(cfg_val(cfg, "mapping", "turnover_lambda", 0.01))
    prev_weights_cfg = cfg_val(cfg, "mapping", "previous_weights_path", None)

    prev_series = None
    if prev_weights_cfg:
        prev_path = Path(prev_weights_cfg)
        prev_df = pd.read_csv(prev_path, index_col=0)
        prev_series = prev_df.iloc[-1]

    # Load weights timeseries and take last row
    w_df = load_weights_csv(weights_path)
    last_row = w_df.iloc[[-1]].fillna(0.0)

    mapper = WeightMapper(
        basis_path=basis_path,
        coeffs_ts_path=coeffs_pivot_path,
        out_dir=out_dir,
        weights_df=last_row,
        turnover_lambda=turnover_lambda,
        prev_basis_weights=prev_series,
    )

    mapper.run()


if __name__ == "__main__":
    main()
