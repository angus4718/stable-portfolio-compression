from pathlib import Path
import sys
import json
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from spc.backtest import StrategyBacktester


def main():
    cfg_path = _ROOT / "analysis_scripts" / "strategy_backtest_config.json"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    basis_path = _ROOT / cfg["input"]["basis"]["value"]
    prices_path = _ROOT / cfg["input"]["prices"]["value"]
    weights_path = _ROOT / cfg["input"]["weights"]["value"]
    tail_percent = float(
        cfg.get("params", {}).get("tail_percent", {}).get("value", 20.0)
    )

    basis_df = pd.read_csv(basis_path, index_col=0, parse_dates=True)
    prices_df = pd.read_csv(prices_path, index_col=0, parse_dates=True)
    weights_df = pd.read_csv(weights_path, index_col=0, parse_dates=True)

    coeffs_path = (
        _ROOT / "analysis_outputs" / "coefficients_ridge_timeseries_by_date.csv"
    )
    coeffs_df = pd.read_csv(coeffs_path) if coeffs_path.exists() else None

    sb = StrategyBacktester(
        prices_df=prices_df,
        weights_df=weights_df,
        basis_df=basis_df,
        coeffs_ts=coeffs_df,
    )

    turnover_lambda = cfg.get("params", {}).get("turnover_lambda", {}).get("value", 0.0)
    try:
        turnover_lambda = float(turnover_lambda)
    except Exception:
        turnover_lambda = 0.0

    print("Running backtest...")
    summary = sb.run_analysis(
        turnover_lambda=turnover_lambda,
        tail_percent=tail_percent,
        out_prefix="analysis_backtest",
    )

    print("Backtest summary:")
    for k, v in summary.items():
        print(f" - {k}: {v}")
    print(
        f"Timeseries saved to: {_ROOT / 'backtest' / 'analysis_backtest_timeseries.csv'}"
    )
    print(f"Summary saved to: {_ROOT / 'backtest' / 'analysis_backtest_summary.json'}")


if __name__ == "__main__":
    main()
