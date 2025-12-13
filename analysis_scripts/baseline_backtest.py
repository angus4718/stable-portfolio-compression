from pathlib import Path
import sys
import json
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from spc.backtest import BaselineBacktester


def compute_topx_weights_time_series(w_spx: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """Construct top-N replicate weights per date from index weights DataFrame.

    Args:
        w_spx: DataFrame of index weights indexed by date with asset columns.
        top_n: number of top constituents to include each date.

    Returns:
        DataFrame of replicate weights indexed by date with columns equal to the
        universe (re-normalized so top-N constituents sum to 1, others are 0).
    """
    rows = []
    for dt in w_spx.index:
        w_row = w_spx.loc[dt]
        top = w_row.nlargest(top_n)
        normalized = top / top.sum()
        rows.append(normalized.reindex(w_spx.columns, fill_value=0.0))

    return pd.DataFrame(rows, index=w_spx.index)


def main():
    cfg_path = _ROOT / "analysis_scripts" / "baseline_backtest_config.json"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    top_n = int(cfg.get("params", {}).get("top_n", {}).get("value", 10))
    tail_percent = float(
        cfg.get("params", {}).get("tail_percent", {}).get("value", 20.0)
    )

    weights_path = _ROOT / cfg["input"]["weights"]["value"]
    prices_path = _ROOT / cfg["input"]["prices"]["value"]

    weights_df = pd.read_csv(weights_path, index_col=0, parse_dates=True)
    prices_df = pd.read_csv(prices_path, index_col=0, parse_dates=True)

    bb = BaselineBacktester(prices_df=prices_df, weights_df=weights_df)
    summary = bb.run_analysis(
        top_n=top_n, tail_percent=tail_percent, out_prefix="baseline_analysis"
    )
    print("Baseline backtest summary (tail):")
    for k, v in summary.items():
        print(f" - {k}: {v}")


if __name__ == "__main__":
    main()
