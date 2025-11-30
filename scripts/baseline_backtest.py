"""
Baseline backtest: replicate the index using the top-N weighted stocks.

For each period t, use index weights from t-1 to select the top-N stocks by weight.
The replicated weights are proportional to their index weights among the top-N (i.e. keep
their relative weights and renormalize to sum to 1). Returns at t are computed using
the t-1 replicated weights.
"""

from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from spc.backtest import BaselineBacktester


def main():
    tail_percent = 50.0
    bt = BaselineBacktester()
    bt.load_inputs()
    top_n = len(bt.basis_df)
    bt.run(top_n=top_n, tail_percent=tail_percent)


if __name__ == "__main__":
    main()
