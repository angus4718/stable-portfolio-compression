"""
Baseline backtest: replicate the index using the top-X weighted stocks.

For each period t, use index weights from t-1 to select the top-X stocks by weight.
The replicated weights are proportional to their index weights among the top-X (i.e. keep
their relative weights and renormalize to sum to 1). Returns at t are computed using
the t-1 replicated weights (same convention as `backtest.py`).

Usage:
    python scripts/baseline_backtest.py [x_override]

If `x_override` is omitted, the script sets X = number of basis in
`synthetic_data/basis_selected.csv` (so x = strategy basis count).
"""

from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
    
from spc.backtest import BaselineBacktester


def main():
    root = Path(__file__).resolve().parent
    x_override = None
    x_percent = 50.0
    if len(sys.argv) > 1:
        try:
            x_override = int(sys.argv[1])
        except Exception:
            try:
                x_percent = float(sys.argv[1])
            except Exception:
                pass
    if len(sys.argv) > 2:
        try:
            x_percent = float(sys.argv[2])
        except Exception:
            pass

    bt = BaselineBacktester(root)
    bt.load_inputs()
    x = x_override if x_override is not None else len(bt.basis_df)
    if x <= 0:
        print("x must be positive; exiting")
        return
    bt.run(x=x, x_percent=x_percent)


if __name__ == "__main__":
    main()
