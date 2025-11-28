"""
Backtest replication of the market index using basis weights.

Computes index returns and replicated returns and reports tracking performance
over the last `x_percent` of dates (monthly data). Uses either time-series
coefficients (if present) to compute date-varying basis weights or falls back
to a snapshot coefficients matrix.

Usage:
    python scripts/backtest.py [x_percent]

If `x_percent` is omitted, defaults to 20 (last 20% of dates).
"""

from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from spc.backtest import StrategyBacktester


def main():
    root = Path(__file__).resolve().parent
    x_percent = 50.0
    if len(sys.argv) > 1:
        try:
            x_percent = float(sys.argv[1])
        except Exception:
            pass

    bt = StrategyBacktester(root)
    bt.run(x_percent=x_percent)


if __name__ == "__main__":
    main()
