"""
Backtest replication of the market index using basis weights.

Computes index returns and replicated returns and reports tracking performance
over the last `tail_percent` of dates (monthly data). Uses either time-series
coefficients (if present) to compute date-varying basis weights or falls back
to a snapshot coefficients matrix.
"""

from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from spc.backtest import StrategyBacktester


def main():
    tail_percent = 50.0
    if len(sys.argv) > 1:
        try:
            tail_percent = float(sys.argv[1])
        except Exception:
            pass

    bt = StrategyBacktester()
    bt.run(tail_percent=tail_percent)


if __name__ == "__main__":
    main()
