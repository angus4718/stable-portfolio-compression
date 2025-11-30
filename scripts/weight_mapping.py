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
import json
import pandas as pd
import numpy as np
import sys

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def load_config() -> dict:
    cfg_path = _ROOT / "scripts" / "basis_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    cfg = load_config()

    outputs_dir = _ROOT / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    basis_cfg_val = (
        cfg.get("output", {})
        .get("basis_path", {})
        .get("value", "outputs/basis_selected.csv")
    )
    basis_path = Path(basis_cfg_val)
    if not basis_path.exists():
        alt = outputs_dir / basis_path.name
        if alt.exists():
            basis_path = alt

    coeffs_path = outputs_dir / "coefficients_ridge.csv"
    coeffs_ts_path = outputs_dir / "coefficients_ridge_timeseries.csv"
    weights_path = Path(
        cfg.get("input", {})
        .get("weights_path", {})
        .get("value", "synthetic_data/market_index_weights.csv")
    )

    print(f"Loading basis list from: {basis_path}")
    basis_df = pd.read_csv(basis_path)
    basis_list = (
        basis_df["ticker"].astype(str).tolist()
        if "ticker" in basis_df.columns
        else basis_df.iloc[:, 0].astype(str).tolist()
    )

    coeffs = None
    coeffs_ts = None
    if coeffs_ts_path.exists():
        print(f"Loading time-series coefficients from: {coeffs_ts_path}")
        coeffs_ts = pd.read_csv(coeffs_ts_path, parse_dates=["date"])
    elif coeffs_path.exists():
        print(f"Loading coefficient snapshot from: {coeffs_path}")
        coeffs = pd.read_csv(coeffs_path, index_col=0)
    else:
        raise FileNotFoundError(
            f"Coefficients file not found: {coeffs_ts_path} or {coeffs_path}"
        )

    print(f"Loading index weights from: {weights_path}")
    if weights_path.exists():
        w_spx = pd.read_csv(weights_path, index_col=0, parse_dates=True)
    else:
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    universe = [str(c) for c in w_spx.columns.tolist()]
    w_spx_aligned = w_spx.reindex(columns=universe).fillna(0.0)

    if coeffs_ts is not None:
        print("Using time-series coefficients to compute date-varying mapping.")
        # Normalize date column to timestamps and sort
        coeffs_ts["date"] = pd.to_datetime(coeffs_ts["date"])
        coeffs_ts = coeffs_ts.sort_values("date")
        # Build a dict mapping date -> pivoted DataFrame (asset x basis)
        coeffs_by_date = {}
        for d, grp in coeffs_ts.groupby("date"):
            pivot = grp.pivot(index="asset", columns="basis", values="coef").fillna(0.0)
            # ensure columns cover all basis_list in correct order
            pivot = pivot.reindex(columns=basis_list, fill_value=0.0)
            coeffs_by_date[pd.Timestamp(d)] = pivot

        coeff_dates = sorted(coeffs_by_date.keys())

        basis_rows = []
        # For each weights date choose the most recent coefficient date (carry-forward).
        for w_date in w_spx_aligned.index:
            # find latest coeff date <= w_date
            sel_date = None
            for cd in coeff_dates:
                if cd <= pd.Timestamp(w_date):
                    sel_date = cd
                else:
                    break

            # Build mapping A_date for this w_date
            A_date = pd.DataFrame(0.0, index=universe, columns=basis_list, dtype=float)
            # identity for basis tickers
            for b in basis_list:
                if b in A_date.index and b in A_date.columns:
                    A_date.at[b, b] = 1.0

            if sel_date is not None:
                pivot = coeffs_by_date[sel_date]
                # fill rows for assets present in pivot
                for asset in pivot.index.astype(str):
                    if asset not in A_date.index:
                        continue
                    # assign coefficient row values aligned to basis_list
                    for b in basis_list:
                        A_date.at[asset, b] = (
                            float(pivot.at[asset, b]) if b in pivot.columns else 0.0
                        )

            # Normalize rows with positive sum
            row_sums_A = A_date.sum(axis=1)
            positive_rows = row_sums_A > 0.0
            if positive_rows.any():
                A_date.loc[positive_rows] = A_date.loc[positive_rows].div(
                    row_sums_A[positive_rows], axis=0
                )

            # Compute basis weights for this date
            w_row = w_spx_aligned.loc[w_date].values.reshape(1, -1)
            basis_w_row = pd.Series(
                (w_row @ A_date.values).flatten(), index=basis_list, name=w_date
            )
            basis_rows.append(basis_w_row)

        if basis_rows:
            basis_weights = pd.DataFrame(basis_rows)
            basis_weights.index = w_spx_aligned.index
        else:
            basis_weights = pd.DataFrame(columns=basis_list, index=w_spx_aligned.index)

    else:
        # Snapshot coefficients (wide format)
        A = pd.DataFrame(0.0, index=universe, columns=basis_list, dtype=float)
        for b in basis_list:
            if b in A.index and b in A.columns:
                A.at[b, b] = 1.0

        # Fill rows from coefficients where available
        for asset in coeffs.index.astype(str):
            if asset not in A.index:
                continue
            for col in coeffs.columns:
                col_str = str(col)
                if col_str in A.columns:
                    A.at[asset, col_str] = float(coeffs.loc[asset, col])

        # Normalize mapping A rows so that each asset's coefficients sum to 1 where possible.
        row_sums_A = A.sum(axis=1)
        positive_rows = row_sums_A != 0.0
        A.loc[positive_rows] = A.loc[positive_rows].div(
            row_sums_A[positive_rows], axis=0
        )

        basis_weights = w_spx_aligned.dot(A)

    out_dir = _ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Validate that basis weights sum to 1 per date (row)
    tol = 1e-6
    row_sums = basis_weights.sum(axis=1)
    is_unit = row_sums.apply(lambda x: np.isclose(x, 1.0, atol=tol))
    n_rows = len(row_sums)
    n_unit = int(is_unit.sum())
    n_zero = int((row_sums == 0.0).sum())

    if n_unit == n_rows and n_zero == 0:
        out_path = out_dir / "basis_weights.csv"
        basis_weights.to_csv(out_path, index=True)
        print(
            f"Saved basis weights to: {out_path} (all rows sum to 1 within tol={tol})"
        )
    else:
        print(
            f"Warning: basis weight rows summing to 1: {n_unit}/{n_rows}; zero-sum rows: {n_zero}"
        )
        print(
            f"Row sums sample (min,max,median): {row_sums.min():.6f}, {row_sums.max():.6f}, {row_sums.median():.6f}"
        )

        raw_out = out_dir / "basis_weights_raw.csv"
        basis_weights.to_csv(raw_out, index=True)
        print(f"Saved raw basis weights to: {raw_out}")

        normalized = basis_weights.copy()
        positive = row_sums > 0.0
        normalized.loc[positive] = normalized.loc[positive].div(
            row_sums[positive], axis=0
        )

        norm_out = out_dir / "basis_weights.csv"
        normalized.to_csv(norm_out, index=True)
        print(
            f"Saved normalized basis weights to: {norm_out} (rows with sum>0 scaled to sum=1)"
        )

        basis_weights = normalized

    # Select only the final (latest-month) basis weights and save that single row.
    print("Selecting final (latest-month) basis weights only...")
    if basis_weights.empty:
        print("No basis weights computed (empty). Writing empty file.")
        out_path = out_dir / "basis_weights.csv"
        basis_weights.to_csv(out_path, index=True)
    else:
        # prefer the most recent row with non-zero sum; fallback to last row
        row_sums = basis_weights.sum(axis=1)
        nonzero_idx = row_sums[row_sums != 0.0].index
        if len(nonzero_idx) > 0:
            sel_idx = nonzero_idx[-1]
        else:
            sel_idx = basis_weights.index[-1]

        latest_row = basis_weights.loc[sel_idx]
        latest_df = latest_row.to_frame().T
        # ensure index is the date string
        latest_df.index = [str(sel_idx)]
        out_path = out_dir / "basis_weights.csv"
        latest_df.to_csv(out_path, index=True)
        print(f"Saved single-row basis weights for date {sel_idx} to: {out_path}")
        print("Top basis weights:")
        print(latest_row.sort_values(ascending=False).head(20))


if __name__ == "__main__":
    main()
