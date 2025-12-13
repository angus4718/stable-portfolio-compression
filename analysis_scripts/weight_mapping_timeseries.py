from pathlib import Path
import json
import pandas as pd
import sys
import numpy as np
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from spc.portfolio_construction import WeightMapper


def main():
    out_dir = _ROOT / "analysis_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load config from analysis_scripts to find input/output paths
    config_path = _ROOT / "analysis_scripts" / "weight_mapping_timeseries_config.json"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    coeffs_ts_path = _ROOT / cfg["input"]["coeffs_ts_path"]["value"]
    weights_path = _ROOT / cfg["input"]["weights_path"]["value"]
    basis_path = _ROOT / cfg["output"]["basis_path"]["value"]

    turnover_lambda = float(
        cfg.get("params", {}).get("turnover_lambda", {}).get("value", 0.01)
    )

    mapper = WeightMapper(
        basis_path=basis_path,
        coeffs_ts_path=coeffs_ts_path,
        out_dir=out_dir,
        turnover_lambda=turnover_lambda,
    )

    coeffs_df = pd.read_csv(coeffs_ts_path, parse_dates=["date"])
    coeffs_df["date"] = coeffs_df["date"].dt.normalize()

    w_df = pd.read_csv(weights_path, index_col=0, parse_dates=True)

    all_bases = sorted(coeffs_df["basis"].unique())

    rows = []
    idx = []
    prev_full_row = pd.Series(0.0, index=all_bases)

    # Normalize coefficient dates and iterate over weight dates
    dates = list(w_df.index)
    for date in tqdm(dates, desc="Mapping dates", total=len(dates)):
        t = pd.Timestamp(date).normalize()
        w_row = w_df.loc[date].fillna(0.0)

        # Select coefficients for this date and pivot to asset x basis
        df_date = coeffs_df[coeffs_df["date"] == t]
        if df_date.empty:
            # No coefficients for this date -> zero basis weights (full union)
            rows.append([0.0] * len(all_bases))
            idx.append(date)
            continue

        basis_list_date = sorted(df_date["basis"].unique())
        pivot = df_date.pivot(index="asset", columns="basis", values="coef").fillna(0.0)

        A_date = pd.DataFrame(0.0, index=w_row.index, columns=basis_list_date)

        for b in basis_list_date:
            if b in A_date.index:
                A_date.at[b, b] = 1.0

        for asset in pivot.index:
            if asset in A_date.index:
                for b in pivot.columns:
                    A_date.at[asset, b] = pivot.at[asset, b]

        if turnover_lambda > 0:
            w_prev_arr = np.array([prev_full_row.get(b, 0.0) for b in basis_list_date])
            basis_w_date = mapper.solve_qp_basis_weights(
                A_date, w_row.values, basis_list_date, w_prev_arr, turnover_lambda
            )
        else:
            basis_w_date = pd.Series(
                (w_row.values.reshape(1, -1) @ A_date.values).flatten(),
                index=basis_list_date,
            )

        # Expand into full union of bases (zeros where basis not present this date)
        full_row = pd.Series(0.0, index=all_bases)
        for b in basis_list_date:
            full_row.at[b] = basis_w_date.at[b]

        prev_full_row = full_row.copy()
        rows.append(full_row.values.tolist())
        idx.append(date)

    out_df = pd.DataFrame(rows, index=idx, columns=all_bases)
    out_path = out_dir / "basis_weights_timeseries.csv"
    out_df.to_csv(out_path, index=True)

    print(f"Saved timeseries basis weights to: {out_path}")


if __name__ == "__main__":
    main()
