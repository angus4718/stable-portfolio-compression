import json
from pathlib import Path
import sys
from typing import Dict

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from spc.portfolio_construction import LocalRidgeRunner


def main():
    cfg_path = _ROOT / "scripts" / "local_ridge_config.json"
    with open(cfg_path) as f:
        cfg = json.load(f)

    prices_path = cfg["input"]["prices_path"]["value"]
    basis_path = cfg["output"]["basis_path"]["value"]
    ridge_alpha = float(cfg["regression"]["ridge_alpha"].get("value", 0.0))
    q_neighbors = int(cfg["regression"]["q_neighbors"].get("value", 3))

    dist_conf = cfg["distance_computation"]

    runner = LocalRidgeRunner(
        prices_path=prices_path,
        basis_path=basis_path,
        ridge_alpha=ridge_alpha,
        q_neighbors=q_neighbors,
        min_periods=dist_conf["min_periods"]["value"],
        corr_method=dist_conf["corr_method"]["value"],
        shrink_method=dist_conf["shrink_method"]["value"],
        pca_n_components=dist_conf["pca_n_components"]["value"],
        pca_explained_variance=dist_conf["pca_explained_variance"]["value"],
    )

    print("Running LocalRidgeRunner")
    coef_df, _, _ = runner.run()

    out_dir = _ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    pivot_path = out_dir / "coefficients_ridge_panel.csv"

    coeffs_pivot = coef_df.pivot(index="asset", columns="basis", values="coef").fillna(
        0.0
    )
    coeffs_pivot.to_csv(pivot_path)
    print(f"Wrote pivoted coefficients to: {pivot_path}")


if __name__ == "__main__":
    main()
