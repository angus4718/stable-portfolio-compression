"""Generate synthetic price and market-index weight data with dynamic asset membership."""

import json
from pathlib import Path
from typing import Any, Dict

# Ensure project root is on sys.path so local package imports work when running script
import sys

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from spc.synthetic_data import DynamicIndexSimulator


def _load_config(cfg_path: Path) -> Dict[str, Any]:
    with open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    config_path = _ROOT / "scripts" / "synthetic_data_config.json"
    if not config_path.exists():
        print(f"ERROR: Configuration file not found at {config_path}")
        return

    config = _load_config(config_path)

    sim_params = config["simulation_parameters"]
    n_sectors = sim_params["n_sectors"]["value"]
    assets_per_sector = sim_params["assets_per_sector"]["value"]
    n_months = sim_params["n_months"]["value"]
    removal_rate = sim_params["removal_rate"]["value"]
    addition_rate = sim_params["addition_rate"]["value"]
    random_seed = sim_params["random_seed"]["value"]
    index_size = sim_params.get("index_size", {}).get("value", None)

    sector_volatility = {
        k: v["value"]
        for k, v in config.get("sector_volatility", {}).items()
        if k != "description"
    }
    sector_correlation = {
        k: v["value"]
        for k, v in config.get("sector_correlation", {}).items()
        if k != "description"
    }
    sector_drift_cfg = config.get("sector_drift", {})

    output_dir = config["output_directory"]["value"]
    market_cap_params = config.get("market_cap_parameters", {})

    removal_bottom_pct = float(
        market_cap_params.get("removal_bottom_pct", {}).get("value", 0.2)
    )
    cap_pareto_alpha_cfg = market_cap_params.get("cap_pareto_alpha", {}).get(
        "value", None
    )
    cap_scale_cfg = market_cap_params.get("cap_scale", {}).get("value", None)

    simulator = DynamicIndexSimulator(
        n_sectors=n_sectors,
        assets_per_sector=assets_per_sector,
        n_months=n_months,
        removal_rate=removal_rate,
        addition_rate=addition_rate,
        random_seed=random_seed,
        index_size=index_size,
        removal_bottom_pct=removal_bottom_pct,
        sector_volatility=sector_volatility,
        sector_correlation=sector_correlation,
        sector_drift=sector_drift_cfg,
        cap_pareto_alpha=(
            float(cap_pareto_alpha_cfg) if cap_pareto_alpha_cfg is not None else None
        ),
        cap_scale=(float(cap_scale_cfg) if cap_scale_cfg is not None else None),
    )

    prices_df, all_tickers, monthly_tickers = simulator.simulate_prices()
    weights_df, market_cap_df = simulator.generate_market_weights(
        prices_df, monthly_tickers
    )

    max_stock_weight = market_cap_params.get("max_stock_weight", {}).get("value", None)
    if max_stock_weight is not None:
        weights_df = simulator.apply_max_stock_weight(
            weights_df, float(max_stock_weight)
        )

    out_path = Path(output_dir)
    if not out_path.is_absolute():
        out_path = _ROOT / out_path
    out_path.mkdir(parents=True, exist_ok=True)

    prices_filename = config["output_files"]["prices"]["filename"]
    weights_filename = config["output_files"]["weights"]["filename"]
    market_caps_filename = config["output_files"]["market_caps"]["filename"]

    prices_df.to_csv(out_path / prices_filename)
    weights_df.to_csv(out_path / weights_filename)
    market_cap_df.to_csv(out_path / market_caps_filename)

    print(f"Saved prices -> {out_path / prices_filename}")
    print(f"Saved weights -> {out_path / weights_filename}")
    print(f"Saved market caps -> {out_path / market_caps_filename}")


if __name__ == "__main__":
    main()
