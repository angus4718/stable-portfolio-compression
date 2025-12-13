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
from spc.utils import cfg_val


def _load_config(cfg_path: Path) -> Dict[str, Any]:
    with open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    config_path = _ROOT / "scripts" / "synthetic_data_config.json"
    if not config_path.exists():
        print(f"ERROR: Configuration file not found at {config_path}")
        return

    config = _load_config(config_path)

    # simulation parameters
    n_sectors = int(cfg_val(config, "simulation_parameters", "n_sectors", 10))
    assets_per_sector = int(
        cfg_val(config, "simulation_parameters", "assets_per_sector", 20)
    )
    n_months = int(cfg_val(config, "simulation_parameters", "n_months", 180))
    removal_rate = float(cfg_val(config, "simulation_parameters", "removal_rate", 0.0))
    addition_rate = float(
        cfg_val(config, "simulation_parameters", "addition_rate", 0.0)
    )
    random_seed = int(cfg_val(config, "simulation_parameters", "random_seed", 0))
    index_size = cfg_val(config, "simulation_parameters", "index_size", None)

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

    output_dir = config.get("output_directory", {}).get("value")
    market_cap_params = config.get("market_cap_parameters", {})

    removal_bottom_pct = float(
        cfg_val(
            {"market_cap_parameters": market_cap_params},
            "market_cap_parameters",
            "removal_bottom_pct",
            0.2,
        )
    )
    cap_pareto_alpha_cfg = float(
        cfg_val(
            {"market_cap_parameters": market_cap_params},
            "market_cap_parameters",
            "cap_pareto_alpha",
            1.2,
        )
    )
    cap_scale_cfg = float(
        cfg_val(
            {"market_cap_parameters": market_cap_params},
            "market_cap_parameters",
            "cap_scale",
            50,
        )
    )

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
        cap_pareto_alpha=cap_pareto_alpha_cfg,
        cap_scale=cap_scale_cfg,
    )

    prices_df, _, _ = simulator.simulate_prices()
    weights_df, market_cap_df = simulator.generate_market_weights(prices_df)

    max_stock_weight = float(
        cfg_val(
            {"market_cap_parameters": market_cap_params},
            "market_cap_parameters",
            "max_stock_weight",
            None,
        )
    )
    if max_stock_weight is not None:
        weights_df = simulator.apply_max_stock_weight(weights_df, max_stock_weight)

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
