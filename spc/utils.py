"""Small shared utilities for SPC scripts and modules.

Keep this file minimal — currently contains `cfg_val` for nested config access.
"""

from typing import Any
from pathlib import Path
import pandas as pd


def cfg_val(config: dict, section: str, key: str, default: Any = None) -> Any:
    """Return `config[section][key]['value']` if present, otherwise `default`.

    This avoids repeated nested .get(...) chains across scripts.
    """
    return config.get(section, {}).get(key, {}).get("value", default)


def load_basis_list_from_path(basis_path: Path) -> list[str]:
    """Read a CSV containing a basis list and return a list of tickers.

    Accepts either a CSV with a `ticker` column or a single-column CSV.
    """
    basis_df = pd.read_csv(basis_path)
    if "ticker" in basis_df.columns:
        return basis_df["ticker"].astype(str).tolist()
    return basis_df.iloc[:, 0].astype(str).tolist()


def resolve_output_and_input_paths(
    config: dict, root: Path
) -> tuple[Path, Path, Path, Path, Path]:
    """Resolve and return (outputs_dir, basis_path, coeffs_path, coeffs_ts_path, weights_path).

    - `config` is the parsed JSON config dict.
    - `root` is the project root Path (used to resolve relative fallback locations).
    """
    outputs_dir = root / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    basis_cfg_val = cfg_val(
        config, "output", "basis_path", "outputs/basis_selected.csv"
    )
    basis_path = Path(basis_cfg_val)
    if not basis_path.exists():
        alt = outputs_dir / basis_path.name
        if alt.exists():
            basis_path = alt

    coeffs_path = outputs_dir / "coefficients_ridge.csv"
    coeffs_ts_path = outputs_dir / "coefficients_ridge_timeseries.csv"

    weights_cfg_val = cfg_val(
        config, "input", "weights_path", "synthetic_data/market_index_weights.csv"
    )
    weights_path = Path(weights_cfg_val)
    if not weights_path.exists():
        altw = root / weights_path
        if altw.exists():
            weights_path = altw

    return outputs_dir, basis_path, coeffs_path, coeffs_ts_path, weights_path


def load_prices_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df.sort_index()


def load_weights_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df.sort_index()
