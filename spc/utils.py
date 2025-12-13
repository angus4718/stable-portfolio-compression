"""Small shared utilities for SPC scripts and modules."""

from __future__ import annotations

from typing import Any
from pathlib import Path
import pandas as pd


def cfg_val(config: dict, section: str, key: str, default: Any = None) -> Any:
    """Return a typed config value from a nested config dictionary.

    The configuration format used in this project stores values under
    ``config[section][key]['value']``. This helper safely retrieves that
    nested value and falls back to ``default`` when any part of the path
    is missing.

    Args:
        config (dict): Configuration dictionary.
        section (str): Top-level configuration section.
        key (str): Second-level key within the section.
        default (Any, optional): Value to return when the requested path is
            not present. Defaults to ``None``.

    Returns:
        Any: The value stored at ``config[section][key]['value']`` or
        ``default`` when the path does not exist.
    """
    return config.get(section, {}).get(key, {}).get("value", default)


def load_basis_list_from_path(basis_path: Path) -> list[str]:
    """Load a basis list CSV and return tickers as a list of strings.

    The function supports multiple CSV layouts commonly produced by the
    project's tooling:
        - A CSV with an explicit ``ticker`` column (one ticker per row).
        - A single-column CSV where each row is a ticker.
        - A timeseries CSV with a ``basis`` column where rows are either
          individual tickers or a delimited list (e.g. ``"A;B;C"``) — in
          that case the most recent non-null row is used and split on ``;``
          or ``,``.

    Args:
        basis_path (Path): Path to the basis CSV file.

    Returns:
        list[str]: List of tickers parsed from the file (may be empty).
    """
    basis_df = pd.read_csv(basis_path)
    # If the file contains an explicit 'ticker' column, use it as a simple list
    if "ticker" in basis_df.columns:
        return basis_df["ticker"].astype(str).tolist()

    # If the file is a timeseries with a 'basis' column containing
    # delimited tickers per date (e.g. "A;B;C"), pick the most recent
    # non-null entry and split it into a list. Support ';' or ',' separators.
    if "basis" in basis_df.columns:
        # If the 'basis' column contains multiple rows where each row is a
        # single ticker, treat it as a simple one-per-row list. If the file
        # contains a single row whose value is a delimited list (e.g.
        # "A;B;C"), treat that as a timeseries-style entry and split it.
        non_null = basis_df["basis"].dropna().astype(str)
        if len(non_null) == 0:
            return []

        # If there is more than one non-null row and none look like a
        # delimited list, assume each row is a separate ticker.
        if len(non_null) > 1:
            # If any row contains a delimiter, assume timeseries-of-lists
            if any((";" in v or "," in v) for v in non_null):
                last = non_null.iloc[-1]
                sep = ";" if ";" in last else ","
                items = [s.strip() for s in last.split(sep) if s.strip()]
                return items
            # otherwise return each row as a ticker
            return non_null.tolist()

        # Single non-null row: split if delimited, otherwise return single ticker
        last = non_null.iloc[-1]
        sep = ";" if ";" in last else "," if "," in last else None
        if sep is not None:
            items = [s.strip() for s in last.split(sep) if s.strip()]
            return items
        return [last.strip()] if last.strip() else []

    # Otherwise, assume a single-column CSV where each row is a ticker
    return basis_df.iloc[:, 0].astype(str).tolist()


def resolve_output_and_input_paths(
    config: dict, root: Path
) -> tuple[Path, Path, Path, Path]:
    """Resolve commonly-used input/output paths from a config and root.

    This helper centralizes the project's lightweight path conventions and
    ensures output directories exist.

    Args:
        config (dict): Configuration dictionary with optional keys used by
            ``cfg_val`` (the function expects nested value objects under
            ``['value']``).
        root (Path): Root directory used to resolve relative paths.

    Returns:
        tuple[Path, Path, Path, Path]: ``(outputs_dir, basis_path,
        coeffs_ts_path, weights_path)`` where ``coeffs_ts_path`` is placed
        inside ``outputs_dir`` and other paths are resolved from config
        defaults when not provided.
    """
    outputs_dir = root / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    basis_cfg_val = cfg_val(
        config, "output", "basis_path", "outputs/basis_selected.csv"
    )
    basis_path = Path(basis_cfg_val)

    coeffs_ts_path = outputs_dir / "coefficients_ridge_panel.csv"

    weights_cfg_val = cfg_val(
        config, "input", "weights_path", "synthetic_data/market_index_weights.csv"
    )
    weights_path = Path(weights_cfg_val)

    return outputs_dir, basis_path, coeffs_ts_path, weights_path


def load_prices_csv(path: Path) -> pd.DataFrame:
    """Load a CSV of prices and return a DataFrame indexed by date.

    The CSV is read with the first column used as the index and parsed as
    datetimes. The returned DataFrame is sorted by its index.

    Args:
        path (Path): Path to the CSV file.

    Returns:
        pd.DataFrame: Prices DataFrame indexed by date.
    """
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df.sort_index()


def load_weights_csv(path: Path) -> pd.DataFrame:
    """Load a weights CSV and return a DataFrame indexed by date.

    Reads the CSV using the first column as the index (parsed as datetimes
    when possible) and returns the DataFrame sorted by index.

    Args:
        path (Path): Path to the CSV file.

    Returns:
        pd.DataFrame: Weights DataFrame indexed by date.
    """
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df.sort_index()
