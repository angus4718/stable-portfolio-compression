"""Small shared utilities for SPC scripts and modules."""

from typing import Any
from pathlib import Path
import pandas as pd


def cfg_val(config: dict, section: str, key: str, default: Any = None) -> Any:
    """
    Retrieve the value of a specific key within a section of a configuration dictionary.
    It looks for the value at `config[section][key]['value']`. If any part of the path is
    missing, the function returns the specified default value.
    Args:
        config (dict): The configuration dictionary to search.
        section (str): The top-level key in the configuration dictionary.
        key (str): The second-level key within the specified section.
        default (Any, optional): The value to return if the specified path does not exist.
            Defaults to None.
    Returns:
        Any: The value found at `config[section][key]['value']`, or the default value
        if the path does not exist.
    """
    return config.get(section, {}).get(key, {}).get("value", default)


def load_basis_list_from_path(basis_path: Path) -> list[str]:
    """
    Read a CSV file containing a basis list and return a list of tickers.
    This function reads a CSV file specified by the `basis_path` parameter and
    extracts a list of tickers. The CSV file can either have a column named
    `ticker` or be a single-column CSV. If the `ticker` column exists, its
    values are returned as a list of strings. Otherwise, the values from the
    first column are returned as strings.
    Args:
        basis_path (Path): The file path to the CSV file containing the basis list.
    Returns:
        list[str]: A list of tickers extracted from the CSV file.
    """
    basis_df = pd.read_csv(basis_path)
    if "ticker" in basis_df.columns:
        return basis_df["ticker"].astype(str).tolist()
    return basis_df.iloc[:, 0].astype(str).tolist()


def resolve_output_and_input_paths(
    config: dict, root: Path
) -> tuple[Path, Path, Path, Path, Path]:
    """
    Resolves and returns the output and input file paths based on the provided configuration and root directory.
    Args:
        config (dict): A dictionary containing configuration values for input and output paths.
        root (Path): The root directory to resolve relative paths.
    Returns:
        tuple[Path, Path, Path, Path, Path]: A tuple containing:
            - outputs_dir (Path): The directory where output files will be stored.
            - basis_path (Path): The resolved path for the basis file.
            - coeffs_path (Path): The path for the coefficients file.
            - coeffs_ts_path (Path): The path for the coefficients timeseries file.
            - weights_path (Path): The resolved path for the weights file.
    Notes:
        - If the specified basis or weights file does not exist at the given path, an alternative path is checked.
        - The `outputs` directory is created if it does not already exist.
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
    """
    Load a CSV file containing price data into a pandas DataFrame.

    The CSV file is expected to have its first column as the index, which will
    be parsed as dates. The resulting DataFrame will be sorted by its index.

    Args:
        path (Path): The file path to the CSV file.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the loaded and sorted data.
    """
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df.sort_index()


def load_weights_csv(path: Path) -> pd.DataFrame:
    """
    Load a CSV file containing weights into a pandas DataFrame.

    This function reads a CSV file from the specified path, using the first column
    as the index and parsing dates where applicable. The resulting DataFrame is
    sorted by its index before being returned.

    Args:
        path (Path): The file path to the CSV file.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the data from the CSV file,
        sorted by index.
    """
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df.sort_index()
