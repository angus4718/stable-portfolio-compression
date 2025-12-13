"""Test suite for spc/utils module.

Test cases for configuration handling, path resolution, and CSV loading utilities.
"""

import unittest
import tempfile
import pandas as pd
import numpy as np
from pathlib import Path

from spc.utils import (
    cfg_val,
    load_basis_list_from_path,
    resolve_output_and_input_paths,
    load_prices_csv,
    load_weights_csv,
)


class TestCfgVal(unittest.TestCase):
    """Test cfg_val configuration retrieval function."""

    def test_cfg_val_basic_retrieval(self):
        """Test basic nested config value retrieval."""
        config = {"section1": {"key1": {"value": "test_value"}}}
        result = cfg_val(config, "section1", "key1")
        self.assertEqual(result, "test_value")

    def test_cfg_val_numeric_value(self):
        """Test retrieving numeric config values."""
        config = {"settings": {"timeout": {"value": 30}}}
        result = cfg_val(config, "settings", "timeout")
        self.assertEqual(result, 30)

    def test_cfg_val_float_value(self):
        """Test retrieving float config values."""
        config = {"params": {"learning_rate": {"value": 0.001}}}
        result = cfg_val(config, "params", "learning_rate")
        self.assertAlmostEqual(result, 0.001)

    def test_cfg_val_list_value(self):
        """Test retrieving list config values."""
        config = {"data": {"columns": {"value": ["A", "B", "C"]}}}
        result = cfg_val(config, "data", "columns")
        self.assertEqual(result, ["A", "B", "C"])

    def test_cfg_val_dict_value(self):
        """Test retrieving dict config values."""
        config = {"mapping": {"sectors": {"value": {"tech": 0.25, "health": 0.20}}}}
        result = cfg_val(config, "mapping", "sectors")
        self.assertEqual(result, {"tech": 0.25, "health": 0.20})

    def test_cfg_val_missing_section_returns_default(self):
        """Test that missing section returns default value."""
        config = {"section1": {"key1": {"value": "test"}}}
        result = cfg_val(config, "missing_section", "key1", "default_value")
        self.assertEqual(result, "default_value")

    def test_cfg_val_missing_key_returns_default(self):
        """Test that missing key returns default value."""
        config = {"section1": {"key1": {"value": "test"}}}
        result = cfg_val(config, "section1", "missing_key", "default_value")
        self.assertEqual(result, "default_value")

    def test_cfg_val_missing_value_returns_default(self):
        """Test that missing value field returns default."""
        config = {"section1": {"key1": {"other_field": "test"}}}
        result = cfg_val(config, "section1", "key1", "default_value")
        self.assertEqual(result, "default_value")

    def test_cfg_val_empty_config_returns_default(self):
        """Test that empty config returns default."""
        config = {}
        result = cfg_val(config, "section", "key", "default")
        self.assertEqual(result, "default")

    def test_cfg_val_default_none_when_not_specified(self):
        """Test that default is None when not specified."""
        config = {}
        result = cfg_val(config, "section", "key")
        self.assertIsNone(result)

    def test_cfg_val_nested_empty_dicts(self):
        """Test handling of nested empty dictionaries."""
        config = {"section1": {}}
        result = cfg_val(config, "section1", "key1", "default")
        self.assertEqual(result, "default")

    def test_cfg_val_boolean_value(self):
        """Test retrieving boolean config values."""
        config = {"flags": {"debug": {"value": True}, "verbose": {"value": False}}}
        self.assertTrue(cfg_val(config, "flags", "debug"))
        self.assertFalse(cfg_val(config, "flags", "verbose"))

    def test_cfg_val_zero_value_not_treated_as_missing(self):
        """Test that zero/falsy values are retrieved correctly."""
        config = {"settings": {"count": {"value": 0}}}
        result = cfg_val(config, "settings", "count")
        self.assertEqual(result, 0)

    def test_cfg_val_empty_string_not_treated_as_missing(self):
        """Test that empty string values are retrieved correctly."""
        config = {"settings": {"name": {"value": ""}}}
        result = cfg_val(config, "settings", "name")
        self.assertEqual(result, "")


class TestLoadBasisListFromPath(unittest.TestCase):
    """Test basis list loading from CSV files."""

    def setUp(self):
        """Create temporary directory for test files."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

    def tearDown(self):
        """Clean up temporary directory."""
        self.temp_dir.cleanup()

    def test_load_basis_with_ticker_column(self):
        """Test loading CSV with explicit ticker column."""
        csv_path = self.temp_path / "basis_ticker.csv"
        df = pd.DataFrame({"ticker": ["AAPL", "MSFT", "GOOG"]})
        df.to_csv(csv_path, index=False)

        result = load_basis_list_from_path(csv_path)
        self.assertEqual(result, ["AAPL", "MSFT", "GOOG"])

    def test_load_basis_single_column(self):
        """Test loading single-column CSV with tickers."""
        csv_path = self.temp_path / "basis_single.csv"
        df = pd.DataFrame({"assets": ["TEC0", "HEA1", "FIN2"]})
        df.to_csv(csv_path, index=False)

        result = load_basis_list_from_path(csv_path)
        self.assertEqual(result, ["TEC0", "HEA1", "FIN2"])

    def test_load_basis_semicolon_delimited_single_row(self):
        """Test loading single row with semicolon-delimited tickers."""
        csv_path = self.temp_path / "basis_delim.csv"
        df = pd.DataFrame({"basis": ["AAPL;MSFT;GOOG"]})
        df.to_csv(csv_path, index=False)

        result = load_basis_list_from_path(csv_path)
        self.assertEqual(result, ["AAPL", "MSFT", "GOOG"])

    def test_load_basis_comma_delimited_single_row(self):
        """Test loading single row with comma-delimited tickers."""
        csv_path = self.temp_path / "basis_comma.csv"
        df = pd.DataFrame({"basis": ["AAPL,MSFT,GOOG"]})
        df.to_csv(csv_path, index=False)

        result = load_basis_list_from_path(csv_path)
        self.assertEqual(result, ["AAPL", "MSFT", "GOOG"])

    def test_load_basis_multiple_rows_single_tickers(self):
        """Test loading timeseries CSV with single ticker per row."""
        csv_path = self.temp_path / "basis_timeseries.csv"
        df = pd.DataFrame(
            {
                "date": ["2020-01-01", "2020-02-01", "2020-03-01"],
                "basis": ["AAPL", "MSFT", "GOOG"],
            }
        )
        df.to_csv(csv_path, index=False)

        result = load_basis_list_from_path(csv_path)
        self.assertEqual(result, ["AAPL", "MSFT", "GOOG"])

    def test_load_basis_timeseries_delimited_last_row(self):
        """Test loading timeseries CSV using most recent delimited row."""
        csv_path = self.temp_path / "basis_ts_delim.csv"
        df = pd.DataFrame(
            {
                "date": ["2020-01-01", "2020-02-01", "2020-03-01"],
                "basis": ["AAPL", "MSFT", "GOOG;TSLA;META"],
            }
        )
        df.to_csv(csv_path, index=False)

        result = load_basis_list_from_path(csv_path)
        self.assertEqual(result, ["GOOG", "TSLA", "META"])

    def test_load_basis_with_whitespace_in_delimited(self):
        """Test that whitespace is trimmed from delimited tickers."""
        csv_path = self.temp_path / "basis_whitespace.csv"
        df = pd.DataFrame({"basis": ["AAPL ; MSFT ; GOOG"]})
        df.to_csv(csv_path, index=False)

        result = load_basis_list_from_path(csv_path)
        self.assertEqual(result, ["AAPL", "MSFT", "GOOG"])

    def test_load_basis_empty_csv_returns_empty_list(self):
        """Test that empty CSV returns empty list."""
        csv_path = self.temp_path / "basis_empty.csv"
        df = pd.DataFrame({"ticker": []})
        df.to_csv(csv_path, index=False)

        result = load_basis_list_from_path(csv_path)
        self.assertEqual(result, [])

    def test_load_basis_all_null_basis_column(self):
        """Test that all-null basis column returns empty list."""
        csv_path = self.temp_path / "basis_null.csv"
        df = pd.DataFrame({"basis": [None, None, None]})
        df.to_csv(csv_path, index=False)

        result = load_basis_list_from_path(csv_path)
        self.assertEqual(result, [])

    def test_load_basis_mixed_null_values(self):
        """Test handling of mixed null and non-null values."""
        csv_path = self.temp_path / "basis_mixed_null.csv"
        df = pd.DataFrame(
            {
                "date": ["2020-01-01", "2020-02-01", "2020-03-01"],
                "basis": [None, "MSFT", "GOOG"],
            }
        )
        df.to_csv(csv_path, index=False)

        result = load_basis_list_from_path(csv_path)
        self.assertEqual(result, ["MSFT", "GOOG"])

    def test_load_basis_numeric_tickers_converted_to_strings(self):
        """Test that numeric ticker values are converted to strings."""
        csv_path = self.temp_path / "basis_numeric.csv"
        df = pd.DataFrame({"ticker": [1, 2, 3]})
        df.to_csv(csv_path, index=False)

        result = load_basis_list_from_path(csv_path)
        self.assertEqual(result, ["1", "2", "3"])

    def test_load_basis_single_ticker_no_delimiter(self):
        """Test loading single ticker with no delimiter."""
        csv_path = self.temp_path / "basis_single_ticker.csv"
        df = pd.DataFrame({"basis": ["AAPL"]})
        df.to_csv(csv_path, index=False)

        result = load_basis_list_from_path(csv_path)
        self.assertEqual(result, ["AAPL"])

    def test_load_basis_empty_string_basis_row(self):
        """Test handling of empty string in basis column."""
        csv_path = self.temp_path / "basis_empty_string.csv"
        df = pd.DataFrame({"date": ["2020-01-01", "2020-02-01"], "basis": ["AAPL", ""]})
        df.to_csv(csv_path, index=False)

        result = load_basis_list_from_path(csv_path)
        # Empty strings are filtered out per the implementation
        self.assertEqual(result, ["AAPL"])

    def test_load_basis_delimited_with_empty_elements(self):
        """Test delimited list with empty elements after split."""
        csv_path = self.temp_path / "basis_empty_delim.csv"
        df = pd.DataFrame({"basis": ["AAPL;;GOOG"]})  # Empty element between delimiters
        df.to_csv(csv_path, index=False)

        result = load_basis_list_from_path(csv_path)
        # Empty strings should be filtered out
        self.assertEqual(result, ["AAPL", "GOOG"])

    def test_load_basis_mixed_delimiters_uses_first(self):
        """Test that mixed delimiters use the most recent row's delimiter."""
        csv_path = self.temp_path / "basis_mixed_delim.csv"
        df = pd.DataFrame(
            {
                "date": ["2020-01-01", "2020-02-01"],
                "basis": ["AAPL;MSFT", "GOOG,TSLA,META"],
            }
        )
        df.to_csv(csv_path, index=False)

        result = load_basis_list_from_path(csv_path)
        # Uses comma from last row
        self.assertEqual(result, ["GOOG", "TSLA", "META"])


class TestResolveOutputAndInputPaths(unittest.TestCase):
    """Test path resolution function."""

    def setUp(self):
        """Create temporary directory for test."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root_path = Path(self.temp_dir.name)

    def tearDown(self):
        """Clean up temporary directory."""
        self.temp_dir.cleanup()

    def test_resolve_paths_creates_outputs_dir(self):
        """Test that outputs directory is created."""
        config = {}
        outputs_dir, _, _, _ = resolve_output_and_input_paths(config, self.root_path)

        self.assertTrue(outputs_dir.exists())
        self.assertEqual(outputs_dir, self.root_path / "outputs")

    def test_resolve_paths_default_basis_path(self):
        """Test default basis path from config."""
        config = {}
        _, basis_path, _, _ = resolve_output_and_input_paths(config, self.root_path)

        self.assertEqual(basis_path, Path("outputs/basis_selected.csv"))

    def test_resolve_paths_custom_basis_path(self):
        """Test custom basis path from config."""
        config = {"output": {"basis_path": {"value": "custom/basis.csv"}}}
        _, basis_path, _, _ = resolve_output_and_input_paths(config, self.root_path)

        self.assertEqual(basis_path, Path("custom/basis.csv"))

    def test_resolve_paths_coeffs_in_outputs_dir(self):
        """Test that coefficients path is in outputs directory."""
        config = {}
        outputs_dir, _, coeffs_path, _ = resolve_output_and_input_paths(
            config, self.root_path
        )

        self.assertEqual(coeffs_path, outputs_dir / "coefficients_ridge_panel.csv")

    def test_resolve_paths_default_weights_path(self):
        """Test default weights path from config."""
        config = {}
        _, _, _, weights_path = resolve_output_and_input_paths(config, self.root_path)

        self.assertEqual(weights_path, Path("synthetic_data/market_index_weights.csv"))

    def test_resolve_paths_custom_weights_path(self):
        """Test custom weights path from config."""
        config = {"input": {"weights_path": {"value": "data/weights_custom.csv"}}}
        _, _, _, weights_path = resolve_output_and_input_paths(config, self.root_path)

        self.assertEqual(weights_path, Path("data/weights_custom.csv"))

    def test_resolve_paths_returns_four_paths(self):
        """Test that function returns four Path objects."""
        config = {}
        result = resolve_output_and_input_paths(config, self.root_path)

        self.assertEqual(len(result), 4)
        self.assertIsInstance(result[0], Path)
        self.assertIsInstance(result[1], Path)
        self.assertIsInstance(result[2], Path)
        self.assertIsInstance(result[3], Path)

    def test_resolve_paths_with_nested_config(self):
        """Test path resolution with complex nested config."""
        config = {
            "output": {"basis_path": {"value": "results/basis.csv"}},
            "input": {"weights_path": {"value": "raw_data/weights.csv"}},
        }
        outputs_dir, basis_path, coeffs_path, weights_path = (
            resolve_output_and_input_paths(config, self.root_path)
        )

        self.assertEqual(basis_path, Path("results/basis.csv"))
        self.assertEqual(weights_path, Path("raw_data/weights.csv"))
        self.assertTrue(outputs_dir.exists())


class TestLoadPricesCSV(unittest.TestCase):
    """Test price CSV loading function."""

    def setUp(self):
        """Create temporary directory for test files."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

    def tearDown(self):
        """Clean up temporary directory."""
        self.temp_dir.cleanup()

    def test_load_prices_basic(self):
        """Test basic price CSV loading."""
        csv_path = self.temp_path / "prices.csv"
        df = pd.DataFrame(
            {
                "date": ["2020-01-01", "2020-02-01", "2020-03-01"],
                "AAPL": [100.0, 110.0, 105.0],
                "MSFT": [200.0, 210.0, 215.0],
            }
        )
        df.to_csv(csv_path, index=False)

        result = load_prices_csv(csv_path)

        self.assertEqual(result.shape, (3, 2))
        self.assertIn("AAPL", result.columns)
        self.assertIn("MSFT", result.columns)

    def test_load_prices_index_parsed_as_datetime(self):
        """Test that first column is parsed as datetime index."""
        csv_path = self.temp_path / "prices_dates.csv"
        df = pd.DataFrame(
            {
                "date": ["2020-01-01", "2020-02-01", "2020-03-01"],
                "AAPL": [100.0, 110.0, 105.0],
            }
        )
        df.to_csv(csv_path, index=False)

        result = load_prices_csv(csv_path)

        self.assertIsInstance(result.index, pd.DatetimeIndex)

    def test_load_prices_sorted_by_index(self):
        """Test that returned DataFrame is sorted by index."""
        csv_path = self.temp_path / "prices_unsorted.csv"
        df = pd.DataFrame(
            {
                "date": ["2020-03-01", "2020-01-01", "2020-02-01"],
                "AAPL": [105.0, 100.0, 110.0],
            }
        )
        df.to_csv(csv_path, index=False)

        result = load_prices_csv(csv_path)

        self.assertTrue((result.index[:-1] <= result.index[1:]).all())

    def test_load_prices_multiple_assets(self):
        """Test loading prices for multiple assets."""
        csv_path = self.temp_path / "prices_multi.csv"
        df = pd.DataFrame(
            {
                "date": ["2020-01-01", "2020-02-01"],
                "A": [100.0, 105.0],
                "B": [50.0, 52.0],
                "C": [75.0, 78.0],
                "D": [200.0, 210.0],
            }
        )
        df.to_csv(csv_path, index=False)

        result = load_prices_csv(csv_path)

        self.assertEqual(result.shape[1], 4)
        self.assertEqual(result.shape[0], 2)

    def test_load_prices_with_nan_values(self):
        """Test handling of NaN values in price data."""
        csv_path = self.temp_path / "prices_nan.csv"
        df = pd.DataFrame(
            {
                "date": ["2020-01-01", "2020-02-01", "2020-03-01"],
                "AAPL": [100.0, np.nan, 105.0],
                "MSFT": [200.0, 210.0, np.nan],
            }
        )
        df.to_csv(csv_path, index=False)

        result = load_prices_csv(csv_path)

        self.assertTrue(result.isna().any().any())
        self.assertTrue(pd.isna(result.loc[result.index[1], "AAPL"]))

    def test_load_prices_preserves_data_types(self):
        """Test that numeric data types are preserved."""
        csv_path = self.temp_path / "prices_numeric.csv"
        df = pd.DataFrame(
            {
                "date": ["2020-01-01", "2020-02-01"],
                "AAPL": [100.5, 110.75],
                "MSFT": [200.25, 215.5],
            }
        )
        df.to_csv(csv_path, index=False)

        result = load_prices_csv(csv_path)

        self.assertTrue(pd.api.types.is_numeric_dtype(result["AAPL"]))
        self.assertTrue(pd.api.types.is_numeric_dtype(result["MSFT"]))

    def test_load_prices_single_asset(self):
        """Test loading prices for single asset."""
        csv_path = self.temp_path / "prices_single.csv"
        df = pd.DataFrame(
            {
                "date": ["2020-01-01", "2020-02-01", "2020-03-01"],
                "AAPL": [100.0, 110.0, 105.0],
            }
        )
        df.to_csv(csv_path, index=False)

        result = load_prices_csv(csv_path)

        self.assertEqual(result.shape[1], 1)

    def test_load_prices_single_row(self):
        """Test loading prices with single row."""
        csv_path = self.temp_path / "prices_one_row.csv"
        df = pd.DataFrame({"date": ["2020-01-01"], "AAPL": [100.0], "MSFT": [200.0]})
        df.to_csv(csv_path, index=False)

        result = load_prices_csv(csv_path)

        self.assertEqual(result.shape[0], 1)


class TestLoadWeightsCSV(unittest.TestCase):
    """Test weights CSV loading function."""

    def setUp(self):
        """Create temporary directory for test files."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

    def tearDown(self):
        """Clean up temporary directory."""
        self.temp_dir.cleanup()

    def test_load_weights_basic(self):
        """Test basic weights CSV loading."""
        csv_path = self.temp_path / "weights.csv"
        df = pd.DataFrame(
            {
                "date": ["2020-01-01", "2020-02-01"],
                "AAPL": [0.3, 0.35],
                "MSFT": [0.5, 0.45],
                "GOOG": [0.2, 0.2],
            }
        )
        df.to_csv(csv_path, index=False)

        result = load_weights_csv(csv_path)

        self.assertEqual(result.shape, (2, 3))
        self.assertIn("AAPL", result.columns)

    def test_load_weights_index_as_datetime(self):
        """Test that index is parsed as datetime."""
        csv_path = self.temp_path / "weights_datetime.csv"
        df = pd.DataFrame(
            {"date": ["2020-01-01", "2020-02-01"], "A": [0.5, 0.5], "B": [0.5, 0.5]}
        )
        df.to_csv(csv_path, index=False)

        result = load_weights_csv(csv_path)

        self.assertIsInstance(result.index, pd.DatetimeIndex)

    def test_load_weights_sorted_by_index(self):
        """Test that DataFrame is sorted by date index."""
        csv_path = self.temp_path / "weights_unsorted.csv"
        df = pd.DataFrame(
            {
                "date": ["2020-03-01", "2020-01-01", "2020-02-01"],
                "A": [0.5, 0.3, 0.4],
                "B": [0.5, 0.7, 0.6],
            }
        )
        df.to_csv(csv_path, index=False)

        result = load_weights_csv(csv_path)

        self.assertTrue((result.index[:-1] <= result.index[1:]).all())

    def test_load_weights_weights_normalized(self):
        """Test loading weights that sum to 1.0."""
        csv_path = self.temp_path / "weights_normalized.csv"
        df = pd.DataFrame(
            {
                "date": ["2020-01-01", "2020-02-01"],
                "AAPL": [0.3, 0.25],
                "MSFT": [0.5, 0.55],
                "GOOG": [0.2, 0.2],
            }
        )
        df.to_csv(csv_path, index=False)

        result = load_weights_csv(csv_path)

        # Verify weights sum to approximately 1.0
        row_sums = result.sum(axis=1)
        np.testing.assert_array_almost_equal(row_sums, np.ones(len(result)))

    def test_load_weights_multiple_periods(self):
        """Test loading weights across multiple time periods."""
        csv_path = self.temp_path / "weights_multi_period.csv"
        dates = ["2020-01-01", "2020-02-01", "2020-03-01", "2020-04-01", "2020-05-01"]
        df = pd.DataFrame(
            {
                "date": dates,
                "A": [0.4, 0.45, 0.5, 0.35, 0.3],
                "B": [0.6, 0.55, 0.5, 0.65, 0.7],
            }
        )
        df.to_csv(csv_path, index=False)

        result = load_weights_csv(csv_path)

        self.assertEqual(result.shape[0], 5)

    def test_load_weights_with_nan_values(self):
        """Test handling of NaN values in weights."""
        csv_path = self.temp_path / "weights_nan.csv"
        df = pd.DataFrame(
            {"date": ["2020-01-01", "2020-02-01"], "A": [0.5, np.nan], "B": [0.5, 0.5]}
        )
        df.to_csv(csv_path, index=False)

        result = load_weights_csv(csv_path)

        self.assertTrue(result.isna().any().any())

    def test_load_weights_preserves_numeric_precision(self):
        """Test that numeric precision is preserved."""
        csv_path = self.temp_path / "weights_precision.csv"
        df = pd.DataFrame(
            {"date": ["2020-01-01"], "A": [0.33333333], "B": [0.66666667]}
        )
        df.to_csv(csv_path, index=False)

        result = load_weights_csv(csv_path)

        # Check values are preserved with reasonable precision
        self.assertAlmostEqual(result.loc[result.index[0], "A"], 0.33333333, places=6)

    def test_load_weights_single_asset(self):
        """Test loading weights for single asset."""
        csv_path = self.temp_path / "weights_single.csv"
        df = pd.DataFrame({"date": ["2020-01-01", "2020-02-01"], "A": [1.0, 1.0]})
        df.to_csv(csv_path, index=False)

        result = load_weights_csv(csv_path)

        self.assertEqual(result.shape[1], 1)
        np.testing.assert_array_almost_equal(result["A"].values, [1.0, 1.0])

    def test_load_weights_single_row(self):
        """Test loading weights with single time period."""
        csv_path = self.temp_path / "weights_one_row.csv"
        df = pd.DataFrame(
            {"date": ["2020-01-01"], "A": [0.25], "B": [0.25], "C": [0.25], "D": [0.25]}
        )
        df.to_csv(csv_path, index=False)

        result = load_weights_csv(csv_path)

        self.assertEqual(result.shape[0], 1)

    def test_load_weights_many_assets(self):
        """Test loading weights for many assets."""
        csv_path = self.temp_path / "weights_many.csv"
        n_assets = 50
        weight_value = 1.0 / n_assets
        data = {
            "date": ["2020-01-01"],
            **{f"Asset{i}": [weight_value] for i in range(n_assets)},
        }
        df = pd.DataFrame(data)
        df.to_csv(csv_path, index=False)

        result = load_weights_csv(csv_path)

        self.assertEqual(result.shape[1], n_assets)


if __name__ == "__main__":
    unittest.main()
