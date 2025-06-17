import json
import pytest
import pandas as pd
from unittest.mock import patch

from backtest.utils.data_loader import prepare_market_data
# noinspection PyProtectedMember
from backtest.utils.data_loader import _load_market_data  # pylint: disable=protected-access


@pytest.mark.unit
class TestPrepareMarketData:

    @pytest.fixture
    def sample_market_data_df(self):
        """Provides a sample market data DataFrame for testing.

        Returns:
            A pandas DataFrame with sample market data, including columns like
            'year', 'symbol', 'sales_growth_5y', 'annual_return', and
            'is_delisted'.
        """
        return pd.DataFrame({
            'year': [2020, 2020, 2021, 2021],
            'symbol': ['A', 'B', 'A', 'B'],
            'sales_growth_5y': [0.1, 0.2, 0.15, 0.25],
            'annual_return': [10, 5, 12, 8],
            'is_delisted': [False, False, False, True]
        })

    @patch('backtest.utils.data_loader._load_market_data')
    @patch('backtest.utils.data_loader.apply_stress_test')
    @patch('click.echo')
    def test_prepare_market_data_no_stress_no_delisted(
            self, mock_click_echo, mock_apply_stress_test, mock_load_market_data, sample_market_data_df
    ):
        mock_load_market_data.return_value = sample_market_data_df.copy()

        result_df = prepare_market_data(
            data_file_path="dummy.csv",
            stress_test="none",
            start_year=2020,
            end_year=2021,
            include_delisted=False
        )

        mock_load_market_data.assert_called_once_with("dummy.csv")
        mock_apply_stress_test.assert_not_called()
        for call_args in mock_click_echo.call_args_list:
            assert "Including delisted companies" not in call_args[0][0]
        pd.testing.assert_frame_equal(result_df, sample_market_data_df)

    @patch('backtest.utils.data_loader._load_market_data')
    @patch('backtest.utils.data_loader.apply_stress_test')
    @patch('click.echo')
    def test_prepare_market_data_with_stress_test(
            self, mock_click_echo, mock_apply_stress_test, mock_load_market_data, sample_market_data_df
    ):
        # The DataFrame that _load_market_data will return.
        loaded_df = sample_market_data_df.copy()
        mock_load_market_data.return_value = loaded_df
        # The DataFrame that apply_stress_test is expected to return.
        stressed_df_output = sample_market_data_df.copy()
        stressed_df_output['annual_return'] = stressed_df_output['annual_return'] * 0.5  # Dummy stress effect
        mock_apply_stress_test.return_value = stressed_df_output

        result_df = prepare_market_data(
            data_file_path="dummy.csv",
            stress_test="2008crisis",
            start_year=2020,
            end_year=2021,
            include_delisted=False
        )

        # Verify _load_market_data was called correctly
        mock_load_market_data.assert_called_once_with("dummy.csv")

        # Verify apply_stress_test was called once
        mock_apply_stress_test.assert_called_once()

        # Get the arguments apply_stress_test was called with
        args, kwargs = mock_apply_stress_test.call_args

        # Assert the DataFrame argument using pandas.testing
        # The first argument to apply_stress_test should be the DataFrame returned by _load_market_data
        pd.testing.assert_frame_equal(args[0], loaded_df)

        # Assert the other arguments
        assert args[1] == "2008crisis"
        assert args[2] == 2020
        assert args[3] == 2021
        assert not kwargs  # Ensure no unexpected keyword arguments were passed

        mock_click_echo.assert_any_call("Applying stress test scenario: 2008crisis for period 2020-2021")
        pd.testing.assert_frame_equal(result_df, stressed_df_output)

    @patch('backtest.utils.data_loader._load_market_data')
    @patch('backtest.utils.data_loader.apply_stress_test')
    @patch('click.echo')
    def test_prepare_market_data_include_delisted_present(
            self, mock_click_echo, mock_apply_stress_test, mock_load_market_data, sample_market_data_df
    ):
        mock_load_market_data.return_value = sample_market_data_df.copy()

        prepare_market_data(
            data_file_path="dummy.csv",
            stress_test="none",
            start_year=2020,
            end_year=2021,
            include_delisted=True
        )

        mock_apply_stress_test.assert_not_called()
        mock_click_echo.assert_any_call("Including delisted companies in analysis.")

    @patch('backtest.utils.data_loader._load_market_data')
    @patch('backtest.utils.data_loader.apply_stress_test')
    @patch('click.echo')
    def test_prepare_market_data_include_delisted_missing_column(
            self, mock_click_echo, mock_apply_stress_test, mock_load_market_data, sample_market_data_df
    ):
        data_without_delisted_col = sample_market_data_df.drop(columns=['is_delisted'])
        mock_load_market_data.return_value = data_without_delisted_col.copy()

        prepare_market_data(
            data_file_path="dummy.csv",
            stress_test="none",
            start_year=2020,
            end_year=2021,
            include_delisted=True
        )

        mock_click_echo.assert_any_call(
            "Warning: Delisted company data ('is_delisted' column) "
            "not available in the provided market data.",
            err=True
        )


# --- Tests for _load_market_data() ---
@pytest.mark.unit
class TestLoadMarketData:
    def test_load_valid_csv(self, tmp_path):
        file_path = tmp_path / "data.csv"
        data = {'col1': [1, 2], 'col2': ['a', 'b']}
        pd.DataFrame(data).to_csv(file_path, index=False)
        df = _load_market_data(str(file_path))
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (2, 2)
        assert list(df.columns) == ['col1', 'col2']

    def test_load_valid_json(self, tmp_path):
        file_path = tmp_path / "data.json"
        data = [{'col1': 1, 'col2': 'a'}, {'col1': 2, 'col2': 'b'}]
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f)
        df = _load_market_data(str(file_path))
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (2, 2)
        assert set(df.columns) == {'col1', 'col2'}  # Order might not be preserved from list of dicts

    def test_load_file_not_found(self):
        with pytest.raises(FileNotFoundError, match="Data file not found: nonexistent.csv"):
            _load_market_data("nonexistent.csv")

    def test_load_unsupported_extension(self, tmp_path):
        file_path = tmp_path / "data.txt"
        file_path.write_text("some data")
        with pytest.raises(ValueError, match="Unsupported file format: '.txt'"):
            _load_market_data(str(file_path))

    def test_load_corrupt_json(self, tmp_path):
        file_path = tmp_path / "data.json"
        file_path.write_text("{'col1': 1, 'col2': 'a'")  # Malformed JSON
        with pytest.raises(ValueError, match="Error decoding JSON"):
            _load_market_data(str(file_path))

    @patch('click.echo')
    def test_load_empty_csv(self, mock_click_echo, tmp_path):
        # Test CSV with headers but no data rows
        file_path_headers_only = tmp_path / "empty_with_headers.csv"
        pd.DataFrame(columns=['h1', 'h2']).to_csv(file_path_headers_only, index=False)
        df_headers_only = _load_market_data(str(file_path_headers_only))
        assert df_headers_only.empty
        assert list(df_headers_only.columns) == ['h1', 'h2']
        mock_click_echo.assert_not_called()  # No warning for this case

        # Test truly empty CSV (0 bytes)
        file_path_truly_empty = tmp_path / "truly_empty.csv"
        file_path_truly_empty.write_text("")

        df_truly_empty = _load_market_data(str(file_path_truly_empty))
        assert df_truly_empty.empty
        assert list(df_truly_empty.columns) == []
        mock_click_echo.assert_any_call(
            f"Warning: CSV file {str(file_path_truly_empty)} is empty. Returning empty DataFrame.",
            err=True
        )

    def test_load_empty_json_list(self, tmp_path):
        file_path = tmp_path / "empty.json"
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump([], f)
        df = _load_market_data(str(file_path))
        assert df.empty