"""End-to-end tests for the example data generation CLI.

These tests verify the `generate_example_data` command, ensuring it
creates data files as expected for different strategies and options.
"""
from datetime import datetime

import pandas as pd
import pytest
from click.testing import CliRunner

from backtest.example_data import cli as generate_data_cli
from backtest.example_data import (
    DEFAULT_START_YEAR_OFFSET,
    NUM_IDEAL_DGI_STOCKS_PER_YEAR,
    NUM_IDEAL_VALUE_STOCKS_PER_YEAR,
    STRATEGY_COLUMNS
)
from backtest.strategies.value_play import ValuePlayStrategy


STRATEGIES_TO_TEST = ['exp_fund', 'dgi', 'value_play']

# S&P Quality Rank Mapping (similar to ValuePlayStrategy and DgiStrategy).
SP_QUALITY_RANK_MAPPING_FOR_TEST = {
    'A+': 6, 'A': 5, 'A-': 4,
    'B+': 3, 'B': 2, 'B-': 1,
    'C': 0, 'D': -1  # C and D for completeness, mapped to lower values.
}
MIN_DGI_SP_QUALITY_NUMERIC = SP_QUALITY_RANK_MAPPING_FOR_TEST['B+']


def dgi_screener_for_test(df_year: pd.DataFrame) -> pd.DataFrame:
    """Applies DGI screening criteria to a DataFrame for a single year.

    This helper function is used in tests to verify that the generated
    DGI example data includes stocks meeting the DGI criteria.

    Args:
        df_year: A pandas DataFrame containing data for a single year.

    Returns:
        A pandas DataFrame containing only the stocks that pass DGI screening.
    """
    df_year_copy = df_year.copy()  # Work on a copy.
    df_year_copy['sp_quality_numeric'] = df_year_copy.get(
        'sp_quality', pd.Series(dtype=str)
    ).map(
        SP_QUALITY_RANK_MAPPING_FOR_TEST
    ).fillna(0)  # Default unmapped to a low numeric value.

    sp_quality_condition = (
        df_year_copy['sp_quality_numeric'] >= MIN_DGI_SP_QUALITY_NUMERIC
    )
    condition = (
            (df_year.get('div_growth_streak', pd.Series(dtype=float)) >= 10) &
            (df_year.get('payout_ratio', pd.Series(dtype=float)) <= 0.60) &
            (df_year.get('eps_cagr_3y', pd.Series(dtype=float)) >= 0.05) &
            (df_year.get('roe', pd.Series(dtype=float)) >= 0.15) &
            (df_year.get('debt_equity', pd.Series(dtype=float)) <=
             df_year.get('industry_debt_equity', pd.Series(dtype=float))) &
            sp_quality_condition
    )
    return df_year[condition]


@pytest.mark.e2e
class TestGenerateExampleDataCLI:
    """Test suite for the `generate_example_data` CLI command."""

    @pytest.mark.parametrize("strategy_name", STRATEGIES_TO_TEST)
    def test_generate_data_all_strategies(
        self, strategy_name: str, tmp_path, monkeypatch
    ):
        """Tests default data generation for each supported strategy.

        Verifies file creation, basic CSV validity, and presence of key columns.

        Args:
            strategy_name: The name of the strategy to test.
            tmp_path: Pytest fixture for a temporary directory.
            monkeypatch: Pytest fixture for modifying attributes.
        """
        runner = CliRunner()
        # Change CWD for the CLI runner to tmp_path so 'data/' subdir is created there.
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(generate_data_cli, [strategy_name])

        assert result.exit_code == 0, \
            f"CLI command failed for {strategy_name}: {result.output}"
        # Default output path is data/[strategy_name].example_data.01.csv.
        expected_file_name = f"{strategy_name}.example_data.01.csv"
        expected_file_path = tmp_path / "data" / expected_file_name
        assert expected_file_path.exists(), \
            f"Output file not found for {strategy_name} at {expected_file_path}"

        try:
            df = pd.read_csv(expected_file_path)
        except Exception as e:  # pylint: disable=broad-except
            pytest.fail(f"Failed to read generated CSV for {strategy_name}: {e}")

        assert not df.empty, f"Generated CSV for {strategy_name} is empty."
        # Check for a few key required columns.
        required_cols = STRATEGY_COLUMNS[strategy_name]['required']
        for col in required_cols[:3]:  # Check first 3 required columns.
            assert col in df.columns, f"Key column '{col}' missing for {strategy_name}."
        assert 'year' in df.columns
        assert 'symbol' in df.columns

    def test_generate_data_custom_output_path(self, tmp_path):
        """Tests data generation with a custom output file path."""
        runner = CliRunner()
        strategy_name = "exp_fund"
        custom_file_name = "custom_exp_fund_data.csv"
        custom_output_path = tmp_path / custom_file_name
        result = runner.invoke(
            generate_data_cli, [strategy_name, str(custom_output_path)]
        )

        assert result.exit_code == 0, \
            f"CLI command failed with custom path: {result.output}"
        assert custom_output_path.exists(), "Output file not found at custom path."

        try:
            df = pd.read_csv(custom_output_path)
            assert not df.empty, "Generated CSV at custom path is empty."
        except Exception as e:  # pylint: disable=broad-except
            pytest.fail(f"Failed to read generated CSV from custom path: {e}")

    def test_generate_data_with_options(self, tmp_path):
        """Tests data generation with custom options like year range and seed."""
        runner = CliRunner()
        strategy_name = "dgi"
        output_file = tmp_path / "dgi_options_data.csv"
        start_year_opt = 2021
        end_year_opt = 2022
        num_companies_opt = 5
        seed_opt = 123
        result = runner.invoke(generate_data_cli, [
            strategy_name, str(output_file),
            "--start-year", str(start_year_opt),
            "--end-year", str(end_year_opt),
            "--num-companies", str(num_companies_opt),
            "--seed", str(seed_opt)
        ])

        assert result.exit_code == 0, \
            f"CLI command failed with options: {result.output}"
        assert output_file.exists(), "Output file with options not found."
        df = pd.read_csv(output_file)
        assert not df.empty, "Generated CSV with options is empty."

        assert df['year'].min() == start_year_opt, "Min year does not match option."
        assert df['year'].max() == end_year_opt, "Max year does not match option."
        assert len(df['year'].unique()) == (end_year_opt - start_year_opt + 1), \
            "Number of unique years is incorrect."
        assert df['symbol'].nunique() == num_companies_opt, \
            "Number of unique symbols does not match option."

        # Verify data consistency with the same seed.
        output_file_seed_check = tmp_path / "dgi_options_data_seed_check.csv"
        runner.invoke(generate_data_cli, [
            strategy_name, str(output_file_seed_check),
            "--start-year", str(start_year_opt),
            "--end-year", str(end_year_opt),
            "--num-companies", str(num_companies_opt),
            "--seed", str(seed_opt)
        ])
        df_seed_check = pd.read_csv(output_file_seed_check)
        pd.testing.assert_frame_equal(
            df, df_seed_check, check_dtype=False
        ), "Data generated with the same seed is not identical."

    def test_generate_data_invalid_strategy(self):
        """Tests CLI behavior when an invalid strategy name is provided."""
        runner = CliRunner()
        result = runner.invoke(generate_data_cli, ["invalid_strategy_name"])

        assert result.exit_code == 2, \
            (f"Expected exit code 2 for invalid choice, got {result.exit_code}. "
             f"Output: {result.output}")
        expected_error_fragment = ("Invalid value for 'STRATEGY': "
                                   "'invalid_strategy_name' is not one of")
        assert expected_error_fragment in result.output, \
            f"Expected error message fragment not found. Output:\n{result.output}"

    @pytest.mark.parametrize("strategy_name", STRATEGIES_TO_TEST)
    def test_generate_data_ideal_stocks(
        self, strategy_name: str, tmp_path, monkeypatch
    ):
        """Tests if generated data includes expected 'ideal' stocks.

        Verifies that for each strategy, a minimum number of stocks pass
        the primary screening criteria each year in the generated data.

        Args:
            strategy_name: The name of the strategy to test.
            tmp_path: Pytest fixture for a temporary directory.
            monkeypatch: Pytest fixture for modifying attributes.
        """
        runner = CliRunner()
        monkeypatch.chdir(tmp_path)
        num_companies = 20
        start_year = datetime.now().year - DEFAULT_START_YEAR_OFFSET
        end_year = datetime.now().year - 1
        result = runner.invoke(generate_data_cli, [
            strategy_name,
            "--num-companies", str(num_companies),
            "--start-year", str(start_year),
            "--end-year", str(end_year),
            "--seed", "42"  # Use a fixed seed for reproducibility.
        ])

        assert result.exit_code == 0, \
            f"Data generation failed for {strategy_name}: {result.output}"
        generated_file_path = tmp_path / "data" / f"{strategy_name}.example_data.01.csv"
        assert generated_file_path.exists()
        df_generated = pd.read_csv(generated_file_path)

        for year_val in range(start_year, end_year + 1):
            df_year = df_generated[df_generated['year'] == year_val].copy()
            if df_year.empty:
                print(f"Warning: No data for year {year_val} for {strategy_name}.")
                continue

            if strategy_name == 'exp_fund':
                strategy_params = {'growth_threshold': 0.2, 'top_n': 10}
                growth_filter_condition = (
                    (df_year.get('sales_growth_5y', pd.Series(dtype=float)) >=
                     strategy_params['growth_threshold']) &
                    (df_year.get('market_cap_rank', pd.Series(dtype=int)) <=
                     strategy_params['top_n'])
                )
                passed_screening = df_year[growth_filter_condition]
                assert not passed_screening.empty, \
                    f"Year {year_val}: No stocks passed ExpFund screening for {strategy_name}."
                assert len(passed_screening) >= 1, \
                    (f"Year {year_val}: Expected >= 1 stock for ExpFund, "
                     f"got {len(passed_screening)}.")
            elif strategy_name == 'dgi':
                passed_screening = dgi_screener_for_test(df_year)
                assert len(passed_screening) >= NUM_IDEAL_DGI_STOCKS_PER_YEAR, \
                    (f"Year {year_val}: Expected >= {NUM_IDEAL_DGI_STOCKS_PER_YEAR} DGI ideal, "
                     f"found {len(passed_screening)}.")
            elif strategy_name == 'value_play':
                value_strategy = ValuePlayStrategy()
                candidates_df = df_year.copy()
                if 'sp_quality' in candidates_df.columns:
                    candidates_df['sp_quality_numeric'] = (
                        value_strategy._map_sp_quality_to_numeric(
                            candidates_df['sp_quality']
                        )
                    )
                else:
                    candidates_df['sp_quality_numeric'] = 0 # Will fail quality filter.

                pe_cond = (candidates_df.get('pe_ratio', pd.Series(dtype=float)) <
                           candidates_df.get('sector_median_pe', pd.Series(dtype=float)) * 0.4)
                pb_cond = (candidates_df.get('pb_ratio', pd.Series(dtype=float)) < 1.0)
                cr_cond = (candidates_df.get('current_ratio', pd.Series(dtype=float)) > 1.5)
                de_cond = (candidates_df.get('debt_equity', pd.Series(dtype=float)) <
                           candidates_df.get('industry_avg_debt_equity', pd.Series(dtype=float)))
                roe_cond = (candidates_df.get('roe', pd.Series(dtype=float)) > 0.15)
                eps_g_cond = (candidates_df.get('eps_growth_5y', pd.Series(dtype=float)) > 0)
                spq_cond = (candidates_df.get('sp_quality_numeric', pd.Series(dtype=float)) >=
                            value_strategy.MIN_SP_QUALITY_NUMERIC)
                mos_cond = (candidates_df.get('margin_of_safety', pd.Series(dtype=float)) > 0.25)

                passed_screening = candidates_df[
                    pe_cond & pb_cond & cr_cond & de_cond &
                    roe_cond & eps_g_cond & spq_cond & mos_cond
                ]
                assert len(passed_screening) >= NUM_IDEAL_VALUE_STOCKS_PER_YEAR, \
                    (f"Year {year_val}: Expected >= {NUM_IDEAL_VALUE_STOCKS_PER_YEAR} Value ideal, "
                     f"found {len(passed_screening)}.")