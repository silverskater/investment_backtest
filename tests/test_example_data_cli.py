"""End-to-end tests for the example data generation CLI.

These tests verify the `generate_example_data` command, ensuring it
creates data files as expected for different strategies and options.
"""
from datetime import datetime

import pandas as pd
import pytest
from click.testing import CliRunner

from backtest.cli import cli
from data_provider.constants import (
    DEFAULT_DEMO_START_YEAR_OFFSET,
    NUM_IDEAL_DGI_STOCKS_PER_YEAR,
    NUM_IDEAL_VALUE_STOCKS_PER_YEAR,
    STRATEGY_COLUMNS
)
from backtest.utils.quality_utils import (
    map_sp_quality_to_numeric,
    MIN_SP_QUALITY_NUMERIC
)

STRATEGIES_TO_TEST = ['dgi', 'exp_fund', 'value_play']


def dgi_screener_for_test(df_year: pd.DataFrame) -> pd.DataFrame:
    """Applies DGI screening criteria to a DataFrame for a single year.

    This helper function is used in tests to verify that the generated
    DGI example data includes stocks meeting the DGI criteria.

    Args:
        df_year: A pandas DataFrame containing data for a single year.

    Returns:
        A pandas DataFrame containing only the stocks that pass DGI screening.
    """
    df_year_copy = df_year.copy()
    df_year_copy['sp_quality_numeric'] = map_sp_quality_to_numeric(
        df_year_copy.get('sp_quality', pd.Series(dtype=str))  # Use .get() for safety
    )
    sp_quality_condition = (
            df_year_copy['sp_quality_numeric'] >= MIN_SP_QUALITY_NUMERIC
    )

    # Using fillna for robustness in test screening conditions.
    condition = (
            (df_year.get('div_growth_streak', pd.Series(dtype=float)).fillna(0) >= 10) &  # Fill NaN with 0
            (df_year.get('payout_ratio', pd.Series(dtype=float)).fillna(
                1.0) <= 0.60) &  # Fill NaN with 1.0 (fails <= 0.60)
            (df_year.get('eps_cagr_3y', pd.Series(dtype=float)).fillna(
                float('-inf')) >= 0.05) &  # Fill NaN with neg inf
            (df_year.get('roe', pd.Series(dtype=float)).fillna(float('-inf')) >= 0.15) &  # Fill NaN with neg inf
            (df_year.get('debt_equity', pd.Series(dtype=float)).fillna(float('inf')) <=  # Fill NaN with inf
             df_year.get('industry_debt_equity', pd.Series(dtype=float)).fillna(float('inf'))) &  # Fill NaN with inf
            sp_quality_condition  # Uses the numeric column derived from imported function/constant
    )
    # Apply condition to the original df_year to keep original columns
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
        result = runner.invoke(cli, ["data", strategy_name])

        assert result.exit_code == 0, \
            f"CLI command failed for {strategy_name}: {result.output}"
        # Default output path is data/[strategy_name].[fetcher].01.csv.
        expected_file_name = f"{strategy_name}.demo.01.csv"
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
        # Check first few required columns for basic structure validation
        cols_to_check = required_cols[:min(len(required_cols), 5)]  # Check up to 5 required columns
        for col in cols_to_check:
            assert col in df.columns, f"Key column '{col}' missing for {strategy_name}."
        assert 'year' in df.columns
        assert 'symbol' in df.columns

    def test_generate_data_custom_output_path(self, tmp_path):
        """Tests data generation with a custom output file path."""
        runner = CliRunner()
        strategy_name = "exp_fund"
        output_file_path = tmp_path / f"{strategy_name}.custom_output_path.csv"
        result = runner.invoke(
            cli, ["data", strategy_name, "--output", str(output_file_path)]
        )

        assert result.exit_code == 0, \
            f"CLI command failed with custom path: {result.output}"
        assert output_file_path.exists(), "Output file not found at custom path."

        try:
            df = pd.read_csv(output_file_path)
            assert not df.empty, "Generated CSV at custom path is empty."
        except Exception as e:  # pylint: disable=broad-except
            pytest.fail(f"Failed to read generated CSV from custom path: {e}")

    @staticmethod
    def _run_generate_data_command(
            runner: CliRunner,
            strategy_name: str,
            output_file_path: str,
            start_year: int,
            end_year: int,
            num_companies: int,
            seed: int,
    ) -> pd.DataFrame:
        """Helper to run the generate data command and load the result."""
        result = runner.invoke(cli, [
            "data",
            strategy_name,
            "--output", str(output_file_path),
            "--demo-start-year", str(start_year),
            "--demo-end-year", str(end_year),
            "--demo-num-companies", str(num_companies),
            "--demo-seed", str(seed),
        ])

        assert result.exit_code == 0, \
            f"CLI 'data' command failed. Output:\n{result.output}"
        assert output_file_path.exists(), f"Output file not found at {output_file_path}"
        return pd.read_csv(output_file_path)


    def test_generate_data_with_options(self, tmp_path, monkeypatch):
        """Tests data generation with custom options like year range and seed."""
        runner = CliRunner()
        # Changing CWD to tmp_path for consistency and to catch relative path issues.
        monkeypatch.chdir(tmp_path)

        strategy_name = "dgi"
        output_file_name = f"{strategy_name}.options_data.csv"
        output_file_path = tmp_path / output_file_name  # Output directly in tmp_path

        start_year_opt = 2021
        end_year_opt = 2022
        num_companies_opt = 5
        seed_opt = 123

        self._run_generate_data_command(
            runner,
            strategy_name,
            output_file_path,
            start_year_opt,
            end_year_opt,
            num_companies_opt,
            seed_opt)
        df_initial = None
        try:
            df_initial = pd.read_csv(output_file_path)
            assert not df_initial.empty, "Generated CSV from 'data' command is empty."

            assert df_initial['year'].min() == start_year_opt, \
                "Min year in generated data does not match --demo-start-year option."
            assert df_initial['year'].max() == end_year_opt, \
                "Max year in generated data does not match --demo-end-year option."
            assert len(df_initial['year'].unique()) == (end_year_opt - start_year_opt + 1), \
                "Number of unique years in generated data is incorrect."

            # This test expects the demo generator to produce exactly num_companies_opt.
            assert df_initial['symbol'].nunique() == num_companies_opt, \
                "Number of unique symbols does not match --demo-num-companies option."
            # This also implies that if num_companies_opt > 0, nunique() > 0.
            # And if num_companies_opt == 0, nunique() == 0.

        except pd.errors.EmptyDataError:
            pytest.fail(f"Generated CSV file {output_file_path} is empty or invalid.")
        except Exception as e:
            pytest.fail(f"Failed to read or validate generated CSV from 'data' command: {e}")

        # Verify data consistency with the same seed.
        output_file_seed_check_name = f"{strategy_name}.options_data_seed_check.csv"
        output_file_seed_check_path = tmp_path / output_file_seed_check_name

        df_seed_check = self._run_generate_data_command(
            runner,
            strategy_name,
            output_file_seed_check_path,
            start_year_opt,
            end_year_opt,
            num_companies_opt,
            seed_opt)

        try:
            df_seed_check = pd.read_csv(output_file_seed_check_path)
            assert not df_seed_check.empty, "Generated CSV for seed check is empty."
        except Exception as e:
            pytest.fail(f"Failed to read generated CSV for seed check: {e}")

        if df_initial is not None:  # Ensure df_initial was loaded
            pd.testing.assert_frame_equal(
                df_initial, df_seed_check, check_dtype=False
            ), "Data generated with the same seed is not identical."
        else:
            pytest.fail("Initial DataFrame (df_initial) was not loaded; cannot perform seed consistency check.")


    def test_generate_data_invalid_strategy(self):
        """Tests CLI behavior when an invalid strategy name is provided."""
        runner = CliRunner()
        result = runner.invoke(cli, ["data", "invalid_strategy_name"])

        assert result.exit_code == 2, \
            (f"Expected exit code 2 for invalid choice, got {result.exit_code}. "
             f"Output: {result.output}")
        # The exact error message might change based on how Click structures it for subcommands
        # This is a general check.
        assert "Invalid value for 'STRATEGY'" in result.output or \
               "invalid_strategy_name" in result.output
        assert "is not one of" in result.output


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
        start_year = datetime.now().year - DEFAULT_DEMO_START_YEAR_OFFSET
        end_year = datetime.now().year - 1
        # Use a unique filename for this test to avoid clashes if run multiple times
        # or if other tests use the default naming.
        output_file_name = f"{strategy_name}.ideal_stocks_test.csv"
        output_file_path = tmp_path / "data" / output_file_name

        self._run_generate_data_command(
            runner,
            strategy_name,
            output_file_path,
            start_year,
            end_year,
            num_companies,
            42)  # Use a fixed seed for reproducibility.
        assert output_file_path.exists(), \
            f"Generated file for ideal stocks test not found at {output_file_path}"
        df_generated = pd.read_csv(output_file_path)

        for year_val in range(start_year, end_year + 1):
            df_year = df_generated[df_generated['year'] == year_val].copy()
            if df_year.empty:
                # This might be acceptable if the test period is very long and
                # some years genuinely have no data from the generator.
                # However, for a 10-year span, we expect data.
                print(f"Warning: No data for year {year_val} for {strategy_name} in ideal stocks test.")
                continue

            if strategy_name == 'dgi':
                passed_screening = dgi_screener_for_test(df_year)
                assert len(passed_screening) >= NUM_IDEAL_DGI_STOCKS_PER_YEAR, \
                    (f"Year {year_val}: Expected >= {NUM_IDEAL_DGI_STOCKS_PER_YEAR} DGI ideal, "
                     f"found {len(passed_screening)} for {strategy_name}. "
                     f"Stocks found: {passed_screening['symbol'].tolist() if not passed_screening.empty else 'None'}")
            elif strategy_name == 'exp_fund':
                strategy_params = {'growth_threshold': 0.2, 'top_n': 10}
                # sales_growth_5y and market_cap_rank are optional for exp_fund
                # .fillna ensures that missing data leads to failing the condition
                growth_filter_condition = (
                        (df_year.get('sales_growth_5y', pd.Series(dtype=float)).fillna(float('-inf')) >=
                         strategy_params['growth_threshold']) &
                        (df_year.get('market_cap_rank', pd.Series(dtype=int)).fillna(float('inf')) <=
                         strategy_params['top_n'])
                )
                passed_screening = df_year[growth_filter_condition]
                assert not passed_screening.empty, \
                    f"Year {year_val}: No stocks passed ExpFund screening for {strategy_name}."
                assert len(passed_screening) >= 1, \
                    (f"Year {year_val}: Expected >= 1 stock for ExpFund, "
                     f"got {len(passed_screening)}.")
            elif strategy_name == 'value_play':
                candidates_df = df_year.copy()
                # Use the imported utility function
                if 'sp_quality' in candidates_df.columns:
                    candidates_df['sp_quality_numeric'] = map_sp_quality_to_numeric(
                        candidates_df['sp_quality']
                    )
                else:
                    # If 'sp_quality' column is missing, assign a value that will fail the quality check
                    candidates_df['sp_quality_numeric'] = MIN_SP_QUALITY_NUMERIC - 1  # Use imported constant

                # Use .get() with fillna for all conditions for robustness
                pe_cond = (candidates_df.get('pe_ratio', pd.Series(dtype=float)).fillna(float('inf')) <
                           candidates_df.get('sector_median_pe', pd.Series(dtype=float)).fillna(0) * 0.4)
                pb_cond = (candidates_df.get('pb_ratio', pd.Series(dtype=float)).fillna(float('inf')) < 1.0)
                cr_cond = (candidates_df.get('current_ratio', pd.Series(dtype=float)).fillna(0) > 1.5)
                de_cond = (
                        candidates_df.get('debt_equity', pd.Series(dtype=float)).fillna(float('inf')) <
                        candidates_df.get('industry_avg_debt_equity', pd.Series(dtype=float)).fillna(float('inf')))
                roe_cond = (candidates_df.get('roe', pd.Series(dtype=float)).fillna(float('-inf')) > 0.15)
                eps_g_cond = (candidates_df.get('eps_growth_5y', pd.Series(dtype=float)).fillna(float('-inf')) > 0)

                # spq_cond uses the already prepared 'sp_quality_numeric' and imported constant
                spq_cond = (candidates_df['sp_quality_numeric'] >= MIN_SP_QUALITY_NUMERIC)
                mos_cond = (candidates_df.get('margin_of_safety', pd.Series(dtype=float)).fillna(float('-inf')) > 0.25)

                passed_screening = candidates_df[
                    pe_cond & pb_cond & cr_cond & de_cond &
                    roe_cond & eps_g_cond & spq_cond & mos_cond
                    ]
                assert len(passed_screening) >= NUM_IDEAL_VALUE_STOCKS_PER_YEAR, \
                    (f"Year {year_val}: Expected >= {NUM_IDEAL_VALUE_STOCKS_PER_YEAR} Value ideal, "
                     f"found {len(passed_screening)} for {strategy_name}.")
