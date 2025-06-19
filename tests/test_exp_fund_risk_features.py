"""End-to-end tests for Exponential Fund strategy's risk features via CLI.

These tests focus on verifying the behavior of dynamic rebalancing,
risk overlay, and stress test scenarios when applied to the Exponential Fund
strategy through the command-line interface.
"""
import json
import pytest

import pandas as pd

from click.testing import CliRunner
from backtest.cli import cli


@pytest.mark.e2e
class TestExpFundRiskFeatures:
    """Test suite for Exponential Fund risk management features."""

    def test_dynamic_rebalancing(self, generic_data_file_factory, tmp_path, monkeypatch):
        """Tests dynamic rebalancing functionality, including 'hold' action.

        Verifies that the backtest runs with dynamic rebalancing enabled and
        checks for the presence of 'hold' or 'rebalance' actions in the
        portfolio history. Also tests with a modified deviation threshold.

        Args:
            generic_data_file_factory: Fixture to create sample data files.
            tmp_path: Pytest fixture for temporary directory.
            monkeypatch: Pytest fixture for modifying attributes.
        """
        monkeypatch.setattr('backtest.cli.validate_strategy', lambda name: True)
        monkeypatch.setattr(
            'backtest.cli.get_strategy_description',
            lambda name: "Dynamic Rebalance Test"
        )
        start_year = 2020
        end_year = 2022

        # Generate data for 3 years to allow for an initial investment and
        # then a potential hold/rebalance.
        data_file_hold = generic_data_file_factory(
            'exp_fund', start_year, end_year, seed=101
        )
        output_json_path_hold = tmp_path / "dynamic_rebalance_hold_results.json"

        runner = CliRunner()
        result_hold = runner.invoke(cli, [
            "run", "exp_fund",
            "--input", data_file_hold,
            '--start-year', str(start_year),
            '--end-year', str(end_year),
            '--dynamic-rebalance',
            '--output', str(output_json_path_hold)
        ])
        assert result_hold.exit_code == 0, \
            f"Dynamic rebalance (hold case) failed: {result_hold.output}"
        assert "Backtest Results:" in result_hold.output

        with open(output_json_path_hold, 'r', encoding='utf-8') as f:
            results_data = json.load(f)
        portfolio_history = results_data.get('portfolio_history', [])

        # This check is indicative; true 'hold' depends on data & strategy logic.
        # A more reliable test of 'hold' would need crafted data where the strategy
        # output in year 2 is identical to year 1, and returns don't cause drift.
        hold_action_found = any(
            period.get('action') == 'hold' for period in portfolio_history
        )
        rebalance_action_found = any(
            period.get('action') == 'rebalance' for period in portfolio_history
        )

        # We expect at least one rebalance (or initial_investment).
        assert rebalance_action_found or any(
            period.get('action') == 'initial_investment' for period in portfolio_history
        ), "No rebalance or initial investment action found in dynamic rebalancing test."
        print(f"Dynamic Rebalance (hold case) - Hold action found: {hold_action_found}")

        # Test with a different deviation threshold (modifying the constant).
        monkeypatch.setattr('backtest.constants.DEFAULT_DEVIATION_THRESHOLD', 0.001)
        data_file_low_thresh = generic_data_file_factory(
            'exp_fund', start_year, end_year, seed=102
        )
        output_json_path_low_thresh = tmp_path / "dynamic_rebalance_low_thresh_results.json"

        result_low_thresh = runner.invoke(cli, [
            "run", "exp_fund",
            "--input", data_file_low_thresh,
            '--start-year', str(start_year),
            '--end-year', str(end_year),
            '--dynamic-rebalance',
            '--output', str(output_json_path_low_thresh)
        ])
        assert result_low_thresh.exit_code == 0, \
            f"Dynamic rebalance (low threshold) failed: {result_low_thresh.output}"

    def test_risk_overlay(self, generic_data_file_factory, tmp_path, monkeypatch):
        """Tests risk overlay functionality with various scenarios.

        Covers cases where P/S threshold is not exceeded, 'ps_ratio' column
        is missing, and 'ps_ratio' contains non-numeric data.

        Args:
            generic_data_file_factory: Fixture to create sample data files.
            tmp_path: Pytest fixture for temporary directory.
            monkeypatch: Pytest fixture for modifying attributes.
        """
        monkeypatch.setattr('backtest.cli.validate_strategy', lambda name: True)
        monkeypatch.setattr(
            'backtest.cli.get_strategy_description',
            lambda name: "Risk Overlay Test"
        )
        runner = CliRunner()
        start_year = 2020
        end_year = 2022

        # Case 1: P/S threshold NOT exceeded.
        data_file_low_ps = generic_data_file_factory(
            'exp_fund', start_year, end_year, seed=201
        )
        output_json_path_low_ps = tmp_path / "risk_overlay_low_ps.json"
        result_low_ps = runner.invoke(cli, [
            "run", "exp_fund",
            "--input", data_file_low_ps,
            '--start-year', str(start_year),
            '--end-year', str(end_year),
            '--risk-overlay', '--ps-threshold', '10.0',  # Explicitly set high threshold.
            '--output', str(output_json_path_low_ps)
        ])
        assert result_low_ps.exit_code == 0, \
            f"Risk overlay (low P/S) failed: {result_low_ps.output}"

        # Case 2: Data missing 'ps_ratio' column.
        def remove_ps_column(df):
            if 'ps_ratio' in df.columns:
                df = df.drop(columns=['ps_ratio'])
            return df

        data_file_no_ps = generic_data_file_factory(
            'exp_fund', start_year, end_year, seed=202,
            custom_data_modifier=remove_ps_column
        )
        result_no_ps = runner.invoke(cli, [
            "run", "exp_fund",
            "--input", data_file_no_ps,
            '--start-year', str(start_year),
            '--end-year', str(end_year),
            '--risk-overlay', '--ps-threshold', '2.0'  # Low threshold.
        ])
        assert result_no_ps.exit_code == 0, \
            f"Risk overlay (no P/S column) failed: {result_no_ps.output}"
        # ExpFundStrategy.apply_risk_overlay should return early if 'ps_ratio' is missing.

        # Case 3: `ps_ratio` or `weight` is non-numeric/NaN.
        def make_ps_non_numeric(df):
            if 'ps_ratio' in df.columns and not df.empty:
                if df['ps_ratio'].dtype != 'object': # Avoid Pandas future warning.
                    df['ps_ratio'] = df['ps_ratio'].astype('object')
                num_rows_to_modify = min(5, len(df) // 2)
                if num_rows_to_modify > 0:
                    indices = df.sample(n=num_rows_to_modify).index
                    df.loc[indices, 'ps_ratio'] = pd.NA
                    if len(indices) > 1:
                        df.loc[indices[0], 'ps_ratio'] = "not_a_number"
            return df

        data_file_bad_ps = generic_data_file_factory(
            'exp_fund', start_year, end_year, seed=203,
            custom_data_modifier=make_ps_non_numeric
        )
        result_bad_ps = runner.invoke(cli, [
            "run", "exp_fund",
            "--input", data_file_bad_ps,
            '--start-year', str(start_year),
            '--end-year', str(end_year),
            '--risk-overlay', '--ps-threshold', '2.0'
        ])
        assert result_bad_ps.exit_code == 0, \
            f"Risk overlay (bad P/S data) failed: {result_bad_ps.output}"
        # ExpFundStrategy.apply_risk_overlay uses pd.to_numeric(..., errors='coerce').

    def test_stress_test_scenarios(self, generic_data_file_factory, tmp_path, monkeypatch):
        """Tests stress test scenarios with basic data verification.

        Verifies that the CLI can apply stress tests (2008 crisis, 2022 rate hike)
        and that the output indicates the application of these scenarios.

        Args:
            generic_data_file_factory: Fixture to create sample data files.
            tmp_path: Pytest fixture for temporary directory.
            monkeypatch: Pytest fixture for modifying attributes.
        """
        monkeypatch.setattr('backtest.cli.validate_strategy', lambda name: True)
        monkeypatch.setattr(
            'backtest.cli.get_strategy_description',
            lambda name: "Stress Test"
        )
        runner = CliRunner()

        # Test 2008 Crisis.
        start_year_2008 = 2007
        end_year_2008 = 2009
        data_file_2008 = generic_data_file_factory(
            'exp_fund', start_year_2008, end_year_2008, seed=301
        )
        output_json_path_2008 = tmp_path / "stress_2008_results.json"

        result_2008 = runner.invoke(cli, [
            "run", "exp_fund",
            "--input", data_file_2008,
            '--start-year', str(start_year_2008),
            '--end-year', str(end_year_2008),
            '--stress-test', '2008crisis',
            '--output', str(output_json_path_2008)
        ])
        assert result_2008.exit_code == 0, \
            f"Stress test (2008crisis) failed: {result_2008.output}"
        assert "Applying stress test scenario: 2008crisis" in result_2008.output

        with open(output_json_path_2008, 'r', encoding='utf-8') as f:
            results_data_2008 = json.load(f)
        portfolio_history_2008 = results_data_2008.get('portfolio_history', [])
        assert portfolio_history_2008, "Portfolio history is empty for 2008 stress test."

        # Test 2022 Rate Hike.
        start_year_2022 = 2021
        end_year_2022 = 2023
        data_file_2022 = generic_data_file_factory(
            'exp_fund', start_year_2022, end_year_2022, seed=302
        )
        output_json_path_2022 = tmp_path / "stress_2022_results.json"
        result_2022 = runner.invoke(cli, [
            "run", "exp_fund",
            "--input", data_file_2022,
            '--start-year', str(start_year_2022),
            '--end-year', str(end_year_2022),
            '--stress-test', '2022ratehike',
            '--output', str(output_json_path_2022)
        ])
        assert result_2022.exit_code == 0, \
            f"Stress test (2022ratehike) failed: {result_2022.output}"
        assert "Applying stress test scenario: 2022ratehike" in result_2022.output

        with open(output_json_path_2022, 'r', encoding='utf-8') as f:
            results_data_2022 = json.load(f)
        portfolio_history_2022 = results_data_2022.get('portfolio_history', [])
        assert portfolio_history_2022, \
            f"Portfolio history is empty for 2022 rate hike stress test. Output: {result_2022.output}"

    def test_rebalance_frequency(self, generic_data_file_factory, monkeypatch):
        """Tests different portfolio rebalancing frequencies.

        Note: The current backtesting engine processes annually. This test
        primarily verifies that the rebalance frequency options are parsed
        correctly and do not cause the backtest to crash.

        Args:
            generic_data_file_factory: Fixture to create sample data files.
            monkeypatch: Pytest fixture for modifying attributes.
        """
        monkeypatch.setattr('backtest.cli.validate_strategy', lambda name: True)
        monkeypatch.setattr(
            'backtest.cli.get_strategy_description',
            lambda name: "Rebalance Freq Test"
        )
        start_year = 2020
        end_year = 2022
        data_file = generic_data_file_factory(
            'exp_fund', start_year, end_year, seed=401
        )
        runner = CliRunner()

        result_quarterly = runner.invoke(cli, [
            "run", "exp_fund",
            "--input", data_file,
            '--start-year', str(start_year),
            '--end-year', str(end_year),
            '--rebalance-frequency', 'quarterly'
        ])
        assert result_quarterly.exit_code == 0, \
            f"Rebalance freq (quarterly) failed: {result_quarterly.output}"
        assert "Backtest Results:" in result_quarterly.output

        result_monthly = runner.invoke(cli, [
            "run", "exp_fund",
            "--input", data_file,
            '--start-year', str(start_year),
            '--end-year', str(end_year),
            '--rebalance-frequency', 'monthly'
        ])
        assert result_monthly.exit_code == 0, \
            f"Rebalance freq (monthly) failed: {result_monthly.output}"
        assert "Backtest Results:" in result_monthly.output