import pytest
import os
import pandas as pd
import json

from click.testing import CliRunner
from backtest.cli import cli

class TestExpFundRiskFeatures:

    def test_dynamic_rebalancing(self, generic_data_file_factory, tmp_path, monkeypatch):
        """Test dynamic rebalancing functionality, including 'hold' action."""
        monkeypatch.setattr('backtest.cli.validate_strategy', lambda name: True)
        monkeypatch.setattr('backtest.cli.get_strategy_description', lambda name: "Dynamic Rebalance Test")

        # --- Test Case 1: Deviations below threshold, expect 'hold' ---
        # We need data where the strategy output is consistent for two years
        # and market movements don't cause significant weight drift.
        # This is hard to guarantee with random data without more control.
        # For now, we'll test if it runs.
        # TODO: A more robust test needs specific data.
        start_year = 2020
        end_year = 2022

        # Generate data for 3 years to allow for an initial investment and then a potential hold/rebalance
        data_file_hold = generic_data_file_factory('exp_fund', start_year, end_year, seed=101)
        output_json_path_hold = tmp_path / "dynamic_rebalance_hold_results.json"

        runner = CliRunner()
        result_hold = runner.invoke(cli, [
            "run", "exp_fund", data_file_hold,
            '--start-year', str(start_year),
            '--end-year', str(end_year),
            '--dynamic-rebalance',
            '--output', str(output_json_path_hold)
        ])
        assert result_hold.exit_code == 0, f"Dynamic rebalance (hold case) failed: {result_hold.output}"
        assert "Backtest Results:" in result_hold.output

        # Basic check: if dynamic rebalancing is active, we might see 'hold' actions.
        # A full verification requires inspecting the JSON.
        with open(output_json_path_hold, 'r') as f:
            results_data = json.load(f)
        portfolio_history = results_data.get('portfolio_history', [])

        # This check is indicative; true 'hold' depends on data & strategy logic
        # For a more reliable test of 'hold', you'd need to craft data where the strategy
        # output in year 2 is identical to year 1, and returns are such that weights don't drift.
        # Or, mock execute_strategy to return controlled portfolios.
        hold_action_found = any(period.get('action') == 'hold' for period in portfolio_history)
        rebalance_action_found = any(period.get('action') == 'rebalance' for period in portfolio_history)

        # We expect at least one rebalance (or initial_investment)
        assert rebalance_action_found or any(
            period.get('action') == 'initial_investment' for period in portfolio_history), \
            "No rebalance or initial investment action found in dynamic rebalancing test."
        # If hold_action_found, it means dynamic rebalancing did skip a rebalance.
        # If not, it means all periods triggered a rebalance. Both are valid outcomes depending on data.
        print(f"Dynamic Rebalance (hold case) - Hold action found: {hold_action_found}")

        # --- Test Case 2: New stock added (implicitly covered by normal rebalance) ---
        # Dynamic rebalancing should trigger if the target portfolio composition changes.
        # This is harder to test without very specific data or mocking strategy output.
        # The default behavior of rebalance() already handles new/sold stocks.
        # The dynamic_rebalance flag primarily adds the "deviation check" and "hold" capability.

        # --- Test Case 3: Stock sold (implicitly covered) ---
        # Similar to new stock added.

        # --- Test with a different deviation threshold (using monkeypatch) ---
        # This tests if the rebalance function uses the threshold.
        # Note: This doesn't test the CLI option as it's not exposed.
        monkeypatch.setattr('backtest.cli.DEFAULT_DEVIATION_THRESHOLD', 0.001)  # Very low threshold
        data_file_low_thresh = generic_data_file_factory('exp_fund', start_year, end_year, seed=102)
        output_json_path_low_thresh = tmp_path / "dynamic_rebalance_low_thresh_results.json"

        result_low_thresh = runner.invoke(cli, [
            "run", "exp_fund", data_file_low_thresh,
            '--start-year', str(start_year),
            '--end-year', str(end_year),
            '--dynamic-rebalance',
            '--output', str(output_json_path_low_thresh)
        ])
        assert result_low_thresh.exit_code == 0, f"Dynamic rebalance (low threshold) failed: {result_low_thresh.output}"
        # We'd expect more rebalances (fewer holds) with a lower threshold, but this is hard to assert robustly
        # without controlling the data and strategy output very precisely.
        # The main check is that it runs.

    def test_risk_overlay(self, generic_data_file_factory, tmp_path, monkeypatch):
        """Test risk overlay functionality with various scenarios."""
        monkeypatch.setattr('backtest.cli.validate_strategy', lambda name: True)
        monkeypatch.setattr('backtest.cli.get_strategy_description', lambda name: "Risk Overlay Test")
        runner = CliRunner()

        # --- Case 1: P/S threshold NOT exceeded ---
        # ExpFundStrategy.apply_risk_overlay reduces weights if avg_weighted_ps > ps_threshold
        # We need data where avg_weighted_ps is likely low.
        # Default ps_threshold for CLI is 10.0.
        # generate_example_data for exp_fund creates ps_ratio around 3 to 5 + noise.
        start_year = 2020
        end_year = 2022
        data_file_low_ps = generic_data_file_factory('exp_fund', start_year, end_year, seed=201)
        output_json_path_low_ps = tmp_path / "risk_overlay_low_ps.json"
        result_low_ps = runner.invoke(cli, [
            "run", "exp_fund", data_file_low_ps,
            '--start-year', str(start_year),
            '--end-year', str(end_year),
            '--risk-overlay', '--ps-threshold', '10.0',  # Explicitly set high threshold
            '--output', str(output_json_path_low_ps)
        ])
        assert result_low_ps.exit_code == 0, f"Risk overlay (low P/S) failed: {result_low_ps.output}"

        # To verify no overlay action: check if weights in portfolio_history are ~1.0 (sum)
        # or if a specific "overlay applied" message is ABSENT.
        # This requires inspecting the strategy's internal logic or output.
        # For now, we check that it runs. A deeper check would involve parsing JSON
        # and comparing weights before/after the point where overlay might apply.
        # A simpler proxy: if the overlay *were* applied, total weight might be < 1.
        # However, the strategy normalizes weights *after* the overlay.
        # The ExpFundStrategy.apply_risk_overlay scales weights down.
        # The final normalization in ExpFundStrategy.execute_strategy will bring sum back to 1.
        # So, direct weight sum check after execute_strategy won't reveal overlay.
        # We'd need to inspect weights *inside* apply_risk_overlay or log its action.
        # For now, we assume if it runs and the logic in ExpFundStrategy is correct, it's fine.

        # --- Case 2: Data missing 'ps_ratio' column ---
        def remove_ps_column(df):
            if 'ps_ratio' in df.columns:
                df = df.drop(columns=['ps_ratio'])
            return df

        data_file_no_ps = generic_data_file_factory('exp_fund', start_year, end_year, seed=202, custom_data_modifier=remove_ps_column)
        result_no_ps = runner.invoke(cli, [
            "run", "exp_fund", data_file_no_ps,
            '--start-year', str(start_year),
            '--end-year', str(end_year),
            '--risk-overlay', '--ps-threshold', '2.0'  # Low threshold
        ])
        assert result_no_ps.exit_code == 0, f"Risk overlay (no P/S column) failed: {result_no_ps.output}"

        # ExpFundStrategy.apply_risk_overlay should return early if 'ps_ratio' is missing.
        # We expect no error and the backtest to complete.
        # A warning might be logged by the strategy if it were designed to do so.

        # --- Case 3: `ps_ratio` or `weight` is non-numeric/NaN ---
        def make_ps_non_numeric(df):
            if 'ps_ratio' in df.columns and len(df) > 0:
                # Explicitly cast to object dtype if not already, to avoid pandas FutureWarning
                if df['ps_ratio'].dtype != 'object':
                    df['ps_ratio'] = df['ps_ratio'].astype('object')
                # Introduce some NaNs and a string
                num_rows_to_modify = min(5, len(df) // 2)
                if num_rows_to_modify > 0:
                    indices = df.sample(n=num_rows_to_modify).index
                    # Assigning pd.NA is fine for object or numeric dtypes that support it
                    df.loc[indices, 'ps_ratio'] = pd.NA
                    if len(indices) > 1:
                        df.loc[indices[0], 'ps_ratio'] = "not_a_number"
            return df

        data_file_bad_ps = generic_data_file_factory('exp_fund', start_year, end_year, seed=203, custom_data_modifier=make_ps_non_numeric)
        result_bad_ps = runner.invoke(cli, [
            "run", "exp_fund", data_file_bad_ps,
            '--start-year', str(start_year),
            '--end-year', str(end_year),
            '--risk-overlay', '--ps-threshold', '2.0'
        ])
        assert result_bad_ps.exit_code == 0, f"Risk overlay (bad P/S data) failed: {result_bad_ps.output}"
        # ExpFundStrategy.apply_risk_overlay uses pd.to_numeric(..., errors='coerce')
        # which should handle these gracefully by converting to NaN.
        # The subsequent logic should not crash.

    def test_stress_test_scenarios(self, generic_data_file_factory, tmp_path, monkeypatch):
        """Test stress test scenarios, with basic data verification."""
        monkeypatch.setattr('backtest.cli.validate_strategy', lambda name: True)
        monkeypatch.setattr('backtest.cli.get_strategy_description', lambda name: "Stress Test")
        runner = CliRunner()

        # --- Test 2008 Crisis ---
        # For a deep check, we'd need to know the exact modification factor of apply_stress_test
        # and the strategy's stock selection for that period.
        # Using a small, predictable dataset and checking if 'annual_return' in the output reflects a change.

        # Create a small dataset (e.g., 2 companies, 3 years: 2007, 2008, 2009)
        # to make JSON inspection easier.
        start_year = 2007
        end_year = 2009
        data_file_2008 = generic_data_file_factory('exp_fund', start_year, end_year, seed=301)
        output_json_path_2008 = tmp_path / "stress_2008_results.json"

        result_2008 = runner.invoke(cli, [
            "run", "exp_fund", data_file_2008,
            '--start-year', str(start_year),
            '--end-year', str(end_year),
            '--stress-test', '2008crisis',
            '--output', str(output_json_path_2008)
        ])
        assert result_2008.exit_code == 0, f"Stress test (2008crisis) failed: {result_2008.output}"
        assert "Applying stress test scenario: 2008crisis" in result_2008.output

        # Basic JSON content verification
        with open(output_json_path_2008, 'r') as f:
            results_data_2008 = json.load(f)
        portfolio_history_2008 = results_data_2008.get('portfolio_history', [])
        assert len(portfolio_history_2008) > 0, "Portfolio history is empty for 2008 stress test."

        # Check if returns for 2008 (if stocks were selected) are generally lower
        # This is a heuristic. A precise check needs knowing the original returns.
        returns_in_2008_period = []
        for period_entry in portfolio_history_2008:
            if period_entry.get('date') == '2008':
                for stock in period_entry.get('stocks', []):
                    if 'annual_return' in stock:
                        returns_in_2008_period.append(float(stock['annual_return']))

        if returns_in_2008_period:
            # This is a very loose check. The actual stress test applies a factor.
            # If original returns were positive, stressed returns should be lower or more negative.
            print(f"Returns in 2008 (stressed): {returns_in_2008_period}")
            # A more robust check would involve comparing to non-stressed run or knowing the exact factor.

        # --- Test 2022 Rate Hike ---
        start_year = 2021
        end_year = 2023
        data_file_2022 = generic_data_file_factory('exp_fund', start_year, end_year, seed=302)
        output_json_path_2022 = tmp_path / "stress_2022_results.json"
        result_2022 = runner.invoke(cli, [
            "run", "exp_fund", data_file_2022,
            '--start-year', str(start_year),
            '--end-year', str(end_year),
            '--stress-test', '2022ratehike',
            '--output', str(output_json_path_2022)
        ])
        assert result_2022.exit_code == 0, f"Stress test (2022ratehike) failed: {result_2022.output}"
        assert "Applying stress test scenario: 2022ratehike" in result_2022.output

        with open(output_json_path_2022, 'r') as f:
            results_data_2022 = json.load(f)
        portfolio_history_2022 = results_data_2022.get('portfolio_history', [])
        assert len(portfolio_history_2022) > 0, \
            f"Portfolio history is empty for 2022 rate hike stress test. Output: {result_2022.output}"

    def test_rebalance_frequency(self, generic_data_file_factory, monkeypatch):
        """
        Test different portfolio rebalancing frequencies.
        Note: Current engine processes annually, so this primarily tests option parsing
        and that the backtest doesn't crash with these settings.
        """
        monkeypatch.setattr('backtest.cli.validate_strategy', lambda name: True)
        monkeypatch.setattr('backtest.cli.get_strategy_description', lambda name: "Rebalance Freq Test")
        start_year = 2020
        end_year = 2022
        data_file = generic_data_file_factory('exp_fund', start_year, end_year, seed=401)

        runner = CliRunner()
        result_quarterly = runner.invoke(cli, [
            "run", "exp_fund", data_file,
            '--start-year', str(start_year),
            '--end-year', str(end_year),
            '--rebalance-frequency', 'quarterly'
        ])
        assert result_quarterly.exit_code == 0, f"Rebalance freq (quarterly) failed: {result_quarterly.output}"
        assert "Backtest Results:" in result_quarterly.output

        result_monthly = runner.invoke(cli, [
            "run", "exp_fund", data_file,
            '--start-year', str(start_year),
            '--end-year', str(end_year),
            '--rebalance-frequency', 'monthly'
        ])
        assert result_monthly.exit_code == 0, f"Rebalance freq (monthly) failed: {result_monthly.output}"
        assert "Backtest Results:" in result_monthly.output