import pytest
import os
import tempfile
import pandas as pd
import json

from click.testing import CliRunner
from backtest.cli import cli


class TestBacktest:
    def test_basic_backtest(self, generic_data_file_factory, monkeypatch): # sample_data_file uses default 'dgi'
        """Test basic backtest functionality with default parameters.

        Verifies that the CLI command runs successfully and produces the expected output.
        """
        # Patch where the functions are looked up (in backtest.cli)
        monkeypatch.setattr('backtest.cli.validate_strategy', lambda name: True)
        monkeypatch.setattr('backtest.cli.get_strategy_description', lambda name: "Test Strategy Description from Mock")
        # If list_available_strategies is called by the error path in cli.backtest:
        # monkeypatch.setattr('backtest.cli.list_available_strategies', lambda: [{"name": "exp_fund", "description": "Mocked Error Path Strategy"}])
        data_file = generic_data_file_factory("exp_fund", seed=1)
        runner = CliRunner()
        result = runner.invoke(cli, ["run", "exp_fund", data_file])
        assert result.exit_code == 0, result.output
        assert "Backtest Results:" in result.output
        assert "Sharpe Ratio:" in result.output
        assert "Maximum Drawdown (MDD):" in result.output
        assert "Running backtest for: Test Strategy Description from Mock" in result.output

    def test_custom_parameters(self, generic_data_file_factory, monkeypatch):
        """Test backtest with custom strategy parameters.

        Verifies that the CLI command accepts and processes custom parameter values.
        """
        monkeypatch.setattr('backtest.cli.validate_strategy', lambda name: True)
        monkeypatch.setattr('backtest.cli.get_strategy_description', lambda name: "Custom Test Strategy")
        data_file = generic_data_file_factory("exp_fund", seed=2)
        runner = CliRunner()
        result = runner.invoke(cli, [
            "run",
            "exp_fund",
            data_file,
            '--start-year', '2020',
            '--end-year', '2022',
            '--growth-threshold', '0.15',
            '--top-n', '5',
            '--hybrid-weighting',
        ])
        assert result.exit_code == 0, result.output
        assert "Backtest Results:" in result.output
        assert "Running backtest for: Custom Test Strategy" in result.output

    def test_output_file(self, generic_data_file_factory, monkeypatch):
        """Test saving backtest results to an output file.

        Verifies that results can be successfully saved to a JSON file
        and that the file contains valid data.
        """
        strategy = "exp_fund"
        monkeypatch.setattr('backtest.cli.validate_strategy', lambda name: True)
        monkeypatch.setattr('backtest.cli.get_strategy_description', lambda name: "Output Test Strategy")
        # Test JSON file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as temp:
            output_json_path = temp.name
        data_file = generic_data_file_factory(strategy, seed=3)
        runner = CliRunner()
        result = runner.invoke(cli, [
            "run", strategy, data_file, '--output', output_json_path
        ])
        assert result.exit_code == 0, result.output
        assert "Running backtest for: Output Test Strategy" in result.output
        assert "Full backtest results saved to" in result.output
        assert os.path.exists(output_json_path)
        # Basic content check
        with open(output_json_path, 'r') as f:
            saved_data = json.load(f)
        assert saved_data['backtest_summary']['strategy_name'] == strategy
        assert 'performance_metrics' in saved_data
        os.unlink(output_json_path)
        # Test CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as temp_csv:
            output_csv_path = temp_csv.name
        result_csv = runner.invoke(cli, [
            "run", strategy, data_file, '--output', output_csv_path
        ])
        assert result_csv.exit_code == 0
        assert f"Metrics saved to {output_csv_path}" in result_csv.output
        assert os.path.exists(output_csv_path)
        # Basic content check
        df_csv = pd.read_csv(output_csv_path)
        assert 'sharpe_ratio' in df_csv.columns
        os.unlink(output_csv_path)

    def test_invalid_data_file(self, monkeypatch):
        """Test behavior with a non-existent data file.

        Verifies that the CLI command fails gracefully with an appropriate
        error message when given an invalid file path.
        """
        # Assume strategy is valid for this test, click handles file exists check
        monkeypatch.setattr('backtest.cli.validate_strategy', lambda name: True)

        runner = CliRunner()
        # Test for "dgi" strategy, "nonexistent_file.csv" data_file
        result = runner.invoke(cli, ["run", "dgi", 'nonexistent_file.csv'])

        # Click's Path(exists=True) validation should trigger this
        assert result.exit_code == 2, result.output
        assert "Error: Invalid value for 'DATA_FILE'" in result.output
        assert "File 'nonexistent_file.csv' does not exist" in result.output

    def test_list_command(self, monkeypatch):
        """Test the 'list' command that shows available strategies.

        Verifies that the command runs successfully and lists strategies.
        """

        monkeypatch.setattr('backtest.cli.list_available_strategies', lambda: [
            {"name": "dgi", "description": "Dividend Growth Investing (DGI) (Mocked)"},
            {"name": "value_play", "description": "Value Investing Strategy (Mocked)"},
        ])

        runner = CliRunner()
        result = runner.invoke(cli, ["list"])

        assert result.exit_code == 0, result.output
        assert "Available investment strategies:" in result.output
        assert "dgi" in result.output
        assert "Dividend Growth Investing (DGI) (Mocked)" in result.output
        assert "value_play" in result.output
        assert "Value Investing Strategy (Mocked)" in result.output

    def test_invalid_strategy(self, generic_data_file_factory, monkeypatch):
        """Test behavior with a non-existent strategy.

        Verifies that the command fails gracefully with a helpful error message
        when an invalid strategy name is provided.
        """
        monkeypatch.setattr('backtest.cli.validate_strategy', lambda name: False)
        monkeypatch.setattr('backtest.cli.list_available_strategies', lambda : [
            {"name": "exp_fund", "description": "Investment Strategy: The Exponential Fund (Mocked Error Path)"},
        ])
        data_file = generic_data_file_factory("exp_fund", seed=4)
        runner = CliRunner()
        result = runner.invoke(cli, ["run", "nonexistent_strategy", data_file])
        assert result.exit_code == 1, result.output  # Custom exit code for error
        assert "Error: Strategy 'nonexistent_strategy' not found" in result.output
        assert "Available strategies:" in result.output
        assert "Investment Strategy: The Exponential Fund (Mocked Error Path)" in result.output

    def test_missing_arguments(self):
        """Test that running the command with no arguments shows an error."""
        runner = CliRunner()
        result = runner.invoke(cli, ["run"], prog_name='backtest')

        assert result.exit_code == 2
        assert "Usage: backtest run [OPTIONS] STRATEGY DATA_FILE" in result.output
        assert "Error: Missing argument 'STRATEGY'." in result.output

    def test_missing_data_file_argument(self, monkeypatch):
        """Test behavior when data_file argument is missing for a strategy."""

        # Assume strategy is valid
        monkeypatch.setattr('backtest.cli.validate_strategy', lambda name: True)

        runner = CliRunner()
        # Invoke with strategy but no data file
        result = runner.invoke(cli, ["run", "exp_fund"])

        # Click's Path(exists=True) validation should trigger this
        assert result.exit_code == 2, result.output
        assert "Missing argument 'DATA_FILE'" in result.output

    # Comprehensive CLI option testing
    @pytest.mark.parametrize(
        "test_id, strategy_to_run, cli_options, expected_exit_code, expected_output_snippets, data_strategy_for_fixture",
        [
            # Test with any strategy to ensure options are generally parsed
            ("dgi_basic_options", "dgi", ["--start-year", "2021", "--end-year", "2021", "--top-n", "3"], 0, ["Period: 2021-2021"], "dgi"),

            # Date Range Tests
            ("valid_single_year", "dgi", ["--start-year", "2021", "--end-year", "2021"], 0, ["Period: 2021-2021"], "dgi"),
            ("start_year_equals_end_year", "dgi", ["--start-year", "2022", "--end-year", "2022"], 0, ["Period: 2022-2022", "Sharpe Ratio: 0.00"], "dgi"), # Expects to run but process no actual periods
            ("start_year_greater_than_end_year", "dgi", ["--start-year", "2022", "--end-year", "2020"], 1, ["Error: start_year cannot be greater than end_year"], "dgi"),

            # Numeric Option Tests (for exp_fund)
            ("growth_threshold_zero", "exp_fund", ["--growth-threshold", "0.0"], 0, ["Backtest Results:"], "exp_fund"),
            ("growth_threshold_high", "exp_fund", ["--growth-threshold", "5.0"], 0, ["Warning: Strategy selected no stocks"], "exp_fund"), # Likely no stocks
            ("top_n_one", "exp_fund", ["--top-n", "1"], 0, ["Backtest Results:"], "exp_fund"),
            ("top_n_high", "exp_fund", ["--top-n", "100"], 0, ["Backtest Results:"], "exp_fund"),
            ("ps_threshold_zero_with_overlay", "exp_fund", ["--risk-overlay", "--ps-threshold", "0.0"], 0, ["Backtest Results:"], "exp_fund"),
            ("ps_threshold_high_with_overlay", "exp_fund", ["--risk-overlay", "--ps-threshold", "100.0"], 0, ["Backtest Results:"], "exp_fund"),

            # Transaction Cost Tests
            ("transaction_cost_zero", "value_play", ["--transaction-cost", "0.0"], 0, ["Total Transaction Costs (as % of avg rebalanced value): 0.00%"], "value_play"),
            ("transaction_cost_mid", "value_play", ["--transaction-cost", "0.1"], 0, ["Total Transaction Costs (as % of avg rebalanced value):"], "value_play"), # Check presence
            ("transaction_cost_high", "value_play", ["--transaction-cost", "1.0"], 0, ["Total Transaction Costs (as % of avg rebalanced value):"], "value_play"), # Check presence
            ("transaction_cost_invalid_too_high", "value_play", ["--transaction-cost", "1.1"], 2, ["Invalid value for '--transaction-cost' / '-tc'"], "value_play"), # Click FloatRange

            # Flag Combination Tests
            ("hybrid_weighting_flag", "value_play", ["--hybrid-weighting"], 0, ["Backtest Results:"], "value_play"),
            ("risk_overlay_flag", "value_play", ["--risk-overlay"], 0, ["Backtest Results:"], "value_play"),
            ("dynamic_rebalance_flag", "value_play", ["--dynamic-rebalance"], 0, ["Backtest Results:"], "value_play"),
            ("all_flags_exp_fund", "value_play", ["--hybrid-weighting", "--risk-overlay", "--dynamic-rebalance"], 0, ["Backtest Results:"], "value_play"),
        ]
    )
    def test_run_command_cli_options(
        self,
        test_id,
        strategy_to_run,
        cli_options,
        expected_exit_code,
        expected_output_snippets,
        data_strategy_for_fixture,
        generic_data_file_factory,
        monkeypatch
    ):
        """Test 'backtest run' with various CLI options."""
        monkeypatch.setattr("backtest.cli.validate_strategy", lambda name: True)
        monkeypatch.setattr("backtest.cli.get_strategy_description", lambda name: f"Mocked {name} Strategy")
        # Generate data appropriate for the strategy being tested or a general one
        # Using 2 years of data for most tests to ensure some processing
        data_file = generic_data_file_factory(
            strategy_name=data_strategy_for_fixture,
            start_year=2020,
            end_year=2022
        )
        runner = CliRunner()
        full_cli_args = ["run", strategy_to_run, data_file] + cli_options
        result = runner.invoke(cli, full_cli_args)
        assert result.exit_code == expected_exit_code, \
            f"Test ID '{test_id}' failed. CLI args: {' '.join(full_cli_args)}\nOutput:\n{result.output}"
        if isinstance(expected_output_snippets, str):
            expected_output_snippets = [expected_output_snippets]
        for snippet in expected_output_snippets:
            assert snippet in result.output, \
                f"Test ID '{test_id}' failed. Snippet '{snippet}' not in output.\nOutput:\n{result.output}"
