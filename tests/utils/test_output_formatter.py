"""Unit tests for output formatting utilities.

This module tests functions responsible for displaying results on the console
and saving them to files, ensuring correct formatting and handling of
different output types (JSON, CSV).
"""
from unittest.mock import patch, mock_open

import pytest

# noinspection PyProtectedMember
from backtest.utils.output_formatter import (  # pylint: disable=protected-access
    _display_results_on_console,
    _save_results_to_file,
    format_and_output_results
)


@pytest.mark.unit
class TestOutputFormatter:
    """Test suite for output formatting and saving functionalities."""

    @pytest.fixture
    def sample_metrics(self):
        """Provides a sample dictionary of performance metrics."""
        return {
            'sharpe_ratio': 1.5, 'max_drawdown': 10.0, 'sp500_comparison': 5.0,
            'turnover_ratio': 25.0, 'average_annual_return': 12.0,
            'total_return': 50.0, 'portfolio_size': 8, 'transaction_costs_total': 0.1
        }

    @pytest.fixture
    def common_params(self, sample_metrics):
        """Provides a common set of parameters for testing output functions."""
        return {
            "final_performance_metrics": sample_metrics,
            "strategy_name": "test_strategy",
            "strategy_description": "A test strategy.",
            "start_year": 2020,
            "end_year": 2022,
            "data_file_path": "dummy_data.csv",
            "rebalance_frequency": "annual",
            "dynamic_rebalance": False,
            "transaction_cost": 0.005,
            "stress_test": "none",
            "include_delisted": False,
            "benchmark": "SPY",
            "strategy_cli_params": {"param1": "value1"},
            "yearly_display_returns": [
                {"year": 2020, "return": 0.1}, {"year": 2021, "return": 0.2}
            ],
            "portfolio_history": [
                {"date": "2020", "stocks": [], "action": "initial"}
            ]
        }

    @patch('click.echo')
    def test_display_results_on_console(self, mock_echo, sample_metrics):
        """Tests the console display of backtest results."""
        _display_results_on_console(sample_metrics, 2020, 2022)
        mock_echo.assert_any_call("\nBacktest Results:")
        mock_echo.assert_any_call("Period: 2020-2022")
        mock_echo.assert_any_call("Sharpe Ratio: 1.50")
        mock_echo.assert_any_call("Maximum Drawdown (MDD): 10.00%")

    @patch('builtins.open', new_callable=mock_open)
    @patch('pandas.DataFrame.to_csv')
    @patch('click.echo')
    def test_save_results_to_file_csv(
        self, mock_echo, mock_to_csv, mock_file_open, tmp_path, common_params
    ):
        """Tests saving results to a CSV file."""
        output_path = str(tmp_path / "results.csv")
        _save_results_to_file(output_path=output_path, **common_params)
        mock_to_csv.assert_called_once_with(output_path, index=False)
        mock_echo.assert_any_call(
            f"Metrics saved to {output_path}. "
            "For full details (portfolio history, etc.), use JSON output."
        )

    @patch('builtins.open', new_callable=mock_open)
    @patch('json.dump')
    @patch('click.echo')
    def test_save_results_to_file_json(
        self, mock_echo, mock_json_dump, mock_file_open, tmp_path, common_params
    ):
        """Tests saving results to a JSON file."""
        output_path = str(tmp_path / "results.json")
        _save_results_to_file(output_path=output_path, **common_params)
        mock_file_open.assert_called_once_with(output_path, 'w', encoding='utf-8')
        mock_json_dump.assert_called_once()
        args, _ = mock_json_dump.call_args
        saved_content = args[0]
        assert saved_content['backtest_summary']['strategy_name'] == "test_strategy"
        assert saved_content['performance_metrics']['sharpe_ratio'] == 1.5
        mock_echo.assert_any_call(f"Full backtest results saved to {output_path}")

    @patch('builtins.open', new_callable=mock_open)
    @patch('json.dump')
    @patch('click.echo')
    def test_save_results_to_file_unsupported_extension_defaults_to_json(
            self, mock_echo, mock_json_dump, mock_file_open, tmp_path, common_params
    ):
        """Tests that an unsupported file extension defaults to JSON."""
        output_path_txt = str(tmp_path / "results.txt")
        expected_json_path = f"{output_path_txt}.json"
        _save_results_to_file(output_path=output_path_txt, **common_params)

        mock_echo.assert_any_call(
            f"Warning: Unsupported output file format '.txt'. "
            f"Saving as JSON to '{expected_json_path}'.",
            err=True
        )
        mock_file_open.assert_called_once_with(
            expected_json_path, 'w', encoding='utf-8'
        )
        mock_json_dump.assert_called_once()
        mock_echo.assert_any_call(f"Full backtest results saved to {expected_json_path}")

    @patch('backtest.utils.output_formatter._display_results_on_console')
    @patch('backtest.utils.output_formatter._save_results_to_file')
    def test_format_and_output_results_with_output_path(
            self, mock_save, mock_display, common_params
    ):
        """Tests the main formatting function when an output path is provided."""
        output_path = "some/path/results.json"
        format_and_output_results(output_path=output_path, **common_params)
        mock_display.assert_called_once_with(
            common_params["final_performance_metrics"],
            common_params["start_year"],
            common_params["end_year"]
        )
        mock_save.assert_called_once()

    @patch('backtest.utils.output_formatter._display_results_on_console')
    @patch('backtest.utils.output_formatter._save_results_to_file')
    def test_format_and_output_results_no_output_path(
            self, mock_save, mock_display, common_params
    ):
        """Tests the main formatting function when no output path is provided."""
        format_and_output_results(output_path=None, **common_params)
        mock_display.assert_called_once()
        mock_save.assert_not_called()

    @patch('builtins.open', side_effect=IOError("Disk full"))
    @patch('click.echo')
    def test_save_results_to_file_json_io_error(
        self, mock_echo, mock_file_open_raises_ioerror, tmp_path, common_params
    ): # pylint: disable=unused-argument
        """Tests IOError handling when saving JSON results."""
        output_path = str(tmp_path / "results.json")
        with pytest.raises(IOError, match="Failed to write JSON"):
            _save_results_to_file(output_path=output_path, **common_params)

    @patch('pandas.DataFrame.to_csv', side_effect=IOError("Permission denied"))
    @patch('click.echo')
    def test_save_results_to_file_csv_io_error(
        self, mock_echo, mock_to_csv_raises_ioerror, tmp_path, common_params
    ): # pylint: disable=unused-argument
        """Tests IOError handling when saving CSV results."""
        output_path = str(tmp_path / "results.csv")
        with pytest.raises(IOError, match="Failed to write CSV"):
            _save_results_to_file(output_path=output_path, **common_params)