"""Utility functions for formatting and outputting backtest results.

This module handles the presentation of backtest metrics to the console
and saving detailed results to files in JSON or CSV format.
"""
import json
import os
from typing import Any, Dict, List, Optional

import click
import pandas as pd


def format_and_output_results(
        final_performance_metrics: Dict[str, float],
        output_path: Optional[str],
        strategy_name: str,
        strategy_description: str,
        start_year: int,
        end_year: int,
        data_file_path: str,
        rebalance_frequency: str,
        dynamic_rebalance: bool,
        transaction_cost: float,
        stress_test: str,
        include_delisted: bool,
        benchmark: str,
        strategy_cli_params: Dict[str, Any],
        yearly_display_returns: List[Dict[str, Any]],
        portfolio_history: List[Dict[str, Any]]
) -> None:
    """Displays metrics on the console and saves results to a file.

    Args:
        final_performance_metrics: Dictionary of overall performance metrics.
        output_path: Optional path to save the results file.
        strategy_name: Name of the executed strategy.
        strategy_description: Description of the strategy.
        start_year: Start year of the backtest.
        end_year: End year of the backtest.
        data_file_path: Path to the input data file.
        rebalance_frequency: Rebalancing frequency used.
        dynamic_rebalance: Whether dynamic rebalancing was enabled.
        transaction_cost: Transaction cost rate.
        stress_test: Name of the stress test scenario applied.
        include_delisted: Whether delisted companies were included.
        benchmark: Benchmark symbol used for comparison.
        strategy_cli_params: CLI parameters specific to the strategy.
        yearly_display_returns: List of yearly return information.
        portfolio_history: Detailed history of portfolio states and actions.

    Raises:
        IOError: If writing to the output file fails.
    """
    _display_results_on_console(final_performance_metrics, start_year, end_year)

    if output_path:
        _save_results_to_file(
            output_path, final_performance_metrics, strategy_name, strategy_description,
            start_year, end_year, data_file_path, rebalance_frequency,
            dynamic_rebalance, transaction_cost, stress_test, include_delisted,
            benchmark, strategy_cli_params, yearly_display_returns, portfolio_history
        )


def _display_results_on_console(
    metrics: Dict[str, float],
    start_year: int,
    end_year: int
) -> None:
    """Displays key backtest metrics on the console.

    Args:
        metrics: A dictionary of calculated performance metrics.
        start_year: The start year of the backtest period.
        end_year: The end year of the backtest period.
    """
    click.echo("\nBacktest Results:")
    click.echo(f"Period: {start_year}-{end_year}")
    click.echo(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0.0):.2f}")
    click.echo(f"Maximum Drawdown (MDD): {metrics.get('max_drawdown', 0.0):.2f}%")
    click.echo(
        f"Outperformance vs S&P 500 (Alpha): "
        f"{metrics.get('sp500_comparison', 0.0):.2f}%"
    )
    click.echo(f"Portfolio Turnover Ratio (PTR): {metrics.get('turnover_ratio', 0.0):.2f}%")
    click.echo(
        f"Compound Annual Growth Rate (CAGR): "
        f"{metrics.get('average_annual_return', 0.0):.2f}%"
    )
    click.echo(f"Total Return: {metrics.get('total_return', 0.0):.2f}%")
    click.echo(
        f"Final Portfolio Size: {metrics.get('portfolio_size', 0)} stocks"
    )
    click.echo(
        f"Total Transaction Costs (as % of avg rebalanced value): "
        f"{metrics.get('transaction_costs_total', 0.0):.2f}%"
    )


def _save_results_to_file(
    output_path: str,
    final_performance_metrics: Dict[str, float],
    strategy_name: str,
    strategy_description: str,
    start_year: int,
    end_year: int,
    data_file_path: str,
    rebalance_frequency: str,
    dynamic_rebalance: bool,
    transaction_cost: float,
    stress_test: str,
    include_delisted: bool,
    benchmark: str,
    strategy_cli_params: Dict[str, Any],
    yearly_display_returns: List[Dict[str, Any]],
    portfolio_history: List[Dict[str, Any]]
) -> None:
    """Saves backtest results to the specified file (JSON or CSV).

    If the `output_path` extension is not '.json', it defaults to saving
    as JSON by appending '.json' to the provided path.

    Args:
        output_path: The path where the results file will be saved.
        final_performance_metrics: Dictionary of overall performance metrics.
        strategy_name: Name of the executed strategy.
        strategy_description: Description of the strategy.
        start_year: Start year of the backtest.
        end_year: End year of the backtest.
        data_file_path: Path to the input data file.
        rebalance_frequency: Rebalancing frequency used.
        dynamic_rebalance: Whether dynamic rebalancing was enabled.
        transaction_cost: Transaction cost rate.
        stress_test: Name of the stress test scenario applied.
        include_delisted: Whether delisted companies were included.
        benchmark: Benchmark symbol used for comparison.
        strategy_cli_params: CLI parameters specific to the strategy.
        yearly_display_returns: List of yearly return information.
        portfolio_history: Detailed history of portfolio states and actions.

    Raises:
        IOError: If writing to the output file fails.
    """
    output_file_extension = os.path.splitext(output_path)[1].lower()
    actual_output_path = output_path

    if output_file_extension == '.csv':
        try:
            pd.DataFrame([final_performance_metrics]).to_csv(actual_output_path, index=False)
            click.echo(
                f"Metrics saved to {actual_output_path}. "
                "For full details (portfolio history, etc.), use JSON output."
            )
        except IOError as e:
            raise IOError(f"Failed to write CSV to {actual_output_path}: {e}") from e
    else:
        if output_file_extension != '.json':
            actual_output_path = f"{output_path}.json"
            click.echo(
                f"Warning: Unsupported output file format '{output_file_extension}'. "
                f"Saving as JSON to '{actual_output_path}'.",
                err=True
            )
        backtest_run_config = {
            'strategy_name': strategy_name,
            'data_file_path': data_file_path,
            'start_year': start_year,
            'end_year': end_year,
            'rebalance_frequency': rebalance_frequency,
            'dynamic_rebalance': dynamic_rebalance,
            'transaction_cost': transaction_cost,
            'stress_test': stress_test,
            'include_delisted': include_delisted,
            'benchmark': benchmark,
            **strategy_cli_params
        }
        output_content = {
            'backtest_summary': {
                'strategy_name': strategy_name,
                'strategy_description': strategy_description,
                'period': f"{start_year}-{end_year}",
            },
            'configuration': backtest_run_config,
            'performance_metrics': final_performance_metrics,
            'yearly_display_returns': yearly_display_returns,
            'portfolio_history': portfolio_history
        }
        try:
            with open(actual_output_path, 'w', encoding='utf-8') as f:
                json.dump(output_content, f, default=str, indent=2)
            click.echo(f"Full backtest results saved to {actual_output_path}")
        except IOError as e:
            raise IOError(f"Failed to write JSON to {actual_output_path}: {e}") from e