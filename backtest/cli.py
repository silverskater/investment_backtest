"""Command-line interface for the Investment Strategy Backtest Tool.

This module provides a CLI for running backtests on various investment
strategies, listing available strategies, and managing data inputs and outputs.
It uses the Click library to define commands and options.
"""
# Standard library imports
from datetime import datetime
import traceback
from typing import Any, Dict, List, Optional, Tuple

# Third-party imports
import click
import pandas as pd

# Local application/ library-specific imports
from backtest.constants import (
    DEBUG,
    DEFAULT_TRANSACTION_COST
)
from backtest.strategy_manager import (
    execute_strategy,
    get_strategy_description,
    list_available_strategies,
    validate_strategy,
)
from backtest.utils.data_loader import prepare_market_data
from backtest.utils.metrics_calculator import calculate_metrics
from backtest.utils.output_formatter import format_and_output_results
from backtest.utils.portfolio_calculations import calculate_portfolio_entry_return
from backtest.utils.rebalance import rebalance
from data_provider.cli import data_command
# Import AVAILABLE_FETCHERS from data_provider.constants
from data_provider.constants import AVAILABLE_FETCHERS


def _perform_backtest_core_logic(
        strategy_name: str,
        input_file_path: str,
        start_year: int,
        end_year: int,
        growth_threshold: float,
        top_n: int,
        hybrid_weighting: bool,
        dynamic_rebalance: bool,
        risk_overlay: bool,
        ps_threshold: float,
        transaction_cost: float,
        stress_test: str,
        include_delisted: bool,
        rebalance_frequency: str
) -> Tuple[Dict[str, float], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any], str]:
    """Performs the core logic of the backtest.

    Args:
        strategy_name: The name of the strategy to run.
        input_file_path: Path to the CSV or JSON file containing market data.
        start_year: The start year of the backtest period.
        end_year: The end year of the backtest period.
        growth_threshold: Minimum annual sales growth threshold.
        top_n: Number of top companies to include.
        hybrid_weighting: Whether to use hybrid weighting.
        dynamic_rebalance: Whether to enable dynamic rebalancing.
        risk_overlay: Whether to reduce exposure based on P/S ratio.
        ps_threshold: P/S ratio threshold for the risk overlay.
        transaction_cost: Transaction cost as a slippage percentage.
        stress_test: Specific stress test scenario to run.
        include_delisted: Whether to include delisted companies.
        rebalance_frequency: Portfolio rebalancing frequency.

    Returns:
        A tuple containing:
            - final_performance_metrics: Dictionary of performance metrics.
            - yearly_display_returns: List of yearly return information.
            - portfolio_history: List of portfolio states over time.
            - strategy_cli_params: Dictionary of CLI parameters used for the strategy.
            - strategy_description: Description of the executed strategy.
    """
    strategy_description = get_strategy_description(strategy_name) or f"Strategy '{strategy_name}'"
    click.echo(f"Running backtest for: {strategy_description}")

    full_market_data = prepare_market_data(
        input_file_path, stress_test, start_year, end_year, include_delisted
    )

    click.echo(f"Running backtest from {start_year} to {end_year}...")

    strategy_cli_params = {
        'growth_threshold': growth_threshold,
        'top_n': top_n,
        'hybrid_weighting': hybrid_weighting,
        'risk_overlay': risk_overlay,
        'ps_threshold': ps_threshold,
    }

    portfolio_history, yearly_display_returns = _execute_backtest_loop(
        start_year, end_year, full_market_data, strategy_name,
        strategy_cli_params, rebalance_frequency, dynamic_rebalance,
        transaction_cost
    )

    # TODO: Load actual benchmark_returns data based on benchmark_symbol for calculate_metrics.
    final_performance_metrics = calculate_metrics(portfolio_history, benchmark_returns=None)

    return (
        final_performance_metrics,
        yearly_display_returns,
        portfolio_history,
        strategy_cli_params,
        strategy_description
    )


def _run_strategy_backtest(
        strategy_name: str,
        input_file_path: str,
        start_year: int,
        end_year: int,
        growth_threshold: float,
        top_n: int,
        hybrid_weighting: bool,
        dynamic_rebalance: bool,
        risk_overlay: bool,
        ps_threshold: float,
        transaction_cost: float,
        stress_test: str,
        include_delisted: bool,
        rebalance_frequency: str,
        output: Optional[str],
        benchmark: str
) -> int:
    """Runs a generic backtest for the specified investment strategy.

    Args:
        strategy_name: The name of the strategy to run.
        input_file_path: Path to the CSV or JSON file containing market data.
        start_year: The start year of the backtest period.
        end_year: The end year of the backtest period.
        growth_threshold: Minimum annual sales growth threshold.
        top_n: Number of top companies to include.
        hybrid_weighting: Whether to use hybrid weighting.
        dynamic_rebalance: Whether to enable dynamic rebalancing.
        risk_overlay: Whether to reduce exposure based on P/S ratio.
        ps_threshold: P/S ratio threshold for the risk overlay.
        transaction_cost: Transaction cost as a slippage percentage.
        stress_test: Specific stress test scenario to run.
        include_delisted: Whether to include delisted companies.
        rebalance_frequency: Portfolio rebalancing frequency.
        output: Optional output file for detailed results (JSON or CSV).
        benchmark: Benchmark symbol for comparison.

    Returns:
        An exit code (0 for success, 1 for failure).
    """
    try:
        if not validate_strategy(strategy_name):
            click.echo(f"Error: Strategy '{strategy_name}' not found.", err=True)
            available_strategies = list_available_strategies()
            if available_strategies:
                click.echo("Available strategies:", err=True)
                for s_info in available_strategies:
                    click.echo(
                        f"- {s_info.get('name', 'N/A')}: {s_info.get('description', 'No description')}",
                        err=True
                    )
            else:
                click.echo("No strategies available.", err=True)
            return 1

        (
            final_performance_metrics,
            yearly_display_returns,
            portfolio_history,
            strategy_cli_params,
            strategy_description
        ) = _perform_backtest_core_logic(
            strategy_name, input_file_path, start_year, end_year,
            growth_threshold, top_n, hybrid_weighting, dynamic_rebalance,
            risk_overlay, ps_threshold, transaction_cost, stress_test,
            include_delisted, rebalance_frequency
        )

        format_and_output_results(
            final_performance_metrics, output, strategy_name, strategy_description,
            start_year, end_year, input_file_path, rebalance_frequency,
            dynamic_rebalance, transaction_cost, stress_test, include_delisted,
            benchmark, strategy_cli_params, yearly_display_returns, portfolio_history
        )
        return 0  # Success

    except FileNotFoundError as e:
        click.echo(f"Error: Data file operation failed. {e}", err=True)
        return 1
    except ValueError as e:
        click.echo(f"Error: Invalid value or configuration. {e}", err=True)
        return 1
    except IOError as e:  # Catches file save error from format_and_output_results
        click.echo(f"Error writing output file: {e}", err=True)
        return 1
    except Exception as e:  # pylint: disable=broad-except
        if DEBUG:
            # Log the full traceback for an unexpected error.
            click.echo("An unexpected error occurred. Full traceback:", err=True)
            click.echo(traceback.format_exc(), err=True)
        else:
            click.echo(
                f"An unexpected error occurred during the backtest: {type(e).__name__} - {e}",
                err=True
            )
        return 1


def _process_single_period_backtest(
        year: int,
        full_market_data: pd.DataFrame,
        strategy_name: str,
        strategy_cli_params: Dict[str, Any],
        current_portfolio_history: List[Dict[str, Any]],
        rebalance_frequency: str,
        dynamic_rebalance: bool,
        transaction_cost: float
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Processes a single period (year) of the backtest.

    Args:
        year: The current year to process.
        full_market_data: DataFrame containing all market data.
        strategy_name: The name of the strategy being executed.
        strategy_cli_params: CLI parameters relevant to the strategy.
        current_portfolio_history: The portfolio history up to the previous period.
        rebalance_frequency: The frequency of rebalancing.
        dynamic_rebalance: Flag indicating if dynamic rebalancing is active.
        transaction_cost: The transaction cost rate.

    Returns:
        A tuple containing:
            - updated_portfolio_history: The portfolio history including the current period.
            - period_display_return_info: Dictionary with return info for the current period.
    """
    click.echo(f"Processing year {year}...")
    period_date_str = str(year)
    current_year_data = full_market_data[full_market_data['year'] == year].copy()

    period_display_return_info = {'year': year, 'return': 0.0}
    # Work with a mutable copy of the history for this period's processing.
    updated_portfolio_history = list(current_portfolio_history)

    if current_year_data.empty:
        click.echo(f"Warning: No data available for {year}. Skipping rebalance.", err=True)
        if updated_portfolio_history:
            # Calculate return based on the last known portfolio state.
            period_display_return_info['return'] = calculate_portfolio_entry_return(
                updated_portfolio_history[-1]
            )
        return updated_portfolio_history, period_display_return_info

    strategy_output = execute_strategy(
        strategy_name, current_year_data, strategy_cli_params
    )
    target_portfolio_for_period = strategy_output.get('portfolio')

    if target_portfolio_for_period is None or \
            not isinstance(target_portfolio_for_period, pd.DataFrame):
        click.echo(
            f"Warning: Strategy '{strategy_name}' did not return a valid portfolio "
            f"DataFrame for {year}. Assuming 100% cash.",
            err=True
        )
        target_portfolio_for_period = pd.DataFrame([{
            'symbol': 'CASH', 'weight': 1.0, 'annual_return': 0.0, 'share_price': 0.0
        }])

    perform_rebalance_this_period = rebalance_frequency in ['annual', 'quarterly', 'monthly']

    if perform_rebalance_this_period:
        if target_portfolio_for_period.empty and updated_portfolio_history:
            click.echo(
                f"Warning: Strategy selected no stocks for {year}. "
                "Rebalancing to 100% CASH.",
                err=True
            )
            cash_portfolio_df = pd.DataFrame([{
                'symbol': 'CASH', 'weight': 1.0,
                'annual_return': 0.0, 'share_price': 0.0
            }])
            updated_portfolio_history = rebalance(
                history=updated_portfolio_history,
                target_stocks_df=cash_portfolio_df,
                current_year_str=period_date_str,
                dynamic_rebalance_active=False,  # Force rebalance to CASH.
                transaction_cost_rate=transaction_cost
            )
        # Covers initial investment (history is empty) or when target is not empty.
        elif not target_portfolio_for_period.empty or not updated_portfolio_history:
            updated_portfolio_history = rebalance(
                history=updated_portfolio_history,
                target_stocks_df=target_portfolio_for_period,
                current_year_str=period_date_str,
                dynamic_rebalance_active=dynamic_rebalance,
                transaction_cost_rate=transaction_cost
            )
        # If target_portfolio_for_period is empty and history is also empty
        # (initial investment to CASH), it's covered by the elif above,
        # as target_portfolio_for_period would have been set to CASH df.

    # Calculate display return for the current period based on the *newly updated* history.
    if updated_portfolio_history:
        period_display_return_info['return'] = calculate_portfolio_entry_return(
            updated_portfolio_history[-1]
        )

    return updated_portfolio_history, period_display_return_info


def _execute_backtest_loop(
        start_year: int,
        end_year: int,
        full_market_data: pd.DataFrame,
        strategy_name: str,
        strategy_cli_params: Dict[str, Any],
        rebalance_frequency: str,
        dynamic_rebalance: bool,
        transaction_cost: float
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Executes the main annual backtesting loop.

    Args:
        start_year: The start year of the backtest.
        end_year: The end year of the backtest.
        full_market_data: DataFrame containing all market data for the backtest period.
        strategy_name: The name of the strategy to execute.
        strategy_cli_params: CLI parameters for the strategy.
        rebalance_frequency: The frequency of rebalancing.
        dynamic_rebalance: Flag for dynamic rebalancing.
        transaction_cost: Transaction cost rate.

    Returns:
        A tuple containing:
            - portfolio_history: The history of portfolio states.
            - yearly_display_returns: List of yearly return information.
    """
    portfolio_history: List[Dict[str, Any]] = []
    yearly_display_returns: List[Dict[str, Any]] = []

    for year in range(start_year, end_year + 1):
        portfolio_history, period_return_info = _process_single_period_backtest(
            year,
            full_market_data,
            strategy_name,
            strategy_cli_params,
            portfolio_history,  # Pass current history.
            rebalance_frequency,
            dynamic_rebalance,
            transaction_cost
        )
        yearly_display_returns.append(period_return_info)

    return portfolio_history, yearly_display_returns


# --- Click CLI Definition ---

@click.group()
def cli():
    """Investment Strategy Backtest Tool.

    Provides commands to list available strategies, run backtests,
    and manage market data.
    """
    pass  # Click manages context.


cli.add_command(data_command)  # Add the 'data' subcommand


@cli.command(name="run")
@click.argument("strategy_name", metavar="STRATEGY")
@click.option(
    "--input", "-i", "input_file_path",
    metavar="INPUT_FILE_PATH",
    type=click.Path(exists=True, dir_okay=False, readable=True, resolve_path=True),
    required=True,
    help="Path to the input file (CSV or JSON) containing historical market data."
)
@click.option(
    "--start-year", "-s",
    default=lambda: datetime.now().year - 10 - 1,
    type=int,
    show_default="current year - 11",
    help="Start year of the backtest period."
)
@click.option(
    "--end-year", "-e",
    default=lambda: datetime.now().year - 1,
    type=int,
    show_default="current year - 1",
    help="End year of the backtest period."
)
@click.option(
    "--growth-threshold", "-g",
    default=0.2,
    type=click.FloatRange(min=0.0),
    show_default=True,
    help="Strategy: Minimum annual sales growth threshold (e.g., 0.2 for 20%)."
)
@click.option(
    "--top-n", "-n",
    default=10,
    type=click.IntRange(min=1),
    show_default=True,
    help="Strategy: Number of top-ranked companies to include."
)
@click.option(
    "--hybrid-weighting", "-hw",
    is_flag=True,
    help="Strategy: Use hybrid weighting (e.g., 50% market cap, 50% P/S ratio)."
)
@click.option(
    "--dynamic-rebalance", "-dr",
    is_flag=True,
    help="Rebalance: Enable dynamic rebalancing if weights deviate significantly."
)
@click.option(
    "--risk-overlay", "-ro",
    is_flag=True,
    help="Strategy: Reduce exposure if portfolio's average P/S exceeds threshold."
)
@click.option(
    "--ps-threshold", "-pt",
    default=10.0,
    type=click.FloatRange(min=0.0),
    show_default=True,
    help="Strategy: P/S ratio threshold for the risk overlay."
)
@click.option(
    "--transaction-cost", "-tc",
    default=DEFAULT_TRANSACTION_COST,
    type=click.FloatRange(min=0.0, max=1.0),
    show_default=True,
    help="Rebalance: Transaction cost rate (e.g., 0.005 for 0.5%)."
)
@click.option(
    "--stress-test", "-st",
    type=click.Choice(['none', '2008crisis', '2022ratehike'], case_sensitive=False),
    default='none',
    show_default=True,
    help="Data: Apply a specific stress test scenario to market data."
)
@click.option(
    "--include-delisted", "-id",
    is_flag=True,
    help="Data: Include delisted companies in analysis (requires 'is_delisted' column)."
)
@click.option(
    "--rebalance-frequency", "-rf",
    type=click.Choice(['annual', 'quarterly', 'monthly'], case_sensitive=False),
    default='annual',
    show_default=True,
    help="Backtest: Portfolio rebalancing frequency (Note: current loop is annual)."
)
@click.option(
    "--output", "-o",
    type=click.Path(writable=True, dir_okay=False),
    help="Output: File path for saving detailed backtesting results (JSON or CSV)."
)
@click.option(
    "--benchmark", "-b",
    default="SPY",  # Common S&P 500 ETF ticker.
    show_default=True,
    help="Metrics: Benchmark symbol for performance comparison (e.g., SPY)."
)
@click.pass_context
def run_command(ctx: click.Context, **kwargs: Any):
    """Runs a backtest for the specified STRATEGY using the data from the
    provided INPUT_FILE_PATH.
    """
    # The Args: section mentioning ctx and kwargs has been removed from the docstring
    # as Click automatically lists defined arguments and options.
    if kwargs['start_year'] > kwargs['end_year']:
        click.echo("Error: start_year cannot be greater than end_year.", err=True)
        ctx.exit(1)
    exit_code = _run_strategy_backtest(**kwargs)
    ctx.exit(exit_code)


@cli.command(name="list")
def list_command():
    """Lists all available investment strategies and data fetchers."""
    strategies_list = list_available_strategies()

    if not strategies_list:
        click.echo("No investment strategies found.")
    else:
        # Sort the list of strategies alphabetically by name.
        sorted_strategies_list = sorted(strategies_list, key=lambda s: s.get('name', '').lower())
        click.echo(click.style("Available investment strategies for backtesting:", fg="cyan", bold=True))
        for strategy_info in sorted_strategies_list:
            click.echo(
                f"- {click.style(strategy_info.get('name', 'Unknown Strategy'), bold=True)}: "
                f"{strategy_info.get('description', 'No description available.')}"
            )

    click.echo()  # Add a blank line for separation

    # List data fetchers, sorted alphabetically
    click.echo(click.style("Available data fetchers:", fg="cyan", bold=True))
    if not AVAILABLE_FETCHERS:
        click.echo("No data fetchers defined.")
    else:
        for fetcher_name in sorted(AVAILABLE_FETCHERS):  # Sort fetchers alphabetically
            if fetcher_name == 'demo':
                # Make 'demo' stand out and add a note
                click.echo(f"- {click.style(fetcher_name, fg='green', bold=True)}"
                           f"{click.style(' (generates synthetic data)', dim=True)}")
            else:
                click.echo(f"- {fetcher_name}")

    click.echo(
        f"\n{click.style('Note:', bold=True)} Use "
        f"{click.style(f'backtest data <STRATEGY> --fetcher [{'|'.join(AVAILABLE_FETCHERS)}]', fg='yellow')}"
        f" to use a specific fetcher."
    )


if __name__ == "__main__":
    # The pylint disable is for Click's way of handling parameters.
    cli()  # pylint: disable=no-value-for-parameter