"""Command-line interface for the Investment Strategy Backtest Tool.

This module provides a CLI for running backtests on various investment
strategies, listing available strategies, and managing data inputs and outputs.
It uses the Click library to define commands and options.
"""

import json
import os

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import click
import pandas as pd
from pandas.errors import EmptyDataError

from backtest.strategy_manager import (
    execute_strategy,
    get_strategy_description,
    list_available_strategies,
    validate_strategy,
)
from backtest.stress_tests import apply_stress_test

# --- Constants ---
# TODO: If risk-free rate can vary, it might be better as a parameter or loaded from config.
ANNUAL_RISK_FREE_RATE = 0.02
DEFAULT_DEVIATION_THRESHOLD = 0.05
DEFAULT_TRANSACTION_COST = 0.005
# Notional portfolio value for calculating monetary buys/sells from weights
# This assumes the portfolio is notionally this size at each rebalance point
# for determining trade values.
NOTIONAL_PORTFOLIO_VALUE_FOR_TRADES = 1_000_000.0


# --- Helper Functions ---

def rebalance(
        current_portfolio_history: List[Dict[str, Any]],
        target_stocks: pd.DataFrame,  # Expected to have 'symbol', 'weight', 'share_price'
        period_date_str: str,
        dynamic_rebalance_active: bool = False,
        deviation_threshold: float = DEFAULT_DEVIATION_THRESHOLD,
        transaction_cost_rate: float = DEFAULT_TRANSACTION_COST
) -> List[Dict[str, Any]]:
    """Rebalances the portfolio based on new target stock allocations.

    Calculates monetary values for buys and sells based on a notional
    portfolio value, and stores these along with transaction costs.

    Args:
        current_portfolio_history: A list of dictionaries, where each
            dictionary represents a past state of the portfolio.
        target_stocks: A DataFrame of stocks for the current period.
            Expected columns: 'symbol', 'weight' and 'share_price'.
        period_date_str: The date string for the current rebalancing period.
        dynamic_rebalance_active: If True, rebalancing only occurs if
            deviations exceed the threshold or if portfolio composition changes.
        deviation_threshold: The maximum allowed deviation from target weight
            before triggering a rebalance in dynamic mode.
        transaction_cost_rate: The cost rate for transactions (e.g., 0.005
            for 0.5%).

    Returns:
        An updated list of portfolio states, including the new state after
        rebalancing (if any).
    """
    # Handle target_stocks weights and share_price missing.
    if not target_stocks.get('weight', pd.Series(dtype=float)).sum() > 0 and not target_stocks.empty:
        click.echo(
            f"Warning: Target stock weights for {period_date_str} sum to zero or are invalid. "
            "Assuming equal weighting for target stocks.",
            err=True
        )
        if len(target_stocks) > 0:
            target_stocks.loc[:, 'weight'] = 1.0 / len(target_stocks)
        else:  # target_stocks is empty but not None
            target_stocks.loc[:, 'weight'] = 0.0
    target_stocks_records = []
    if not target_stocks.empty:
        if 'share_price' not in target_stocks.columns:
            click.echo(f"Warning: 'share_price' missing in target_stocks for {period_date_str}. Using placeholder 1.0.",
                       err=True)
            target_stocks['share_price'] = 1.0  # Placeholder
        target_stocks_records = target_stocks.to_dict('records')
    updated_portfolio_history = list(current_portfolio_history)  # Make a copy to append to

    if not current_portfolio_history:
        # Initial investment
        # All target weight is "bought" against a cash position.
        value_bought_for_period = NOTIONAL_PORTFOLIO_VALUE_FOR_TRADES * target_stocks['weight'].sum()
        value_sold_for_period = 0.0  # Explicitly 0 for initial investment
        # Transaction cost is on the value bought.
        calculated_transaction_cost = value_bought_for_period * transaction_cost_rate

        updated_portfolio_history.append({
            'date': period_date_str,
            'stocks': target_stocks_records,
            'action': 'initial_investment',
            'value_bought': value_bought_for_period,
            'value_sold': value_sold_for_period,
            'transaction_cost': calculated_transaction_cost
        })
        return updated_portfolio_history

    last_portfolio_state = current_portfolio_history[-1]
    last_stocks_list = last_portfolio_state.get('stocks', [])
    if not isinstance(last_stocks_list, list):
        last_stocks_list = []

    last_holdings = {
        stock['symbol']: stock for stock in last_stocks_list if isinstance(stock, dict)
    }
    # Use a dictionary for target_holdings for efficient lookup by symbol
    target_holdings_map = {
        row['symbol']: row for _, row in target_stocks.iterrows()
    }

    if dynamic_rebalance_active:
        needs_rebalance_flag = False
        # Check for weight deviations or new stocks in the target.
        for symbol, target_item_series in target_holdings_map.items():
            target_weight = target_item_series.get('weight', 0.0)
            if symbol in last_holdings:
                current_weight = last_holdings[symbol].get('weight', 0.0)
                if abs(current_weight - target_weight) > deviation_threshold:
                    needs_rebalance_flag = True
                    break
            elif target_weight > 0:  # New stock in target with positive weight
                needs_rebalance_flag = True
                break
        # Check for sold stocks (in last_holdings but not in target_holdings_map or target weight is 0).
        if not needs_rebalance_flag:
            for symbol, last_item in last_holdings.items():
                if last_item.get('weight', 0.0) > 0:  # Only consider if it was actually held
                    target_item_series = target_holdings_map.get(symbol)
                    if target_item_series is None or target_item_series.get('weight',
                                                                            0.0) == 0.0:  # Sold or weight reduced to 0
                        needs_rebalance_flag = True
                        break

        if not needs_rebalance_flag:
            # No rebalancing needed based on dynamic criteria. Record a 'hold' action.
            updated_portfolio_history.append({
                'date': period_date_str,
                'stocks': last_stocks_list,  # Carry forward the last holdings
                'action': 'hold',
                'value_bought': 0.0,
                'value_sold': 0.0,
                'transaction_cost': 0.0
            })
            return updated_portfolio_history  # Return the updated history

    # Calculate monetary value of buys and sells for 'rebalance' action
    value_bought_for_period = 0.0
    value_sold_for_period = 0.0
    all_involved_symbols = set(last_holdings.keys()) | set(target_holdings_map.keys())

    for symbol in all_involved_symbols:
        old_weight = last_holdings.get(symbol, {}).get('weight', 0.0)
        # Get new weight from target_holdings_map (derived from target_stocks DataFrame)
        new_weight = target_holdings_map.get(symbol, {}).get('weight', 0.0)
        weight_change = new_weight - old_weight
        trade_value = abs(weight_change) * NOTIONAL_PORTFOLIO_VALUE_FOR_TRADES
        # Only account for buys/sells of non-CASH assets for these metrics
        if symbol.upper() != 'CASH': # Make comparison case-insensitive for robustness
            if weight_change > 0:  # Buy non-CASH asset
                value_bought_for_period += trade_value
            elif weight_change < 0:  # Sell non-CASH asset
                value_sold_for_period += trade_value

    # Transaction cost is on the sum of buys and sells (total traded volume)
    calculated_transaction_cost = (value_bought_for_period + value_sold_for_period) * transaction_cost_rate

    updated_portfolio_history.append({
        'date': period_date_str,
        'stocks': target_stocks_records,
        'action': 'rebalance',
        'value_bought': value_bought_for_period,
        'value_sold': value_sold_for_period,
        'transaction_cost': calculated_transaction_cost
    })
    return updated_portfolio_history


def calculate_metrics(
        portfolio_history: List[Dict[str, Any]],
        benchmark_returns: Optional[pd.Series] = None
) -> Dict[str, float]:
    """Calculates key performance metrics for the backtested portfolio.

    Args:
        portfolio_history: A list of portfolio states over time. Each state
            is a dictionary including 'stocks', 'value_bought', 'value_sold',
            and 'transaction_cost'.
        benchmark_returns: An optional pandas Series of benchmark returns
            for the same periods as the portfolio.

    Returns:
        A dictionary containing calculated performance metrics:
            - sharpe_ratio: Risk-adjusted return.
            - max_drawdown: Largest peak-to-trough decline.
            - sp500_comparison: Alpha relative to the benchmark.
            - turnover_ratio: Measure of trading activity.
            - average_annual_return: CAGR.
            - total_return: Total return over the entire period.
            - portfolio_size: Number of stocks in the final portfolio.
            - transaction_costs_total: Average transaction cost per rebalance
                                     event as a percentage of notional value.
    """
    if not portfolio_history:
        return {
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'sp500_comparison': 0.0,
            'turnover_ratio': 0.0,
            'average_annual_return': 0.0,
            'total_return': 0.0,
            'portfolio_size': 0,
            'transaction_costs_total': 0.0
        }

    final_portfolio_state = portfolio_history[-1]
    # Ensure the 'stocks' key exists and is a list, default to empty if not.
    final_stocks_list = final_portfolio_state.get('stocks', [])
    if not isinstance(final_stocks_list, list):  # Defensive check
        final_stocks_list = []

    periodic_returns = []
    # Start from the second entry to calculate returns based on the first period's holdings
    for i in range(1, len(portfolio_history)):
        # The return for period 'i' is based on the portfolio defined at the start of period 'i'
        # which is usually the composition set at the end of period 'i-1' or the start of 'i'.
        # The current logic uses portfolio_history[i]'s stocks for period_history[i]'s return.
        # This implies 'annual_return' in portfolio_history[i]['stocks'] is the return achieved
        # by holding that portfolio during the period ending at portfolio_history[i]['date'].
        current_period_target_stocks = portfolio_history[i].get('stocks', [])
        if not isinstance(current_period_target_stocks, list):
            current_period_target_stocks = []
        weighted_return_sum = 0.0
        total_weight = 0.0
        for stock_data in current_period_target_stocks:
            if not isinstance(stock_data, dict): continue
            try:
                stock_annual_return_pct = float(stock_data.get('annual_return', 0.0))
                stock_weight = float(stock_data.get('weight', 0.0))
                if pd.notna(stock_annual_return_pct) and pd.notna(stock_weight):
                    weighted_return_sum += stock_annual_return_pct * stock_weight
                    total_weight += stock_weight
            except (ValueError, TypeError):
                click.echo(
                    f"Warning: Invalid numeric data for stock {stock_data.get('symbol')} "
                    f"in period {portfolio_history[i].get('date')}.",
                    err=True)
        if total_weight > 0:
            period_avg_return_pct = weighted_return_sum / total_weight
            periodic_returns.append(period_avg_return_pct / 100.0)
        elif portfolio_history[i - 1].get('stocks'): # If previous period had stocks, assume 0% return if current is cash/empty
            periodic_returns.append(0.0)

    # Calculate Portfolio Turnover Ratio (PTR)
    periodic_ptr_fractions = []
    num_rebalance_events_for_ptr = 0
    for i in range(len(portfolio_history)): # Iterate all history for actions
        period_info = portfolio_history[i]
        action = period_info.get('action')
        if action == 'rebalance' or action == 'initial_investment':
            value_bought = period_info.get('value_bought', 0.0)
            value_sold = period_info.get('value_sold', 0.0)
            # Turnover is min of buys or sells for that rebalance event
            turnover_monetary_for_event = min(value_bought, value_sold)
            # PTR for the event, relative to the notional value
            # If it's initial_investment, value_sold is 0, so turnover_monetary is 0.
            # This is correct as PTR measures changes to an *existing* portfolio.
            # However, some definitions might include initial investment as 100% turnover if starting from cash.
            # Standard PTR is usually for ongoing management.
            # Let's only consider 'rebalance' actions for periodic turnover calculation.
            if action == 'rebalance':
                 ptr_for_event_fractional = turnover_monetary_for_event / NOTIONAL_PORTFOLIO_VALUE_FOR_TRADES
                 periodic_ptr_fractions.append(ptr_for_event_fractional)
                 num_rebalance_events_for_ptr +=1

    portfolio_turnover_ratio_pct = 0.0
    if num_rebalance_events_for_ptr > 0 : # Avoid division by zero if only initial investment
        portfolio_turnover_ratio_pct = (sum(periodic_ptr_fractions) / num_rebalance_events_for_ptr) * 100

    # Initialize metrics to default values.
    sharpe_ratio = 0.0
    max_drawdown_pct = 0.0
    benchmark_alpha_pct = 0.0
    cagr_pct = 0.0
    total_return_pct = 0.0
    if periodic_returns:
        returns_series = pd.Series(periodic_returns)
        if not returns_series.empty:
            if returns_series.std(ddof=0) > 0:
                sharpe_ratio = ((returns_series.mean() - ANNUAL_RISK_FREE_RATE) /
                                returns_series.std(ddof=0))
            elif returns_series.mean() > ANNUAL_RISK_FREE_RATE:
                sharpe_ratio = float('inf')  # Sharpe if std is 0 and mean <= risk_free (remains 0.0)

            # --- Max Drawdown Calculation ---
            # Create a series representing the portfolio value over time, starting at 1.
            # (1 + return_period_1), (1 + return_period_1)*(1 + return_period_2), ...
            compounded_growth_factors = (1 + returns_series).cumprod()
            # Prepend the initial portfolio value (1.0) to this series
            portfolio_value_series = pd.concat([pd.Series([1.0]), compounded_growth_factors], ignore_index=True)
            # Calculate the running peak of the portfolio value
            running_max_value = portfolio_value_series.cummax()
            # Calculate drawdown at each point: (current_value - running_peak) / running_peak
            drawdown_series = (portfolio_value_series - running_max_value) / running_max_value
            # Max drawdown is the minimum value in the drawdown series (most negative)
            if not drawdown_series.empty:
                max_drawdown_pct = abs(drawdown_series.min()) * 100

            # --- Benchmark Alpha Calculation ---
            if benchmark_returns is not None and not benchmark_returns.empty:
                aligned_portfolio_returns, aligned_benchmark_returns = \
                    returns_series.align(benchmark_returns, join='inner')
                if not aligned_portfolio_returns.empty and not aligned_benchmark_returns.empty:
                    benchmark_alpha_pct = (aligned_portfolio_returns.mean() - aligned_benchmark_returns.mean()) * 100

            num_periods = len(periodic_returns)
            if num_periods > 0:
                # Clip returns at -100% (-1.0) to prevent issues with (1+r) becoming negative
                safe_returns_for_compounding = returns_series.clip(lower=-1.0)
                total_growth_factor = (1 + safe_returns_for_compounding).prod()
                cagr_pct = ((total_growth_factor ** (1.0 / num_periods)) - 1) * 100
                total_return_pct = (total_growth_factor - 1) * 100

    final_portfolio_size = len([
        s for s in final_stocks_list if isinstance(s, dict) and s.get('symbol') != 'CASH'
    ])

    # Average transaction cost per rebalance event
    # expressed as a percentage of the 'NOTIONAL_PORTFOLIO_VALUE_FOR_TRADES'.
    # It helps gauge the typical cost of rebalancing relative to a fixed notional portfolio size.

    # How it's derived:
    # 1. 'transaction_cost' in 'portfolio_history' is the monetary cost of a single rebalance.
    # 2. Sum these monetary costs for all rebalance events to get 'total_monetary_costs'.
    # 3. Divide 'total_monetary_costs' by the product of 'num_rebalance_events_for_ptr'
    #    and 'NOTIONAL_PORTFOLIO_VALUE_FOR_TRADES'.
    # 4. Multiply by 100 to get the percentage.
    #
    # Essentially: (Average monetary cost per rebalance / NOTIONAL_PORTFOLIO_VALUE_FOR_TRADES) * 100
    avg_transaction_cost_pct = 0.0
    if num_rebalance_events_for_ptr > 0:  # Use the same count as PTR
        total_monetary_costs = sum(
            p.get('transaction_cost', 0.0) for p in portfolio_history if p.get('action') == 'rebalance')
        avg_transaction_cost_pct = (total_monetary_costs / (
                    num_rebalance_events_for_ptr * NOTIONAL_PORTFOLIO_VALUE_FOR_TRADES)) * 100

    return {
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown_pct,
        'sp500_comparison': benchmark_alpha_pct,
        'turnover_ratio': portfolio_turnover_ratio_pct,  # New PTR
        'average_annual_return': cagr_pct,
        'total_return': total_return_pct,
        'portfolio_size': final_portfolio_size,
        'transaction_costs_total': avg_transaction_cost_pct  # Changed to average periodic cost %
    }


def load_market_data(file_path: str) -> pd.DataFrame:
    """Loads market data from a CSV or JSON file.

    Args:
        file_path: The path to the data file.

    Returns:
        A pandas DataFrame containing the market data.

    Raises:
        FileNotFoundError: If the specified file_path does not exist.
        ValueError: If the file format is unsupported (not CSV or JSON).
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")

    file_extension = os.path.splitext(file_path)[1].lower()
    if file_extension == '.csv':
        try:
            return pd.read_csv(file_path)
        except EmptyDataError:
            click.echo(f"Warning: CSV file {file_path} is empty. Returning empty DataFrame.", err=True)
            return pd.DataFrame()
        except Exception as e:
            raise ValueError(f"Error processing CSV file {file_path}: {e}")
    elif file_extension == '.json':
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data_list = json.load(f)
            # Handle empty JSON list specifically
            if isinstance(data_list, list) and not data_list:
                return pd.DataFrame()
            return pd.DataFrame(data_list)
        except json.JSONDecodeError as e:
            raise ValueError(f"Error decoding JSON from {file_path}: {e}")
        except Exception as e:  # Catch other potential errors during file reading/DataFrame creation
            raise ValueError(f"Error processing JSON file {file_path}: {e}")
    else:
        raise ValueError(
            f"Unsupported file format: '{file_extension}'. Please use CSV or JSON."
        )

def _run_strategy_backtest(
            strategy_name: str,
            data_file_path: str,
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
    """Runs a generic backtest for the specified investment strategy."""
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

        strategy_description = get_strategy_description(strategy_name) or f"Strategy '{strategy_name}'"
        click.echo(f"Running backtest for: {strategy_description}")

        full_market_data = _prepare_market_data(
            data_file_path, stress_test, start_year, end_year, include_delisted
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

        # TODO: Load actual benchmark_returns data based on benchmark_symbol.
        final_performance_metrics = calculate_metrics(portfolio_history, benchmark_returns=None)

        try:
            _format_and_output_results(
                final_performance_metrics, output, strategy_name, strategy_description,
                start_year, end_year, data_file_path, rebalance_frequency,
                dynamic_rebalance, transaction_cost, stress_test, include_delisted,
                benchmark, strategy_cli_params, yearly_display_returns, portfolio_history
            )
            return 0  # Success
        except IOError as e:
            click.echo(f"{e}", err=True)
            return 1

    except FileNotFoundError as e:
        click.echo(f"Error: Data file operation failed. {e}", err=True)
        return 1
    except ValueError as e:
        click.echo(f"Error: Invalid value or configuration. {e}", err=True)
        return 1
    except Exception as e:
        click.echo(
            f"An unexpected error occurred during the backtest: {type(e).__name__} - {e}",
            err=True
        )
        return 1


def _prepare_market_data(
        data_file_path: str,
        stress_test: str,
        start_year: int,
        end_year: int,
        include_delisted: bool
) -> pd.DataFrame:
    """Loads and preprocesses market data."""
    click.echo(f"Loading market data from {data_file_path}...")
    full_market_data = load_market_data(data_file_path)

    if stress_test != 'none':
        click.echo(f"Applying stress test scenario: {stress_test} for period {start_year}-{end_year}")
        full_market_data = apply_stress_test(
            full_market_data, stress_test, start_year, end_year
        )

    if include_delisted:
        if 'is_delisted' in full_market_data.columns:
            click.echo("Including delisted companies in analysis.")
        else:
            click.echo(
                "Warning: Delisted company data ('is_delisted' column) "
                "not available in the provided market data.",
                err=True
            )
    return full_market_data


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
    """Executes the main annual backtesting loop."""
    portfolio_history: List[Dict[str, Any]] = []
    yearly_display_returns: List[Dict[str, Any]] = []

    for year in range(start_year, end_year + 1):
        click.echo(f"Processing year {year}...")
        period_date_str = str(year)
        current_year_data = full_market_data[full_market_data['year'] == year].copy()

        if current_year_data.empty:
            click.echo(f"Warning: No data available for {year}. Skipping rebalance.", err=True)
            yearly_display_returns.append({'year': year, 'return': 0.0})
            continue

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

        perform_rebalance_this_period = False
        if rebalance_frequency in ['annual', 'quarterly', 'monthly']:
            perform_rebalance_this_period = True

        if perform_rebalance_this_period:
            if target_portfolio_for_period.empty and portfolio_history:
                click.echo(
                    f"Warning: Strategy selected no stocks for {year}. "
                    "Rebalancing to 100% CASH.",
                    err=True
                )
                cash_portfolio_df = pd.DataFrame([{
                    'symbol': 'CASH', 'weight': 1.0,
                    'annual_return': 0.0, 'share_price': 0.0
                }])
                portfolio_history = rebalance(
                    current_portfolio_history=portfolio_history,
                    target_stocks=cash_portfolio_df,
                    period_date_str=period_date_str,
                    dynamic_rebalance_active=False,
                    transaction_cost_rate=transaction_cost
                )
            elif not target_portfolio_for_period.empty or not portfolio_history:
                portfolio_history = rebalance(
                    current_portfolio_history=portfolio_history,
                    target_stocks=target_portfolio_for_period,
                    period_date_str=period_date_str,
                    dynamic_rebalance_active=dynamic_rebalance,
                    transaction_cost_rate=transaction_cost
                )

        current_period_display_return = 0.0
        if portfolio_history:
            last_period_stocks_df = pd.DataFrame(portfolio_history[-1].get('stocks', []))
            if not last_period_stocks_df.empty and \
               'annual_return' in last_period_stocks_df.columns and \
               'weight' in last_period_stocks_df.columns:
                annual_returns_numeric = pd.to_numeric(
                    last_period_stocks_df['annual_return'], errors='coerce'
                ).fillna(0.0)
                weights_numeric = pd.to_numeric(
                    last_period_stocks_df['weight'], errors='coerce'
                ).fillna(0.0)
                current_period_display_return = (annual_returns_numeric * weights_numeric).sum()
        yearly_display_returns.append({'year': year, 'return': current_period_display_return})

    return portfolio_history, yearly_display_returns


def _format_and_output_results(
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
    """Displays metrics on console and saves results to file.

    Raises:
        IOError: If the output was not written successfully.
    """
    click.echo("\nBacktest Results:")
    click.echo(f"Period: {start_year}-{end_year}")
    click.echo(f"Sharpe Ratio: {final_performance_metrics.get('sharpe_ratio', 0.0):.2f}")
    click.echo(f"Maximum Drawdown (MDD): {final_performance_metrics.get('max_drawdown', 0.0):.2f}%")
    click.echo(
        f"Outperformance vs S&P 500 (Alpha): "
        f"{final_performance_metrics.get('sp500_comparison', 0.0):.2f}%"
    )
    click.echo(f"Portfolio Turnover Ratio (PTR): {final_performance_metrics.get('turnover_ratio', 0.0):.2f}%")
    click.echo(
        f"Compound Annual Growth Rate (CAGR): "
        f"{final_performance_metrics.get('average_annual_return', 0.0):.2f}%"
    )
    click.echo(f"Total Return: {final_performance_metrics.get('total_return', 0.0):.2f}%")
    click.echo(
        f"Final Portfolio Size: {final_performance_metrics.get('portfolio_size', 0)} stocks"
    )
    click.echo(
        f"Total Transaction Costs (as % of avg rebalanced value): "
        f"{final_performance_metrics.get('transaction_costs_total', 0.0):.2f}%"
    )

    if output_path:
        output_file_extension = os.path.splitext(output_path)[1].lower()
        actual_output_path = output_path

        if output_file_extension == '.csv':
            pd.DataFrame([final_performance_metrics]).to_csv(actual_output_path, index=False)
            click.echo(
                f"Metrics saved to {actual_output_path}. "
                "For full details (portfolio history, etc.), use JSON output."
            )
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
            with open(actual_output_path, 'w', encoding='utf-8') as f:
                json.dump(output_content, f, default=str, indent=2)
            click.echo(f"Full backtest results saved to {actual_output_path}")


# --- Click CLI Definition ---

@click.group()
def cli():
    """Investment Strategy Backtest Tool.

    Provides commands to list available strategies and run backtests.
    Use 'backtest run --help' for detailed options on running a backtest.
    """
    pass  # Click manages context


@cli.command(name="run")
@click.argument("strategy_name", metavar="STRATEGY")
@click.argument(
    "data_file_path",
    metavar="DATA_FILE",
    type=click.Path(exists=True, dir_okay=False, readable=True)
)
@click.option(
    "--start-year", "-s",
    default=lambda: datetime.now().year - 10 - 1,  # A 10-year period ending last year
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
    default="SPY",  # Common S&P 500 ETF ticker
    show_default=True,
    help="Metrics: Benchmark symbol for performance comparison (e.g., SPY)."
)
@click.pass_context
def run_command(ctx: click.Context, **kwargs: Any):
    """Runs a backtest for the specified STRATEGY using DATA_FILE."""
    if kwargs['start_year'] > kwargs['end_year']:
        click.echo("Error: start_year cannot be greater than end_year.", err=True)
        ctx.exit(1)
    exit_code = _run_strategy_backtest(**kwargs)
    ctx.exit(exit_code)  # Use ctx.exit for consistency


@cli.command(name="list")
def list_strategies_command():
    """Lists all available investment strategies."""
    strategies_list = list_available_strategies()
    if not strategies_list:
        click.echo("No investment strategies found.")
        return

    # Sort the list of strategies alphabetically by name
    sorted_strategies_list = sorted(strategies_list, key=lambda s: s.get('name', '').lower())

    click.echo("Available investment strategies:")
    for strategy_info in sorted_strategies_list:
        click.echo(
            f"- {strategy_info.get('name', 'Unknown Strategy')}: "
            f"{strategy_info.get('description', 'No description available.')}"
        )


if __name__ == "__main__":
    # The pylint disable is for Click's way of handling parameters.
    cli() # pylint: disable=no-value-for-parameter