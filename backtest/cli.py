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
import numpy as np

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


# --- Helper Functions for calculate_metrics ---

def _get_periodic_returns_from_history(portfolio_history: List[Dict[str, Any]]) -> List[float]:
    """
    Extracts periodic returns from the portfolio history.
    Assumes the first entry is an initial state and returns are from the second entry onwards.
    Handles 'CASH' positions as 0% return for that period.
    """
    periodic_returns = []
    if len(portfolio_history) > 1:
        for i in range(1, len(portfolio_history)):
            period_entry = portfolio_history[i]
            if period_entry['stocks'] and period_entry['stocks'][0].get('symbol') == 'CASH':
                # If portfolio is all cash, assume 0% return for that period's performance
                # (unless 'annual_return' is explicitly provided for CASH, which is unusual)
                period_return = period_entry['stocks'][0].get('annual_return', 0.0) / 100.0
            else:
                # Calculate weighted average return for the period
                # Assumes 'annual_return' is in percentage points (e.g., 10.0 for 10%)
                # Assumes 'weight' is a fraction (e.g., 0.5 for 50%)
                current_period_return = sum(
                    stock.get('weight', 0) * (stock.get('annual_return', 0) / 100.0)
                    for stock in period_entry['stocks']
                )
                period_return = current_period_return
            periodic_returns.append(period_return)
    return periodic_returns


def _calculate_return_group_metrics(periodic_returns: List[float]) -> Dict[str, float]:
    """Calculates total return and average annual return (CAGR)."""
    if not periodic_returns:
        return {"total_return": 0.0, "average_annual_return": 0.0}

    # Total Return
    total_return_factor = np.prod([1 + r for r in periodic_returns])
    total_return_pct = (total_return_factor - 1) * 100

    # Average Annual Return (CAGR)
    num_periods = len(periodic_returns)
    cagr = (total_return_factor ** (1 / num_periods) - 1) if num_periods > 0 else 0.0
    cagr_pct = cagr * 100

    return {"total_return": total_return_pct, "average_annual_return": cagr_pct}


def _calculate_risk_group_metrics(periodic_returns: List[float], risk_free_rate: float) -> Dict[str, float]:
    """Calculates Sharpe ratio and maximum drawdown."""
    if not periodic_returns:
        return {"sharpe_ratio": 0.0, "max_drawdown": 0.0}

    returns_series = pd.Series(periodic_returns)
    num_periods = len(periodic_returns)

    # Sharpe Ratio
    excess_returns = returns_series - risk_free_rate
    mean_excess_return = excess_returns.mean()
    std_dev_excess_returns = excess_returns.std(
        ddof=0 if num_periods == 1 else 1)  # ddof=0 for population std if only 1 period

    if std_dev_excess_returns == 0:
        if mean_excess_return > 0:
            sharpe = float('inf')
        elif mean_excess_return == 0:  # Handles returns == risk_free_rate
            sharpe = 0.0
        else:  # mean_excess_return < 0 and std_dev is 0
            sharpe = 0.0  # Or float('-inf'), tests imply 0.0 for this case
    else:
        # Assuming annual returns, so sqrt(1) for annualization factor of Sharpe.
        # If periodic_returns are for a different frequency, this sqrt factor would change.
        sharpe = mean_excess_return / std_dev_excess_returns * np.sqrt(1)

    # Maximum Drawdown
    # Prepend 1 to represent the initial value before any returns
    initial_value = pd.Series([1.0])
    cumulative_growth_factors = (1 + returns_series).cumprod()
    equity_curve = pd.concat([initial_value, cumulative_growth_factors], ignore_index=True)

    peak = equity_curve.expanding(min_periods=1).max()
    drawdown = (equity_curve - peak) / peak
    max_drawdown_val = drawdown.min()
    max_drawdown_pct = abs(max_drawdown_val * 100)

    return {"sharpe_ratio": sharpe, "max_drawdown": max_drawdown_pct}


def _calculate_activity_group_metrics(portfolio_history: List[Dict[str, Any]], notional_value_for_ptr: float) -> Dict[
    str, float]:
    """Calculates portfolio turnover ratio and total transaction costs."""
    turnover_events = []
    total_transaction_cost_monetary = 0
    num_rebalance_events_for_cost_avg = 0

    if len(portfolio_history) > 1:  # Need at least one rebalance/activity event
        for entry in portfolio_history:
            action = entry.get('action', '')
            if action not in ['initial_investment', 'hold'] and action != '':  # Consider rebalances
                value_bought = entry.get('value_bought', 0.0)
                value_sold = entry.get('value_sold', 0.0)
                turnover_for_event = min(value_bought, value_sold) / notional_value_for_ptr
                turnover_events.append(turnover_for_event)

                total_transaction_cost_monetary += entry.get('transaction_cost', 0.0)
                if entry.get('transaction_cost',
                             0.0) > 0 or value_bought > 0 or value_sold > 0:  # Count if actual rebalance activity
                    num_rebalance_events_for_cost_avg += 1

    avg_turnover_ratio = np.mean(turnover_events) * 100 if turnover_events else 0.0

    avg_transaction_cost_pct = 0.0
    if num_rebalance_events_for_cost_avg > 0 and notional_value_for_ptr > 0:
        # Average cost as % of notional value per rebalance event
        avg_transaction_cost_pct = (total_transaction_cost_monetary / (
                num_rebalance_events_for_cost_avg * notional_value_for_ptr)) * 100

    return {"turnover_ratio": avg_turnover_ratio, "transaction_costs_total": avg_transaction_cost_pct}


def _get_final_portfolio_size(portfolio_history: List[Dict[str, Any]]) -> int:
    """Calculates the number of non-CASH holdings in the final portfolio period."""
    if not portfolio_history:
        return 0
    last_entry_stocks = portfolio_history[-1].get('stocks', [])
    if not last_entry_stocks:
        return 0
    # Count stocks that are not 'CASH'
    size = sum(1 for stock in last_entry_stocks if stock.get('symbol') != 'CASH')
    return size


def calculate_metrics(
        portfolio_history: List[Dict[str, Any]],
        benchmark_returns: Optional[pd.Series] = None,  # Placeholder for future use
        risk_free_rate: float = ANNUAL_RISK_FREE_RATE,
        notional_value_for_ptr: float = NOTIONAL_PORTFOLIO_VALUE_FOR_TRADES
) -> Dict[str, float]:
    """
    Calculates key performance metrics for a backtest.
    """
    if not portfolio_history or len(portfolio_history) <= 1:  # Need at least initial state + 1 period for returns
        return {
            'sharpe_ratio': 0.0, 'max_drawdown': 0.0, 'sp500_comparison': 0.0,
            'turnover_ratio': 0.0, 'average_annual_return': 0.0,
            'total_return': 0.0, 'portfolio_size': 0, 'transaction_costs_total': 0.0
        }

    periodic_returns = _get_periodic_returns_from_history(portfolio_history)

    metrics = {}
    metrics.update(_calculate_return_group_metrics(periodic_returns))
    metrics.update(_calculate_risk_group_metrics(periodic_returns, risk_free_rate))
    metrics.update(_calculate_activity_group_metrics(portfolio_history, notional_value_for_ptr))

    metrics['portfolio_size'] = _get_final_portfolio_size(portfolio_history)

    # Placeholder for S&P 500 comparison or other benchmark
    # This would require benchmark_returns to be processed aligned with portfolio_returns
    metrics['sp500_comparison'] = 0.0  # Default if not implemented or benchmark_returns not provided

    return metrics


# --- Helper Functions for rebalance ---

def _prepare_target_stocks(target_stocks_df: pd.DataFrame, current_year_str: str) -> pd.DataFrame:
    """
    Prepares the target stocks DataFrame by ensuring essential columns and valid weights.
    Works on a copy of the input DataFrame.
    """
    df = target_stocks_df.copy()

    if 'share_price' not in df.columns and not df.empty:
        click.echo(
            f"Warning: 'share_price' missing in target_stocks for {current_year_str}. Using placeholder 1.0.",
            err=True
        )
        df['share_price'] = 1.0

    if 'weight' not in df.columns and not df.empty:
        click.echo(
            f"Warning: 'weight' missing in target_stocks for {current_year_str}. Assuming equal weighting.",
            err=True
        )
        df['weight'] = 1.0 / len(df) if len(df) > 0 else 0.0
    elif not df.empty:
        current_total_weight = df['weight'].sum()
        # Check if weights sum to zero (or very close to it), but not if it's an empty target df
        if np.isclose(current_total_weight, 0.0) and not np.isclose(current_total_weight, 1.0):
            click.echo(
                f"Warning: Target stock weights for {current_year_str} sum to zero or are invalid ({current_total_weight}). "
                "Assuming equal weighting for target stocks.",
                err=True
            )
            if len(df) > 0:  # Avoid division by zero if df is empty after all
                df['weight'] = 1.0 / len(df)
        elif not np.isclose(current_total_weight, 1.0):  # Normalize if not 1.0 and not 0.0
            if current_total_weight > 0:  # Avoid division by zero
                df['weight'] = df['weight'] / current_total_weight
            # If current_total_weight is < 0 (highly unlikely with typical strategy outputs)
            # or still 0 after previous checks (e.g. df was empty), weights remain as they are or 0.

    # Ensure all records have a symbol, default to UNKNOWN if missing (though unlikely from strategy)
    if 'symbol' not in df.columns and not df.empty:
        df['symbol'] = [f"UNKNOWN_{i}" for i in range(len(df))]
    elif not df.empty:
        df['symbol'] = df['symbol'].fillna("UNKNOWN_SYMBOL")

    return df


def _determine_rebalance_action(
        last_portfolio_entry: Optional[Dict[str, Any]],
        prepared_target_df: pd.DataFrame,
        dynamic_rebalance_active: bool,
        deviation_threshold: float
) -> str:
    """Determines the rebalancing action: 'initial_investment', 'rebalance', or 'hold'."""
    if not last_portfolio_entry:
        return 'initial_investment'

    if not dynamic_rebalance_active:
        return 'rebalance'

    # Dynamic rebalancing logic
    last_holdings_map = {stock['symbol']: stock for stock in last_portfolio_entry.get('stocks', [])}
    target_holdings_map = {
        row['symbol']: row for _, row in prepared_target_df.iterrows()
    } if not prepared_target_df.empty else {}

    # 1. Check for new stocks in target
    if any(s not in last_holdings_map for s in target_holdings_map):
        return 'rebalance'
    # 2. Check for sold stocks (in last but not in target)
    if any(s not in target_holdings_map for s in last_holdings_map):
        return 'rebalance'
    # 3. Check for weight deviations
    for symbol, target_stock_data in target_holdings_map.items():
        last_stock_data = last_holdings_map.get(symbol)
        if last_stock_data:  # Should always exist due to checks above if sets are same
            target_weight = target_stock_data.get('weight', 0)
            last_weight = last_stock_data.get('weight', 0)
            if abs(target_weight - last_weight) > deviation_threshold:
                return 'rebalance'

    # 4. Check if target portfolio is empty (rebalance to cash) when last holdings were not empty
    #    And ensure last holdings were not already just CASH (which would be an empty target_holdings_map)
    if not target_holdings_map and last_holdings_map and \
            not (len(last_holdings_map) == 1 and 'CASH' in last_holdings_map):
        return 'rebalance'

    # 5. Check if target portfolio has stocks when last holdings were effectively empty (e.g. just CASH)
    if target_holdings_map and (not last_holdings_map or (len(last_holdings_map) == 1 and 'CASH' in last_holdings_map)):
        return 'rebalance'

    return 'hold'


def _calculate_trade_values(
        current_stocks_map: Dict[str, Dict[str, Any]],  # symbol -> stock_data
        target_stocks_map: Dict[str, Dict[str, Any]],  # symbol -> stock_data
        notional_value: float
) -> Tuple[float, float]:
    """Calculates the total value bought and sold to transition from current to target portfolio."""
    value_bought_for_period = 0.0
    value_sold_for_period = 0.0

    all_symbols = set(current_stocks_map.keys()) | set(target_stocks_map.keys())

    for symbol in all_symbols:
        current_weight = current_stocks_map.get(symbol, {}).get('weight', 0.0)
        target_weight = target_stocks_map.get(symbol, {}).get('weight', 0.0)

        weight_change = target_weight - current_weight

        if weight_change > 0:  # Buy
            value_bought_for_period += weight_change * notional_value
        elif weight_change < 0:  # Sell
            # Do not count "selling" CASH as part of value_sold for turnover purposes
            if symbol != 'CASH':
                value_sold_for_period += abs(weight_change) * notional_value

    return value_bought_for_period, value_sold_for_period


def _get_portfolio_state_and_trades_for_action(
        action: str,
        last_portfolio_entry: Optional[Dict[str, Any]],
        prepared_target_df: pd.DataFrame,
        current_stocks_map: Dict[str, Dict[str, Any]],
        notional_value: float
) -> Tuple[List[Dict[str, Any]], float, float]:
    """
    Determines the final stocks for the new portfolio entry and calculates
    trade values based on the rebalance action.
    """
    if action == 'hold':
        # Stocks remain the same as the last period.
        # For trade calculation, target is effectively the same as current.
        final_stocks_for_entry_list = last_portfolio_entry['stocks'] if last_portfolio_entry else []
        value_bought, value_sold = 0.0, 0.0  # Explicitly zero for 'hold'
    else:  # 'initial_investment' or 'rebalance'
        final_stocks_for_entry_list = prepared_target_df.to_dict('records') if not prepared_target_df.empty else []
        effective_target_map_for_trades = {
            stock['symbol']: stock for stock in final_stocks_for_entry_list
        }
        value_bought, value_sold = _calculate_trade_values(
            current_stocks_map,
            effective_target_map_for_trades,
            notional_value
        )
    return final_stocks_for_entry_list, value_bought, value_sold


def rebalance(
        history: List[Dict[str, Any]],
        target_stocks_df: pd.DataFrame,
        current_year_str: str,
        transaction_cost_rate: float,
        dynamic_rebalance_active: bool = False,
        deviation_threshold: float = DEFAULT_DEVIATION_THRESHOLD,
        notional_value: float = NOTIONAL_PORTFOLIO_VALUE_FOR_TRADES
) -> List[Dict[str, Any]]:
    """
    Rebalances the portfolio based on target stocks and strategy settings.
    """
    # Work on a copy, prepare target (handles missing prices, normalizes weights if needed)
    prepared_target_df = _prepare_target_stocks(target_stocks_df, current_year_str)
    last_portfolio_entry = history[-1] if history else None

    action = _determine_rebalance_action(
        last_portfolio_entry,
        prepared_target_df,
        dynamic_rebalance_active,
        deviation_threshold
    )

    current_stocks_map = {
        stock['symbol']: stock for stock in last_portfolio_entry['stocks']
    } if last_portfolio_entry and last_portfolio_entry.get('stocks') else {}

    final_stocks_for_entry_list, value_bought, value_sold = \
        _get_portfolio_state_and_trades_for_action(
            action,
            last_portfolio_entry,
            prepared_target_df,
            current_stocks_map,
            notional_value
        )

    transaction_cost_for_period = (value_bought + value_sold) * transaction_cost_rate

    new_entry = {
        'date': current_year_str,
        'stocks': final_stocks_for_entry_list,
        'action': action,
        'value_bought': round(value_bought, 2),
        'value_sold': round(value_sold, 2),
        'transaction_cost': round(transaction_cost_for_period, 2)
    }

    return history + [new_entry]


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
                    history=portfolio_history,
                    target_stocks_df=cash_portfolio_df,
                    current_year_str=period_date_str,
                    dynamic_rebalance_active=False,
                    transaction_cost_rate=transaction_cost
                )
            elif not target_portfolio_for_period.empty or not portfolio_history:
                portfolio_history = rebalance(
                    history=portfolio_history,
                    target_stocks_df=target_portfolio_for_period,
                    current_year_str=period_date_str,
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