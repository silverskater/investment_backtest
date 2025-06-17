"""Utility functions for portfolio rebalancing logic.

This module contains functions to determine rebalancing actions, calculate
trade values, and update portfolio history based on target allocations and
rebalancing strategies (e.g., dynamic rebalancing).
"""
from typing import Any, Dict, List, Optional, Tuple

import click
import numpy as np
import pandas as pd

from backtest.constants import (
    DEFAULT_DEVIATION_THRESHOLD,
    NOTIONAL_PORTFOLIO_VALUE_FOR_TRADES
)


def rebalance(
        history: List[Dict[str, Any]],
        target_stocks_df: pd.DataFrame,
        current_year_str: str,
        transaction_cost_rate: float,
        dynamic_rebalance_active: bool = False,
        deviation_threshold: float = DEFAULT_DEVIATION_THRESHOLD,
        notional_value: float = NOTIONAL_PORTFOLIO_VALUE_FOR_TRADES
) -> List[Dict[str, Any]]:
    """Rebalances the portfolio based on target stocks and strategy settings.

    This function determines the rebalancing action (initial investment,
    rebalance, or hold), calculates necessary trades, and appends a new
    entry to the portfolio history.

    Args:
        history: A list of dictionaries representing the portfolio's state
                 over previous periods.
        target_stocks_df: A DataFrame of target stocks for the current period,
                          including 'symbol', 'weight', and 'share_price'.
        current_year_str: A string representing the current year/period.
        transaction_cost_rate: The rate for calculating transaction costs.
        dynamic_rebalance_active: If True, dynamic rebalancing rules are applied.
        deviation_threshold: The weight deviation threshold to trigger a
                             rebalance in dynamic mode.
        notional_value: The notional value of the portfolio used for calculating
                        trade sizes and costs.

    Returns:
        An updated list of portfolio history entries, including the new
        state after rebalancing for the current period.
    """
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


def _prepare_target_stocks(
    target_stocks_df: pd.DataFrame, current_year_str: str
) -> pd.DataFrame:
    """Prepares the target stocks DataFrame.

    Ensures essential columns ('share_price', 'weight', 'symbol') exist and
    that weights are valid (e.g., sum to 1.0 or are handled appropriately).
    Operates on a copy of the input DataFrame.

    Args:
        target_stocks_df: The DataFrame of target stocks from the strategy.
        current_year_str: String representing the current year, for warnings.

    Returns:
        A prepared DataFrame of target stocks.
    """
    df = target_stocks_df.copy()

    if 'share_price' not in df.columns and not df.empty:
        click.echo(
            f"Warning: 'share_price' missing in target_stocks for {current_year_str}. "
            "Using placeholder 1.0.",
            err=True
        )
        df['share_price'] = 1.0

    if 'weight' not in df.columns and not df.empty:
        click.echo(
            f"Warning: 'weight' missing in target_stocks for {current_year_str}. "
            "Assuming equal weighting.",
            err=True
        )
        df['weight'] = 1.0 / len(df) if len(df) > 0 else 0.0
    elif not df.empty:
        current_total_weight = df['weight'].sum()
        if np.isclose(current_total_weight, 0.0) and not np.isclose(current_total_weight, 1.0):
            click.echo(
                f"Warning: Target stock weights for {current_year_str} sum to zero or "
                f"are invalid ({current_total_weight}). Assuming equal weighting.",
                err=True
            )
            if len(df) > 0:
                df['weight'] = 1.0 / len(df)
        elif not np.isclose(current_total_weight, 1.0) and current_total_weight > 0:
            df['weight'] = df['weight'] / current_total_weight

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
    """Determines the rebalancing action for the current period.

    Actions can be 'initial_investment', 'rebalance', or 'hold'.

    Args:
        last_portfolio_entry: The portfolio state from the previous period.
        prepared_target_df: The prepared DataFrame of target stocks for the current period.
        dynamic_rebalance_active: Flag indicating if dynamic rebalancing is enabled.
        deviation_threshold: The weight deviation threshold for dynamic rebalancing.

    Returns:
        A string representing the rebalancing action.
    """
    if not last_portfolio_entry:
        return 'initial_investment'

    if not dynamic_rebalance_active:
        return 'rebalance'

    last_holdings_map = {
        stock['symbol']: stock for stock in last_portfolio_entry.get('stocks', [])
    }
    target_holdings_map = {
        row['symbol']: row for _, row in prepared_target_df.iterrows()
    } if not prepared_target_df.empty else {}

    # Check for new stocks in target or sold stocks from last period.
    if any(s not in last_holdings_map for s in target_holdings_map) or \
       any(s not in target_holdings_map for s in last_holdings_map):
        return 'rebalance'

    # Check for significant weight deviations.
    for symbol, target_stock_data in target_holdings_map.items():
        last_stock_data = last_holdings_map.get(symbol)
        if last_stock_data:
            target_weight = target_stock_data.get('weight', 0.0)
            last_weight = last_stock_data.get('weight', 0.0)
            if abs(target_weight - last_weight) > deviation_threshold:
                return 'rebalance'

    # Check for rebalance to/from cash if holdings structure changes significantly.
    is_last_cash_only = len(last_holdings_map) == 1 and 'CASH' in last_holdings_map
    is_target_cash_only = not target_holdings_map

    if is_target_cash_only and not is_last_cash_only and last_holdings_map: # Stocks to CASH
        return 'rebalance'
    if not is_target_cash_only and is_last_cash_only: # CASH to Stocks
        return 'rebalance'

    return 'hold'


def _calculate_trade_values(
        current_stocks_map: Dict[str, Dict[str, Any]],
        target_stocks_map: Dict[str, Dict[str, Any]],
        notional_value: float
) -> Tuple[float, float]:
    """Calculates the total monetary value bought and sold.

    Compares current portfolio holdings with target holdings to determine
    the necessary trades based on weight changes.

    Args:
        current_stocks_map: A dictionary mapping symbols to their data in the
                            current portfolio.
        target_stocks_map: A dictionary mapping symbols to their data in the
                           target portfolio.
        notional_value: The notional value of the portfolio.

    Returns:
        A tuple (value_bought, value_sold).
    """
    value_bought_for_period = 0.0
    value_sold_for_period = 0.0
    all_symbols = set(current_stocks_map.keys()) | set(target_stocks_map.keys())

    for symbol in all_symbols:
        current_weight = current_stocks_map.get(symbol, {}).get('weight', 0.0)
        target_weight = target_stocks_map.get(symbol, {}).get('weight', 0.0)
        weight_change = target_weight - current_weight

        if weight_change > 0:  # Buy.
            value_bought_for_period += weight_change * notional_value
        elif weight_change < 0:  # Sell.
            if symbol != 'CASH': # Do not count "selling" CASH for turnover.
                value_sold_for_period += abs(weight_change) * notional_value

    return value_bought_for_period, value_sold_for_period


def _get_portfolio_state_and_trades_for_action(
        action: str,
        last_portfolio_entry: Optional[Dict[str, Any]],
        prepared_target_df: pd.DataFrame,
        current_stocks_map: Dict[str, Dict[str, Any]],
        notional_value: float
) -> Tuple[List[Dict[str, Any]], float, float]:
    """Determines the final stocks for the new entry and calculates trade values.

    Args:
        action: The rebalancing action ('hold', 'initial_investment', 'rebalance').
        last_portfolio_entry: The portfolio state from the previous period.
        prepared_target_df: The prepared DataFrame of target stocks.
        current_stocks_map: Map of current stock holdings.
        notional_value: The notional value of the portfolio.

    Returns:
        A tuple containing:
            - final_stocks_for_entry_list: List of stock dicts for the new entry.
            - value_bought: Total monetary value of stocks bought.
            - value_sold: Total monetary value of stocks sold.
    """
    if action == 'hold':
        final_stocks_for_entry_list = last_portfolio_entry['stocks'] if last_portfolio_entry else []
        value_bought, value_sold = 0.0, 0.0
    else:  # 'initial_investment' or 'rebalance'.
        final_stocks_for_entry_list = prepared_target_df.to_dict('records') \
            if not prepared_target_df.empty else []
        effective_target_map_for_trades = {
            stock['symbol']: stock for stock in final_stocks_for_entry_list
        }
        value_bought, value_sold = _calculate_trade_values(
            current_stocks_map,
            effective_target_map_for_trades,
            notional_value
        )
    return final_stocks_for_entry_list, value_bought, value_sold
