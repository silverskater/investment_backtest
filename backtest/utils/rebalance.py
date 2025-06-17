from typing import Any, Dict, List, Optional, Tuple

import click
import pandas as pd
import numpy as np

from backtest.constants import (
    NOTIONAL_PORTFOLIO_VALUE_FOR_TRADES,
    DEFAULT_DEVIATION_THRESHOLD
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
