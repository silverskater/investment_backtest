from typing import Any, Dict, List, Optional

import pandas as pd
import numpy as np

from backtest.constants import (
    ANNUAL_RISK_FREE_RATE,
    NOTIONAL_PORTFOLIO_VALUE_FOR_TRADES
)


def calculate_metrics(
        portfolio_history: List[Dict[str, Any]],
        benchmark_returns: Optional[pd.Series] = None,  # TODO: Placeholder for future use
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

    # TODO: Placeholder for S&P 500 comparison or other benchmark
    # This would require benchmark_returns to be processed aligned with portfolio_returns
    metrics['sp500_comparison'] = 0.0  # Default if not implemented or benchmark_returns not provided

    return metrics


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
