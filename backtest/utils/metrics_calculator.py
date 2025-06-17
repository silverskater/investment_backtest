"""Utility functions for calculating performance metrics from backtest results."""
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from backtest.constants import (
    ANNUAL_RISK_FREE_RATE,
    NOTIONAL_PORTFOLIO_VALUE_FOR_TRADES
)
from backtest.utils.portfolio_calculations import calculate_portfolio_entry_return


def calculate_metrics(
        portfolio_history: List[Dict[str, Any]],
        benchmark_returns: Optional[pd.Series] = None,
        risk_free_rate: float = ANNUAL_RISK_FREE_RATE,
        notional_value_for_ptr: float = NOTIONAL_PORTFOLIO_VALUE_FOR_TRADES
) -> Dict[str, float]:
    """Calculates key performance metrics for a backtest.

    Args:
        portfolio_history: A list of dictionaries, where each dictionary
                           represents the portfolio state at a period end.
        benchmark_returns: Optional pandas Series of benchmark returns, aligned
                           with the portfolio's periodic returns. (Currently unused).
        risk_free_rate: The annual risk-free rate used for calculations like
                        the Sharpe ratio.
        notional_value_for_ptr: The notional portfolio value used for calculating
                                Portfolio Turnover Ratio (PTR).

    Returns:
        A dictionary containing calculated performance metrics such as
        Sharpe ratio, max drawdown, total return, CAGR, PTR, etc.
        Returns a dictionary of zeros if portfolio history is insufficient.
    """
    # Need at least initial state + 1 period for returns.
    if not portfolio_history or len(portfolio_history) <= 1:
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

    # TODO: Implement S&P 500 comparison or other benchmark.
    metrics['sp500_comparison'] = 0.0

    return metrics


def _get_periodic_returns_from_history(
    portfolio_history: List[Dict[str, Any]]
) -> List[float]:
    """Extracts periodic returns from the portfolio history.

    Assumes the first entry in `portfolio_history` is an initial state, and
    returns are calculated from the second entry onwards. 'CASH' positions
    are treated as having a 0% return for the period.

    Args:
        portfolio_history: A list of portfolio state dictionaries.

    Returns:
        A list of periodic returns (as decimals, e.g., 0.10 for 10%).
    """
    periodic_returns = []
    if len(portfolio_history) > 1:
        for i in range(1, len(portfolio_history)):
            period_entry = portfolio_history[i]
            period_return = calculate_portfolio_entry_return(period_entry)
            periodic_returns.append(period_return)
    return periodic_returns


def _calculate_return_group_metrics(periodic_returns: List[float]) -> Dict[str, float]:
    """Calculates total return and average annual return (CAGR).

    Args:
        periodic_returns: A list of periodic returns (as decimals).

    Returns:
        A dictionary with 'total_return' and 'average_annual_return' (CAGR),
        both as percentages.
    """
    if not periodic_returns:
        return {"total_return": 0.0, "average_annual_return": 0.0}

    total_return_factor = np.prod([1 + r for r in periodic_returns])
    total_return_pct = (total_return_factor - 1) * 100

    num_periods = len(periodic_returns)
    cagr = (total_return_factor ** (1 / num_periods) - 1) if num_periods > 0 else 0.0
    cagr_pct = cagr * 100

    return {"total_return": total_return_pct, "average_annual_return": cagr_pct}


def _calculate_risk_group_metrics(
    periodic_returns: List[float], risk_free_rate: float
) -> Dict[str, float]:
    """Calculates Sharpe ratio and maximum drawdown.

    Args:
        periodic_returns: A list of periodic returns (as decimals).
        risk_free_rate: The annual risk-free rate (as a decimal).

    Returns:
        A dictionary with 'sharpe_ratio' (unitless) and 'max_drawdown'
        (as a positive percentage).
    """
    if not periodic_returns:
        return {"sharpe_ratio": 0.0, "max_drawdown": 0.0}

    returns_series = pd.Series(periodic_returns)
    num_periods = len(periodic_returns)

    excess_returns = returns_series - risk_free_rate
    mean_excess_return = excess_returns.mean()
    std_dev_excess_returns = excess_returns.std(ddof=0 if num_periods == 1 else 1)

    if std_dev_excess_returns == 0:
        if mean_excess_return > 0:
            sharpe = float('inf')
        else:  # Handles mean_excess_return <= 0.
            sharpe = 0.0
    else:
        # Assuming annual returns, so sqrt(1) for annualization factor.
        sharpe = mean_excess_return / std_dev_excess_returns * np.sqrt(1)

    initial_value = pd.Series([1.0])
    cumulative_growth_factors = (1 + returns_series).cumprod()
    equity_curve = pd.concat([initial_value, cumulative_growth_factors], ignore_index=True)

    peak = equity_curve.expanding(min_periods=1).max()
    drawdown = (equity_curve - peak) / peak
    max_drawdown_val = drawdown.min() if not drawdown.empty else 0.0
    max_drawdown_pct = abs(max_drawdown_val * 100)

    return {"sharpe_ratio": sharpe, "max_drawdown": max_drawdown_pct}


def _calculate_activity_group_metrics(
    portfolio_history: List[Dict[str, Any]], notional_value_for_ptr: float
) -> Dict[str, float]:
    """Calculates portfolio turnover ratio and total transaction costs.

    Args:
        portfolio_history: A list of portfolio state dictionaries.
        notional_value_for_ptr: The notional portfolio value for PTR calculation.

    Returns:
        A dictionary with 'turnover_ratio' and 'transaction_costs_total',
        both as percentages.
    """
    turnover_events = []
    total_transaction_cost_monetary = 0.0
    num_rebalance_events_for_cost_avg = 0

    if len(portfolio_history) > 1:
        for entry in portfolio_history:
            action = entry.get('action', '')
            if action not in ['initial_investment', 'hold'] and action != '':
                value_bought = entry.get('value_bought', 0.0)
                value_sold = entry.get('value_sold', 0.0)
                if notional_value_for_ptr > 0:
                    turnover_for_event = min(value_bought, value_sold) / notional_value_for_ptr
                    turnover_events.append(turnover_for_event)

                total_transaction_cost_monetary += entry.get('transaction_cost', 0.0)
                if entry.get('transaction_cost', 0.0) > 0 or value_bought > 0 or value_sold > 0:
                    num_rebalance_events_for_cost_avg += 1

    avg_turnover_ratio = np.mean(turnover_events) * 100 if turnover_events else 0.0

    avg_transaction_cost_pct = 0.0
    if num_rebalance_events_for_cost_avg > 0 and notional_value_for_ptr > 0:
        avg_transaction_cost_pct = (
            total_transaction_cost_monetary /
            (num_rebalance_events_for_cost_avg * notional_value_for_ptr)
        ) * 100

    return {"turnover_ratio": avg_turnover_ratio,
            "transaction_costs_total": avg_transaction_cost_pct}


def _get_final_portfolio_size(portfolio_history: List[Dict[str, Any]]) -> int:
    """Calculates the number of non-CASH holdings in the final portfolio period.

    Args:
        portfolio_history: A list of portfolio state dictionaries.

    Returns:
        The number of unique, non-CASH stock symbols in the last portfolio entry.
    """
    if not portfolio_history:
        return 0
    last_entry_stocks = portfolio_history[-1].get('stocks', [])
    if not last_entry_stocks:
        return 0
    size = sum(1 for stock in last_entry_stocks if stock.get('symbol') != 'CASH')
    return size