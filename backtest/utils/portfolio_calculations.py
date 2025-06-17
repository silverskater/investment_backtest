"""Utility functions for portfolio-related calculations."""
from typing import Any, Dict, Optional

import pandas as pd


def calculate_portfolio_entry_return(
    portfolio_entry: Optional[Dict[str, Any]]
) -> float:
    """Calculates the weighted average return from a portfolio history entry.

    The return is calculated based on the 'stocks' list within the entry,
    considering 'annual_return' and 'weight' for each stock.

    Args:
        portfolio_entry: A dictionary representing a single period in the
                         portfolio history. Expected to have a 'stocks' key
                         containing a list of stock dictionaries. Each stock
                         dictionary should have 'annual_return' (as percentage)
                         and 'weight'.

    Returns:
        The calculated weighted average return for the portfolio entry as a
        decimal (e.g., 0.10 for 10%). Returns 0.0 if the entry is invalid,
        empty, or necessary data is missing.
    """
    if not portfolio_entry or not portfolio_entry.get('stocks'):
        return 0.0

    stocks_df = pd.DataFrame(portfolio_entry['stocks'])
    if stocks_df.empty or 'annual_return' not in stocks_df.columns or \
       'weight' not in stocks_df.columns:
        # Handle case where portfolio is just CASH with no annual_return,
        # or essential columns are missing.
        if len(stocks_df) == 1 and stocks_df.iloc[0].get('symbol') == 'CASH':
            cash_return_val = stocks_df.iloc[0].get('annual_return', 0.0)
            numeric_cash_return = pd.to_numeric(cash_return_val, errors='coerce')
            if pd.isna(numeric_cash_return): # Use pd.isna for pandas Series/DataFrame context
                numeric_cash_return = 0.0
            return numeric_cash_return / 100.0
        return 0.0

    # Convert 'annual_return' from percentage points to decimal.
    # Coerce errors to NaN, then fill NaN with 0.0.
    annual_returns_decimal = pd.to_numeric(
        stocks_df['annual_return'], errors='coerce'
    ).fillna(0.0) / 100.0
    weights_numeric = pd.to_numeric(
        stocks_df['weight'], errors='coerce'
    ).fillna(0.0)

    return (annual_returns_decimal * weights_numeric).sum()