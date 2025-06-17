from typing import Optional, Dict, Any

import numpy as np
import pandas as pd


def calculate_portfolio_entry_return(portfolio_entry: Optional[Dict[str, Any]]) -> float:
    """Calculates the weighted average return from a portfolio history entry's stocks."""
    if not portfolio_entry or not portfolio_entry.get('stocks'):
        return 0.0

    stocks_df = pd.DataFrame(portfolio_entry['stocks'])
    if stocks_df.empty or 'annual_return' not in stocks_df.columns or 'weight' not in stocks_df.columns:
        # Handle case where portfolio is just CASH with no annual_return specified, or missing columns
        if len(stocks_df) == 1 and stocks_df.iloc[0].get('symbol') == 'CASH':
            cash_return_val = stocks_df.iloc[0].get('annual_return', 0.0)
            # Convert to numeric, coercing errors to NaN
            numeric_cash_return = pd.to_numeric(cash_return_val, errors='coerce')
            # If conversion resulted in NaN (e.g., from a non-numeric string), default to 0.0
            if np.isnan(numeric_cash_return): # Use np.isnan for float NaN check
                numeric_cash_return = 0.0
            return numeric_cash_return / 100.0
        return 0.0

    # Ensure 'annual_return' and 'weight' are numeric, fill NaNs appropriately
    # Convert annual_return from percentage points to decimal
    annual_returns_decimal = pd.to_numeric(stocks_df['annual_return'], errors='coerce').fillna(0.0) / 100.0
    weights_numeric = pd.to_numeric(stocks_df['weight'], errors='coerce').fillna(0.0)

    return (annual_returns_decimal * weights_numeric).sum()
