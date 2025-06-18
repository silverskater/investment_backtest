"""Module for applying stress test scenarios to market data.

This module contains functions to modify historical market data to simulate
the effects of specific economic stress events, such as financial crises or
interest rate hike cycles.
"""
import pandas as pd
import numpy as np


def apply_stress_test(
    data: pd.DataFrame, scenario: str, start_year: int, end_year: int
) -> pd.DataFrame:
    """Applies a stress test scenario to the market data.

    Modifies 'annual_return' column based on the selected stress test scenario
    to simulate historical market conditions.

    Args:
        data: The original market data DataFrame.
        scenario: The stress test scenario to apply ('2008crisis' or '2022ratehike').
        start_year: The start year of the backtest period.
        end_year: The end year of the backtest period.

    Returns:
        A modified DataFrame with stress test scenario effects applied.
    """
    data_copy = data.copy()

    if scenario == '2008crisis':
        data_copy = _apply_2008_crisis(data_copy, start_year, end_year)
    elif scenario == '2022ratehike':
        data_copy = _apply_2022_rate_hike(data_copy, start_year, end_year)

    return data_copy


def _apply_2008_crisis(data: pd.DataFrame, start_year: int, end_year: int) -> pd.DataFrame:
    """Applies the 2008 financial crisis scenario to market data.

    Simulates the 2008-2009 market conditions where value stocks outperformed
    growth stocks by approximately 35%. This function modifies the
    'annual_return' for growth stocks in the crisis year.

    Args:
        data: Original market data DataFrame.
        start_year: Start year of the backtest period.
        end_year: End year of the backtest period.

    Returns:
        A modified DataFrame with financial crisis effects applied.
    """
    crisis_year = max(2008, start_year)
    if crisis_year > end_year:
        return data  # Crisis year not in the backtest period.

    data_copy = data.copy()
    crisis_mask = data_copy['year'] == crisis_year

    if 'annual_return' in data_copy.columns and np.any(crisis_mask):
        growth_threshold = 0.2  # 20% sales growth.
        growth_mask = crisis_mask & (data_copy.get('sales_growth_5y', pd.Series(dtype=float)) >= growth_threshold)
        # Value stocks (sales_growth_5y < growth_threshold) are not explicitly modified.

        # Apply return adjustments using vectorized operations.
        # Growth stocks: reduce returns by 35 percentage points.
        data_copy.loc[growth_mask, 'annual_return'] -= 35.0

    return data_copy


def _apply_2022_rate_hike(data: pd.DataFrame, start_year: int, end_year: int) -> pd.DataFrame:
    """Applies the 2022 interest rate hike scenario to market data.

    Simulates the 2022 interest rate increase market conditions where high-growth
    stocks (like ARK Innovation ETF, which fell ~67%) were significantly impacted
    compared to the broader market (S&P 500 fell ~19%).
    This function applies differential impacts to 'annual_return' based on
    sales growth rates.

    Args:
        data: Original market data DataFrame.
        start_year: Start year of the backtest period.
        end_year: End year of the backtest period.

    Returns:
        A modified DataFrame with rate hike effects applied.
    """
    rate_hike_year = max(2022, start_year)
    if rate_hike_year > end_year:
        return data  # Rate hike year not in backtest period.

    data_copy = data.copy()
    rate_hike_mask = data_copy['year'] == rate_hike_year

    if 'annual_return' in data_copy.columns and np.any(rate_hike_mask):
        # Create masks for different growth categories.
        high_growth_mask = rate_hike_mask & (data_copy.get('sales_growth_5y', pd.Series(dtype=float)) >= 0.3)
        medium_growth_mask = rate_hike_mask & \
                             (data_copy.get('sales_growth_5y', pd.Series(dtype=float)) >= 0.2) & \
                             (data_copy.get('sales_growth_5y', pd.Series(dtype=float)) < 0.3)
        low_growth_mask = rate_hike_mask & (data_copy.get('sales_growth_5y', pd.Series(dtype=float)) < 0.2)

        # Apply return adjustments using vectorized operations.
        # High growth stocks: significant negative impact (e.g., -48 points, floor at -67%).
        data_copy.loc[high_growth_mask, 'annual_return'] = np.maximum(
            -67.0,
            data_copy.loc[high_growth_mask, 'annual_return'] - 48.0
        )
        # Medium growth stocks: moderate negative impact (e.g., -30 points, floor at -40%).
        data_copy.loc[medium_growth_mask, 'annual_return'] = np.maximum(
            -40.0,
            data_copy.loc[medium_growth_mask, 'annual_return'] - 30.0
        )
        # Lower growth stocks: less impact (e.g., -10 points, floor at -19%).
        data_copy.loc[low_growth_mask, 'annual_return'] = np.maximum(
            -19.0,
            data_copy.loc[low_growth_mask, 'annual_return'] - 10.0
        )
    return data_copy