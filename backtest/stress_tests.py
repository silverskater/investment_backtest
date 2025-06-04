import pandas as pd
import numpy as np

def apply_stress_test(data: pd.DataFrame, scenario: str, start_year: int, end_year: int) -> pd.DataFrame:
    """Apply a stress test scenario to the market data.

    Modifies returns based on the selected stress test scenario to simulate
    historical market conditions.

    Args:
        data: The original market data DataFrame
        scenario: The stress test scenario to apply ('2008crisis' or '2022ratehike')
        start_year: Start year of the backtest period
        end_year: End year of the backtest period

    Returns:
        Modified DataFrame with stress test scenario effects applied
    """
    data_copy = data.copy()

    if scenario == '2008crisis':
        # Simulate 2008 financial crisis effects
        # Value outperformed growth by 35%
        data_copy = apply_2008_crisis(data_copy, start_year, end_year)
    elif scenario == '2022ratehike':
        # Simulate 2022 rate hike cycle
        # ARK Innovation ETF fell 67% vs. S&P -19%
        data_copy = apply_2022_rate_hike(data_copy, start_year, end_year)

    return data_copy

def apply_2008_crisis(data: pd.DataFrame, start_year: int, end_year: int) -> pd.DataFrame:
    """Apply the 2008 financial crisis scenario to market data.

    Simulates the 2008-2009 market conditions where value stocks outperformed
    growth stocks by approximately 35%.

    Args:
        data: Original market data DataFrame
        start_year: Start year of the backtest period
        end_year: End year of the backtest period

    Returns:
        Modified DataFrame with financial crisis effects applied
    """
    # Define the crisis year (adjust based on the start and end of the backtest period)
    crisis_year = max(2008, start_year)
    if crisis_year > end_year:
        return data  # Crisis year not in the backtest period

    # Create a copy to avoid modifying the original
    data_copy = data.copy()

    # Use vectorized operations with boolean masks
    crisis_mask = data_copy['year'] == crisis_year

    if 'annual_return' in data_copy.columns and np.any(crisis_mask):
        # Create a growth stock mask
        growth_threshold = 0.2  # 20% growth
        growth_mask = crisis_mask & (data_copy['sales_growth_5y'] >= growth_threshold)
        value_mask = crisis_mask & (data_copy['sales_growth_5y'] < growth_threshold)

        # Apply return adjustments using vectorized operations
        # Growth stocks: reduce returns by 35%
        data_copy.loc[growth_mask, 'annual_return'] -= 35.0

        # Value stocks: keep returns the same
        # This is already handled implicitly since we're not changing these values

    return data_copy

def apply_2022_rate_hike(data: pd.DataFrame, start_year: int, end_year: int) -> pd.DataFrame:
    """Apply the 2022 interest rate hike scenario to market data.

    Simulates the 2022 interest rate increase market conditions where high-growth
    stocks (like ARK Innovation ETF) fell 67% compared to S&P 500's 19% decline.
    Applies different impacts to stocks based on their growth rates.

    Args:
        data: Original market data DataFrame
        start_year: Start year of the backtest period
        end_year: End year of the backtest period

    Returns:
        Modified DataFrame with rate hike effects applied
    """
    # Define the rate hike year
    rate_hike_year = max(2022, start_year)
    if rate_hike_year > end_year:
        return data  # Rate hike year not in backtest period

    # Create a copy to avoid modifying the original
    data_copy = data.copy()

    # Get data for the rate hike year
    rate_hike_mask = data_copy['year'] == rate_hike_year

    if 'annual_return' in data_copy.columns and np.any(rate_hike_mask):
        # Create masks for different growth categories
        high_growth_mask = rate_hike_mask & (data_copy['sales_growth_5y'] >= 0.3)  # 30% growth
        medium_growth_mask = rate_hike_mask & (data_copy['sales_growth_5y'] >= 0.2) & (data_copy['sales_growth_5y'] < 0.3)
        low_growth_mask = rate_hike_mask & (data_copy['sales_growth_5y'] < 0.2)

        # Apply return adjustments using vectorized operations
        # High growth stocks: significant negative impact
        data_copy.loc[high_growth_mask, 'annual_return'] = np.maximum(
            -67.0, 
            data_copy.loc[high_growth_mask, 'annual_return'] - 48.0
        )

        # Medium growth stocks: moderate negative impact
        data_copy.loc[medium_growth_mask, 'annual_return'] = np.maximum(
            -40.0, 
            data_copy.loc[medium_growth_mask, 'annual_return'] - 30.0
        )

        # Lower growth stocks: less impact (similar to S&P 500)
        data_copy.loc[low_growth_mask, 'annual_return'] = np.maximum(
            -19.0, 
            data_copy.loc[low_growth_mask, 'annual_return'] - 10.0
        )

    return data_copy
