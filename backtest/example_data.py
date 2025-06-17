"""Generates synthetic market data for investment strategy backtesting.

This module provides functionality to create realistic-looking financial data
for multiple companies across several years. The generated data includes
various financial metrics, with attempts to model some correlations between
them. Data generation can be tailored for specific investment strategies.
"""
import os
import sys
from datetime import datetime
from typing import Dict

import click
import numpy as np
import pandas as pd

# --- Constants ---
# Define required and optional columns for each strategy.
STRATEGY_COLUMNS = {
    'exp_fund': {
        'required': ['year', 'symbol', 'market_cap', 'market_cap_rank',
                     'sales_growth_5y', 'annual_return', 'share_price'],
        'optional': ['ps_ratio', 'shares_outstanding'],
    },
    'dgi': {
        'required': ['year', 'symbol', 'share_price', 'div_growth_streak',
                     'payout_ratio', 'eps_cagr_3y', 'roe', 'debt_equity',
                     'industry_debt_equity', 'sp_quality', 'dividend_yield',
                     'div_growth_5y', 'quality_score'],
        'optional': ['market_cap', 'market_cap_rank', 'sales_growth_5y',
                     'annual_return', 'ps_ratio', 'shares_outstanding'],
    },
    'value_play': {
        'required': [
            'year', 'symbol', 'share_price', 'annual_return',  # Basic
            'pe_ratio', 'sector_median_pe', 'pb_ratio', 'sector_median_pb',  # Undervaluation
            'fcf_yield', 'sector_median_fcf_yield', 'tangible_book_value_per_share',
            'debt_equity', 'industry_avg_debt_equity', 'total_debt', 'net_current_asset_value',  # Financial Health
            'current_ratio', 'positive_ni_5y_streak', 'roe',
            'margin_of_safety', 'total_book_value',  # Margin of Safety
            'sp_quality', 'eps_growth_5y', 'significant_insider_activity',  # Quality
            'composite_value_score', 'quality_score',  # For sorting & weighting
            'sector'  # For sector medians
        ],
        'optional': ['market_cap', 'market_cap_rank', 'ps_ratio', 'shares_outstanding'],
    }
}

# Define default parameters for data generation.
DEFAULT_START_YEAR_OFFSET = 10  # Default backtest period is 10 years ending last year.
DEFAULT_NUM_COMPANIES = 50
NUM_IDEAL_DGI_STOCKS_PER_YEAR = 3  # Number of stocks to ensure pass DGI criteria.
NUM_IDEAL_VALUE_STOCKS_PER_YEAR = 4  # Number of stocks to ensure pass Value criteria.


# --- Helper Functions for Data Generation ---

def _generate_core_arrays(
    rng: np.random.Generator, start_year: int, end_year: int, num_companies: int
):
    """Generates core arrays for year, symbol, and market cap ranks.

    Args:
        rng: NumPy random number generator.
        start_year: The first year for data generation.
        end_year: The last year for data generation.
        num_companies: The number of unique companies.

    Returns:
        A tuple containing:
            - years_array: NumPy array of years.
            - symbols_array: NumPy array of stock symbols.
            - market_cap_ranks_array: NumPy array of market cap ranks.
            - total_rows: Total number of rows in the generated arrays.
    """
    num_years = end_year - start_year + 1
    total_rows = num_years * num_companies

    years_array = np.repeat(np.arange(start_year, end_year + 1), num_companies)
    symbols_array = np.tile(
        [f'STOCK{i:03d}' for i in range(1, num_companies + 1)], num_years
    )

    # Generate Market Cap Ranks with Year-to-Year Consistency.
    yearly_ranks_list = []
    current_year_ranks = np.arange(1, num_companies + 1)
    rng.shuffle(current_year_ranks)
    yearly_ranks_list.append(current_year_ranks)

    for _ in range(1, num_years):
        previous_year_ranks = yearly_ranks_list[-1].copy()
        # Swap neighboring ranks with a 20% probability for minor shifts.
        swap_indices = rng.choice(
            num_companies - 1,
            size=int((num_companies - 1) * 0.2),
            replace=False
        )
        for idx in swap_indices:
            previous_year_ranks[idx], previous_year_ranks[idx + 1] = \
                previous_year_ranks[idx + 1], previous_year_ranks[idx]
        yearly_ranks_list.append(previous_year_ranks)

    market_cap_ranks_array = np.concatenate(yearly_ranks_list)

    return years_array, symbols_array, market_cap_ranks_array, total_rows


def _generate_share_prices_over_time(
    rng: np.random.Generator,
    symbols_array: np.ndarray,
    num_companies_total: int,
    total_rows: int,
    base_price_low: float,
    base_price_high: float,
    annual_change_loc: float,
    annual_change_scale: float,
    min_price: float = 1.0
) -> np.ndarray:
    """Generates share prices iteratively over time for all stocks.

    Args:
        rng: NumPy random number generator.
        symbols_array: Full NumPy array of symbols for all rows.
        num_companies_total: Number of unique companies.
        total_rows: Total number of rows to generate prices for.
        base_price_low: Lower bound for initial base share prices.
        base_price_high: Upper bound for initial base share prices.
        annual_change_loc: Mean of the normal distribution for annual price change percentage.
        annual_change_scale: Standard deviation of the normal distribution for annual price change.
        min_price: Minimum allowed share price.

    Returns:
        A NumPy array of generated share prices.
    """
    base_prices = rng.uniform(base_price_low, base_price_high, size=num_companies_total)
    share_prices_array = np.zeros(total_rows)

    # Assumes symbols_array[:num_companies_total] contains unique symbols for the first year.
    unique_symbols_first_year = symbols_array[:num_companies_total]
    stock_base_price_map = {symbol: base_prices[i] for i, symbol in enumerate(unique_symbols_first_year)}

    for i in range(total_rows):
        symbol = symbols_array[i]
        is_first_year_for_stock = (i < num_companies_total)

        if is_first_year_for_stock:
            share_prices_array[i] = stock_base_price_map[symbol]
        else:
            # Assumes data is ordered: all companies for year 1, then all for year 2, etc.
            # So, previous year's data for the same stock is num_companies_total rows behind.
            prev_row_absolute_idx = i - num_companies_total
            prev_price = share_prices_array[prev_row_absolute_idx]
            annual_change_pct = rng.normal(loc=annual_change_loc, scale=annual_change_scale)
            share_prices_array[i] = np.maximum(min_price, prev_price * (1 + annual_change_pct))
    return share_prices_array


def _calculate_annual_returns_from_prices(
    share_prices_array: np.ndarray,
    num_companies_total: int,
    total_rows: int
) -> np.ndarray:
    """Calculates annual returns in percentage from a share prices array.

    Args:
        share_prices_array: NumPy array of share prices.
        num_companies_total: Number of unique companies.
        total_rows: Total number of rows in the share_prices_array.

    Returns:
        A NumPy array of calculated annual returns (as percentages).
    """
    annual_returns_array = np.zeros(total_rows)
    for i in range(total_rows):
        is_not_first_year_for_stock = (i >= num_companies_total)

        if is_not_first_year_for_stock:
            prev_row_absolute_idx = i - num_companies_total
            prev_price = share_prices_array[prev_row_absolute_idx]
            current_price = share_prices_array[i]
            if prev_price > 0:  # Avoid division by zero.
                annual_returns_array[i] = ((current_price - prev_price) / prev_price) * 100.0
            # else: annual_returns_array[i] remains 0.0 (for cases like prev_price <= 0).
        # else: annual_returns_array[i] remains 0.0 (correct for the first year).
    return annual_returns_array


def _generate_exp_fund_data(
    rng: np.random.Generator, years_array: np.ndarray,
    symbols_array: np.ndarray, market_cap_ranks_array: np.ndarray,
    total_rows: int
) -> pd.DataFrame:
    """Generates synthetic data specific to the 'exp_fund' strategy.

    Args:
        rng: NumPy random number generator.
        years_array: NumPy array of years.
        symbols_array: NumPy array of stock symbols.
        market_cap_ranks_array: NumPy array of market cap ranks.
        total_rows: Total number of rows for data generation.

    Returns:
        A pandas DataFrame with data for the 'exp_fund' strategy.
    """
    # Market Cap based on rank.
    base_market_caps = rng.integers(10 ** 10, 10 ** 12, size=total_rows)  # $10B-$1T.
    # Higher ranks (lower numbers) should generally have higher market caps.
    # Using inverse square root of rank for a non-linear scaling.
    rank_adjustment_factor = 1.0 / np.sqrt(market_cap_ranks_array)
    market_caps_array = base_market_caps * rank_adjustment_factor

    # Shares Outstanding and Share Price.
    # Ensure shares_outstanding is not zero to avoid division errors.
    # Range from 50 million to 1 billion shares.
    shares_outstanding_array = rng.integers(50_000_000, 1_000_000_000, size=total_rows)
    share_prices_array = market_caps_array / shares_outstanding_array

    # Growth Rates and Correlated Annual Returns.
    sales_growth_rates_array = rng.uniform(0.1, 0.4, size=total_rows)
    # Returns: base correlated with growth + some random noise.
    growth_correlated_return_component = (
            sales_growth_rates_array * 100 * rng.uniform(0.7, 1.3, size=total_rows)
    )
    random_return_component = rng.uniform(-10, 10, size=total_rows)
    annual_returns_array = growth_correlated_return_component + random_return_component

    # P/S Ratios Correlated with Market Cap.
    # Normalize market caps within each year to a 0-1 range for P/S calculation.
    def normalize_series(series: pd.Series) -> pd.Series:
        min_val, max_val = series.min(), series.max()
        if max_val > min_val:
            return (series - min_val) / (max_val - min_val)
        return pd.Series(0.0, index=series.index)

    temp_df = pd.DataFrame({
        'year': years_array,
        'symbol': symbols_array,
        'market_cap': market_caps_array,
        'market_cap_rank': market_cap_ranks_array,
        'sales_growth_5y': sales_growth_rates_array,
        'annual_return': annual_returns_array,
        'shares_outstanding': shares_outstanding_array,
        'share_price': share_prices_array,
    })

    temp_df['market_cap_normalized_temp'] = temp_df.groupby('year')['market_cap'].transform(
        normalize_series
    )
    # Base P/S: 5-20, larger companies (higher normalized market cap) get higher P/S.
    ps_ratio_base = 5 + (temp_df['market_cap_normalized_temp'] * 15)
    ps_ratio_noise = rng.uniform(-2, 2, size=total_rows)
    temp_df['ps_ratio'] = np.maximum(3.0, ps_ratio_base + ps_ratio_noise)  # Min P/S of 3.

    temp_df = temp_df.drop('market_cap_normalized_temp', axis=1)

    return temp_df


def _craft_ideal_dgi_stocks(
    rng: np.random.Generator,
    metrics_data: Dict[str, np.ndarray],
    num_companies_total: int
):
    """Modifies the metrics_data in-place to ensure some stocks meet ideal DGI criteria.

    Args:
        rng: NumPy random number generator.
        metrics_data: Dictionary of NumPy arrays representing stock metrics.
                      This dictionary is modified in-place.
        num_companies_total: The total number of unique companies.
    """
    symbols_array = metrics_data['symbol']
    unique_symbols = np.unique(symbols_array[:num_companies_total])
    num_ideal_to_craft = min(NUM_IDEAL_DGI_STOCKS_PER_YEAR, num_companies_total)
    if num_ideal_to_craft == 0:
        return
    ideal_symbols_chosen = rng.choice(unique_symbols, size=num_ideal_to_craft, replace=False)

    # Retrieve arrays from metrics_data dictionary.
    div_growth_streak_array = metrics_data['div_growth_streak']
    payout_ratio_array = metrics_data['payout_ratio']
    eps_cagr_array = metrics_data['eps_cagr_3y']
    roe_array = metrics_data['roe']
    debt_equity_array = metrics_data['debt_equity']
    industry_debt_equity_array = metrics_data['industry_debt_equity']
    sp_quality_array = metrics_data['sp_quality']
    dividend_yield_array = metrics_data['dividend_yield']
    div_growth_5y_array = metrics_data['div_growth_5y']
    quality_score_array = metrics_data['quality_score']

    for ideal_sym in ideal_symbols_chosen:
        indices_for_ideal_stock = np.where(symbols_array == ideal_sym)[0]
        num_rows_for_stock = len(indices_for_ideal_stock)
        if num_rows_for_stock == 0:
            continue

        div_growth_streak_array[indices_for_ideal_stock] = rng.integers(10, 31, size=num_rows_for_stock)
        payout_ratio_array[indices_for_ideal_stock] = rng.uniform(0.25, 0.55, size=num_rows_for_stock)
        eps_cagr_array[indices_for_ideal_stock] = rng.uniform(0.055, 0.18, size=num_rows_for_stock)
        roe_array[indices_for_ideal_stock] = rng.uniform(0.155, 0.35, size=num_rows_for_stock)

        ideal_debt_equity = rng.uniform(0.1, 1.0, size=num_rows_for_stock)
        debt_equity_array[indices_for_ideal_stock] = ideal_debt_equity
        industry_debt_equity_array[indices_for_ideal_stock] = ideal_debt_equity * rng.uniform(1.0, 1.5, size=num_rows_for_stock)

        sp_quality_array[indices_for_ideal_stock] = rng.choice(
            ['B+', 'A-', 'A', 'A+'], size=num_rows_for_stock, p=[0.3, 0.3, 0.2, 0.2]
        )
        dividend_yield_array[indices_for_ideal_stock] = rng.uniform(0.02, 0.055, size=num_rows_for_stock)
        div_growth_5y_array[indices_for_ideal_stock] = rng.uniform(0.055, 0.15, size=num_rows_for_stock)
        quality_score_array[indices_for_ideal_stock] = np.maximum(
            0.0, (roe_array[indices_for_ideal_stock] * 0.6) + (eps_cagr_array[indices_for_ideal_stock] * 0.4)
        )


def _generate_dgi_data(
    rng: np.random.Generator, years_array: np.ndarray,
    symbols_array: np.ndarray, market_cap_ranks_array: np.ndarray,
    total_rows: int
) -> pd.DataFrame:
    """Generates synthetic data specific to the 'dgi' strategy.

    Ensures some stocks meet DGI criteria.

    Args:
        rng: NumPy random number generator.
        years_array: NumPy array of years.
        symbols_array: NumPy array of stock symbols.
        market_cap_ranks_array: NumPy array of market cap ranks.
        total_rows: Total number of rows for data generation.

    Returns:
        A pandas DataFrame with data for the 'dgi' strategy.
    """
    num_companies_total = len(np.unique(
        symbols_array[:total_rows // (years_array[-1] - years_array[0] + 1)] if total_rows > 0 else []
    ))

    share_prices_array = _generate_share_prices_over_time(
        rng, symbols_array, num_companies_total, total_rows,
        base_price_low=20.0, base_price_high=200.0,
        annual_change_loc=0.08, annual_change_scale=0.15, min_price=1.0
    )

    streak_base = 30.0 / np.sqrt(market_cap_ranks_array)
    streak_noise = rng.integers(-5, 10, size=total_rows)
    div_growth_streak_array = np.maximum(0, (streak_base + streak_noise)).astype(int)
    div_growth_streak_array = np.maximum(
        div_growth_streak_array, rng.choice([0, 5, 10, 15], size=total_rows, p=[0.7, 0.1, 0.1, 0.1])
    )

    payout_ratio_base = rng.uniform(0.2, 0.8, size=total_rows)
    payout_ratio_base -= (div_growth_streak_array / 100.0) * 0.1
    payout_ratio_array = np.clip(payout_ratio_base + rng.uniform(-0.15, 0.15, size=total_rows), 0.05, 0.95)

    eps_cagr_base = rng.uniform(0.00, 0.20, size=total_rows)
    eps_cagr_base += (div_growth_streak_array / 100.0) * 0.02
    eps_cagr_array = np.clip(eps_cagr_base + rng.uniform(-0.05, 0.05, size=total_rows), -0.10, 0.30)

    roe_base = rng.uniform(0.05, 0.30, size=total_rows)
    roe_base += (eps_cagr_array * 0.5)
    roe_array = np.clip(roe_base + rng.uniform(-0.07, 0.07, size=total_rows), 0.01, 0.50)

    debt_equity_array = rng.uniform(0.1, 2.5, size=total_rows)
    industry_debt_equity_array = debt_equity_array * rng.uniform(0.7, 1.3, size=total_rows) + \
                                 rng.uniform(-0.3, 0.3, size=total_rows)
    industry_debt_equity_array = np.maximum(0.05, industry_debt_equity_array)

    quality_proxy = (roe_array * 10) - (debt_equity_array * 2) + (div_growth_streak_array / 5)
    sp_quality_array = pd.cut(
        quality_proxy,
        bins=[-np.inf, 5, 10, 15, 20, 25, np.inf],
        labels=['B-', 'B', 'B+', 'A-', 'A', 'A+'],
        right=True, duplicates='drop'
    ).astype(str)
    sp_quality_array[pd.isna(sp_quality_array)] = rng.choice(
        ['B-', 'B', 'B+'], size=pd.isna(sp_quality_array).sum()
    )

    dividend_yield_base = rng.uniform(0.005, 0.06, size=total_rows)
    dividend_yield_base += (payout_ratio_array * 0.02)
    dividend_yield_base -= (eps_cagr_array * 0.1)
    dividend_yield_array = np.clip(dividend_yield_base + rng.uniform(-0.015, 0.015, size=total_rows), 0.001, 0.12)

    div_growth_5y_base = rng.uniform(0.00, 0.20, size=total_rows)
    div_growth_5y_base += (eps_cagr_array * 0.7)
    div_growth_5y_base += (div_growth_streak_array / 100.0) * 0.03
    div_growth_5y_array = np.clip(div_growth_5y_base + rng.uniform(-0.05, 0.05, size=total_rows), -0.05, 0.30)

    quality_score_array = np.maximum(0.0, (roe_array * 0.6) + (eps_cagr_array * 0.4))

    metrics_data = {
        'year': years_array, 'symbol': symbols_array,
        'div_growth_streak': div_growth_streak_array, 'payout_ratio': payout_ratio_array,
        'eps_cagr_3y': eps_cagr_array, 'roe': roe_array,
        'debt_equity': debt_equity_array, 'industry_debt_equity': industry_debt_equity_array,
        'sp_quality': sp_quality_array, 'dividend_yield': dividend_yield_array,
        'div_growth_5y': div_growth_5y_array, 'quality_score': quality_score_array,
        'share_price': share_prices_array, 'market_cap_rank': market_cap_ranks_array,
    }

    _craft_ideal_dgi_stocks(rng, metrics_data, num_companies_total)

    annual_returns_array = _calculate_annual_returns_from_prices(
        metrics_data['share_price'], num_companies_total, total_rows
    )
    metrics_data['annual_return'] = annual_returns_array

    data_df = pd.DataFrame(metrics_data)

    shares_outstanding_array = rng.integers(20_000_000, 1_500_000_000, size=total_rows)
    data_df['shares_outstanding'] = shares_outstanding_array
    data_df['market_cap'] = data_df['share_price'] * data_df['shares_outstanding']
    data_df = data_df.sort_values(by=['year', 'market_cap'], ascending=[True, False])
    data_df['market_cap_rank'] = data_df.groupby('year')['market_cap'].rank(
        method='min', ascending=False
    ).astype(int)
    return data_df


def _craft_ideal_value_stocks(
    rng: np.random.Generator,
    metrics_data: Dict[str, np.ndarray],
    num_companies_total: int
):
    """Modifies the metrics_data in-place to ensure some stocks meet ideal Value criteria.

    Args:
        rng: NumPy random number generator.
        metrics_data: Dictionary of NumPy arrays representing stock metrics.
                      This dictionary is modified in-place.
        num_companies_total: The total number of unique companies.
    """
    symbols_array = metrics_data['symbol']
    unique_symbols = np.unique(symbols_array[:num_companies_total])
    num_ideal_to_craft = min(NUM_IDEAL_VALUE_STOCKS_PER_YEAR, num_companies_total)
    if num_ideal_to_craft == 0:
        return
    ideal_symbols_chosen = rng.choice(unique_symbols, size=num_ideal_to_craft, replace=False)

    # Retrieve arrays from metrics_data dictionary.
    share_prices_array = metrics_data['share_price']
    market_cap_array = metrics_data['market_cap']
    pe_ratio_array = metrics_data['pe_ratio']
    sector_median_pe_array = metrics_data['sector_median_pe']
    pb_ratio_array = metrics_data['pb_ratio']
    fcf_yield_array = metrics_data['fcf_yield']
    sector_median_fcf_yield_array = metrics_data['sector_median_fcf_yield']
    tangible_book_value_per_share_array = metrics_data['tangible_book_value_per_share']
    current_ratio_array = metrics_data['current_ratio']
    debt_equity_array = metrics_data['debt_equity']
    industry_avg_debt_equity_array = metrics_data['industry_avg_debt_equity']
    roe_array = metrics_data['roe']
    positive_ni_5y_streak_array = metrics_data['positive_ni_5y_streak']
    total_book_value_array = metrics_data['total_book_value']
    total_debt_array = metrics_data['total_debt']
    net_current_asset_value_array = metrics_data['net_current_asset_value']
    margin_of_safety_array = metrics_data['margin_of_safety']
    sp_quality_array = metrics_data['sp_quality']
    eps_growth_5y_array = metrics_data['eps_growth_5y']
    significant_insider_activity_array = metrics_data['significant_insider_activity']
    composite_value_score_array = metrics_data['composite_value_score']
    quality_score_array = metrics_data['quality_score']

    for ideal_sym in ideal_symbols_chosen:
        indices_for_ideal_stock = np.where(symbols_array == ideal_sym)[0]
        num_rows_for_stock = len(indices_for_ideal_stock)
        if num_rows_for_stock == 0:
            continue

        # Undervaluation.
        ideal_pe = rng.uniform(5, 10, size=num_rows_for_stock)
        pe_ratio_array[indices_for_ideal_stock] = ideal_pe
        sector_median_pe_array[indices_for_ideal_stock] = ideal_pe * rng.uniform(2.6, 3.5, size=num_rows_for_stock)
        pb_ratio_array[indices_for_ideal_stock] = rng.uniform(0.4, 0.95, size=num_rows_for_stock)
        fcf_yield_array[indices_for_ideal_stock] = rng.uniform(0.08, 0.15, size=num_rows_for_stock)
        sector_median_fcf_yield_array[indices_for_ideal_stock] = fcf_yield_array[indices_for_ideal_stock] * \
                                                                 rng.uniform(0.5, 0.9, size=num_rows_for_stock)
        tangible_book_value_per_share_array[indices_for_ideal_stock] = share_prices_array[indices_for_ideal_stock] / \
                                                                        rng.uniform(0.4, 0.66, size=num_rows_for_stock)

        # Financial Health.
        current_ratio_array[indices_for_ideal_stock] = rng.uniform(1.6, 4.0, size=num_rows_for_stock)
        ideal_company_de = rng.uniform(0.1, 0.6, size=num_rows_for_stock)
        debt_equity_array[indices_for_ideal_stock] = ideal_company_de
        industry_avg_debt_equity_array[indices_for_ideal_stock] = ideal_company_de * \
                                                                  rng.uniform(1.1, 2.5, size=num_rows_for_stock)
        roe_array[indices_for_ideal_stock] = rng.uniform(0.155, 0.35, size=num_rows_for_stock)
        positive_ni_5y_streak_array[indices_for_ideal_stock] = rng.integers(5, 11, size=num_rows_for_stock)
        ideal_tbv = market_cap_array[indices_for_ideal_stock] * rng.uniform(0.5, 1.0, size=num_rows_for_stock)
        total_book_value_array[indices_for_ideal_stock] = ideal_tbv
        total_debt_array[indices_for_ideal_stock] = ideal_tbv * rng.uniform(0.1, 0.9, size=num_rows_for_stock)
        net_current_asset_value_array[indices_for_ideal_stock] = total_debt_array[indices_for_ideal_stock] / \
                                                                 rng.uniform(0.5, 1.9, size=num_rows_for_stock)

        # Margin of Safety.
        margin_of_safety_array[indices_for_ideal_stock] = rng.uniform(0.26, 0.65, size=num_rows_for_stock)

        # Quality Filters.
        sp_quality_array[indices_for_ideal_stock] = rng.choice(
            ['B+', 'A-', 'A', 'A+'], size=num_rows_for_stock, p=[0.25, 0.3, 0.25, 0.2]
        )
        eps_growth_5y_array[indices_for_ideal_stock] = rng.uniform(0.01, 0.20, size=num_rows_for_stock)
        significant_insider_activity_array[indices_for_ideal_stock] = rng.choice(
            [True, False], size=num_rows_for_stock, p=[0.5, 0.5]
        )

        # Recalculate scores for ideal stocks.
        ideal_pe_score = np.maximum(0, 1 - (pe_ratio_array[indices_for_ideal_stock] / 40))
        ideal_pb_score = np.maximum(0, 1 - (pb_ratio_array[indices_for_ideal_stock] / 2.5))
        composite_value_score_array[indices_for_ideal_stock] = \
            (margin_of_safety_array[indices_for_ideal_stock] * 0.5) + \
            (ideal_pe_score * 0.25) + (ideal_pb_score * 0.25) + \
            rng.uniform(0.1, 0.3, size=num_rows_for_stock)
        quality_score_array[indices_for_ideal_stock] = \
            (np.clip(roe_array[indices_for_ideal_stock], 0, 0.5) * 2 * 0.5) + \
            (np.clip(positive_ni_5y_streak_array[indices_for_ideal_stock], 0, 10) / 10 * 0.3) + \
            (significant_insider_activity_array[indices_for_ideal_stock].astype(int) * 0.2)
        quality_score_array[indices_for_ideal_stock] = np.clip(
            quality_score_array[indices_for_ideal_stock], 0.6, 1.0
        )


def _generate_value_play_data(
    rng: np.random.Generator, years_array: np.ndarray,
    symbols_array: np.ndarray, market_cap_ranks_array: np.ndarray,
    total_rows: int
) -> pd.DataFrame:
    """Generates synthetic data specific to the 'value_play' strategy.

    Ensures some stocks meet Value criteria.

    Args:
        rng: NumPy random number generator.
        years_array: NumPy array of years.
        symbols_array: NumPy array of stock symbols.
        market_cap_ranks_array: NumPy array of market cap ranks.
        total_rows: Total number of rows for data generation.

    Returns:
        A pandas DataFrame with data for the 'value_play' strategy.
    """
    num_companies_total = len(np.unique(
        symbols_array[:total_rows // (years_array[-1] - years_array[0] + 1)] if total_rows > 0 else []
    ))

    share_prices_array = _generate_share_prices_over_time(
        rng, symbols_array, num_companies_total, total_rows,
        base_price_low=10.0, base_price_high=300.0,
        annual_change_loc=0.07, annual_change_scale=0.20, min_price=0.50
    )

    sectors = ['Financials', 'Industrials', 'Consumer Staples', 'Technology',
               'Healthcare', 'Energy', 'Utilities', 'Consumer Discretionary',
               'Materials', 'Real Estate', 'Communication Services']
    sector_array = rng.choice(sectors, size=total_rows)

    pe_ratio_array = np.clip(rng.normal(loc=15, scale=8, size=total_rows), 3, 60)
    sector_median_pe_array = np.clip(rng.normal(loc=18, scale=5, size=total_rows), 8, 40)
    pb_ratio_array = np.clip(rng.normal(loc=1.5, scale=0.8, size=total_rows), 0.2, 7)
    sector_median_pb_array = np.clip(rng.normal(loc=1.8, scale=0.6, size=total_rows), 0.5, 5)
    fcf_yield_array = np.clip(rng.normal(loc=0.06, scale=0.04, size=total_rows), -0.05, 0.20)
    sector_median_fcf_yield_array = np.clip(rng.normal(loc=0.05, scale=0.02, size=total_rows), 0.00, 0.15)
    tangible_book_value_per_share_array = share_prices_array * rng.uniform(0.3, 2.0, size=total_rows)

    debt_equity_array = np.clip(rng.normal(loc=0.8, scale=0.6, size=total_rows), 0.05, 4.0)
    industry_avg_debt_equity_array = np.clip(rng.normal(loc=1.0, scale=0.5, size=total_rows), 0.1, 3.0)
    current_ratio_array = np.clip(rng.normal(loc=2.0, scale=1.2, size=total_rows), 0.3, 7.0)
    positive_ni_5y_streak_array = rng.integers(0, 11, size=total_rows)
    roe_array = np.clip(rng.normal(loc=0.12, scale=0.15, size=total_rows), -0.5, 0.6)

    shares_outstanding_array = rng.integers(10_000_000, 3_000_000_000, size=total_rows)
    market_cap_array = share_prices_array * shares_outstanding_array
    net_current_asset_value_array = market_cap_array * rng.uniform(0.05, 0.6, size=total_rows)
    total_book_value_array = market_cap_array * rng.uniform(0.1, 1.2, size=total_rows)
    total_book_value_array = np.maximum(0.01 * market_cap_array, total_book_value_array)
    total_debt_array = debt_equity_array * total_book_value_array
    total_debt_array = np.maximum(0, total_debt_array)

    margin_of_safety_array = np.clip(rng.normal(loc=0.10, scale=0.20, size=total_rows), -0.7, 0.8)

    sp_quality_labels = ['D', 'C', 'B-', 'B', 'B+', 'A-', 'A', 'A+']
    sp_quality_probabilities = [0.05, 0.10, 0.20, 0.20, 0.18, 0.12, 0.10, 0.05]
    sp_quality_array = rng.choice(sp_quality_labels, size=total_rows, p=sp_quality_probabilities)
    eps_growth_5y_array = np.clip(rng.normal(loc=0.03, scale=0.12, size=total_rows), -0.3, 0.4)
    significant_insider_activity_array = rng.choice([True, False], size=total_rows, p=[0.10, 0.90])

    pe_score_comp = np.maximum(0, 1 - (pe_ratio_array / 40))
    pb_score_comp = np.maximum(0, 1 - (pb_ratio_array / 2.5))
    composite_value_score_array = (margin_of_safety_array * 0.4) + \
                                  (pe_score_comp * 0.3) + \
                                  (pb_score_comp * 0.3) + \
                                  rng.normal(0, 0.05, size=total_rows)
    composite_value_score_array = np.clip(composite_value_score_array, -1, 1)

    quality_score_array = (np.clip(roe_array, 0, 0.5) * 2 * 0.5) + \
                          (np.clip(positive_ni_5y_streak_array, 0, 10) / 10 * 0.3) + \
                          (significant_insider_activity_array.astype(int) * 0.2)
    quality_score_array = np.clip(quality_score_array, 0, 1)

    metrics_data = {
        'year': years_array, 'symbol': symbols_array, 'share_price': share_prices_array,
        'sector': sector_array, 'pe_ratio': pe_ratio_array,
        'sector_median_pe': sector_median_pe_array, 'pb_ratio': pb_ratio_array,
        'sector_median_pb': sector_median_pb_array, 'fcf_yield': fcf_yield_array,
        'sector_median_fcf_yield': sector_median_fcf_yield_array,
        'tangible_book_value_per_share': tangible_book_value_per_share_array,
        'debt_equity': debt_equity_array, 'industry_avg_debt_equity': industry_avg_debt_equity_array,
        'current_ratio': current_ratio_array, 'positive_ni_5y_streak': positive_ni_5y_streak_array,
        'roe': roe_array, 'total_debt': total_debt_array,
        'net_current_asset_value': net_current_asset_value_array,
        'margin_of_safety': margin_of_safety_array, 'total_book_value': total_book_value_array,
        'sp_quality': sp_quality_array, 'eps_growth_5y': eps_growth_5y_array,
        'significant_insider_activity': significant_insider_activity_array,
        'composite_value_score': composite_value_score_array, 'quality_score': quality_score_array,
        'market_cap_rank': market_cap_ranks_array, 'shares_outstanding': shares_outstanding_array,
        'market_cap': market_cap_array
    }

    _craft_ideal_value_stocks(rng, metrics_data, num_companies_total)

    annual_returns_array = _calculate_annual_returns_from_prices(
        metrics_data['share_price'], num_companies_total, total_rows
    )
    metrics_data['annual_return'] = annual_returns_array

    data_df = pd.DataFrame(metrics_data)
    data_df = data_df.sort_values(by=['year', 'market_cap'], ascending=[True, False])
    data_df['market_cap_rank'] = data_df.groupby('year')['market_cap'].rank(
        method='min', ascending=False
    ).astype(int)

    return data_df


def generate_example_data(
        strategy_name: str,
        start_year: int = None,
        end_year: int = None,
        num_companies: int = DEFAULT_NUM_COMPANIES,
        seed: int = None
) -> pd.DataFrame:
    """Generates example market data for a specific strategy.

    Args:
        strategy_name: The name of the strategy ('exp_fund', 'dgi', or 'value_play').
        start_year: The first year for data generation.
            Defaults to `DEFAULT_START_YEAR_OFFSET` years before the `end_year`.
        end_year: The last year for data generation.
            Defaults to the year before the current year.
        num_companies: The number of unique companies to generate data for.
        seed: An optional seed for the random number generator for reproducibility.

    Returns:
        A pandas DataFrame containing the generated example data.

    Raises:
        ValueError: If `start_year` is greater than `end_year` or if
            `strategy_name` is not supported.
        NotImplementedError: If data generation for the strategy is not implemented.
    """
    rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()

    current_system_year = datetime.now().year
    effective_end_year = end_year if end_year is not None else current_system_year - 1
    effective_start_year = start_year if start_year is not None else effective_end_year - DEFAULT_START_YEAR_OFFSET

    if effective_start_year > effective_end_year:
        raise ValueError("start_year cannot be greater than end_year.")

    if strategy_name not in STRATEGY_COLUMNS:
        raise ValueError(f"Unsupported strategy '{strategy_name}'. "
                         f"Supported strategies: {list(STRATEGY_COLUMNS.keys())}")

    years_array, symbols_array, market_cap_ranks_array, total_rows = _generate_core_arrays(
        rng, effective_start_year, effective_end_year, num_companies
    )

    if strategy_name == 'exp_fund':
        data_df = _generate_exp_fund_data(
            rng, years_array, symbols_array, market_cap_ranks_array, total_rows
        )
    elif strategy_name == 'dgi':
        data_df = _generate_dgi_data(
            rng, years_array, symbols_array, market_cap_ranks_array, total_rows
        )
    elif strategy_name == 'value_play':
        data_df = _generate_value_play_data(
            rng, years_array, symbols_array, market_cap_ranks_array, total_rows
        )
    else:
        # This case should ideally not be reached due to the check above.
        raise NotImplementedError(f"Data generation not implemented for strategy: {strategy_name}")

    required_cols = STRATEGY_COLUMNS[strategy_name]['required']
    optional_cols = STRATEGY_COLUMNS[strategy_name]['optional']
    current_cols = list(data_df.columns)
    final_cols_ordered = []

    for col in required_cols:
        if col in current_cols:
            final_cols_ordered.append(col)
        else:
            data_df[col] = pd.NA
            final_cols_ordered.append(col)
            click.echo(
                f"Warning: Required column '{col}' for strategy '{strategy_name}' "
                "was missing and added as NA.",
                err=True
            )

    for col in optional_cols:
        if col in current_cols and col not in final_cols_ordered:
            final_cols_ordered.append(col)

    for col in current_cols:
        if col not in final_cols_ordered:
            final_cols_ordered.append(col)

    data_df = data_df[final_cols_ordered]
    data_df = data_df.sort_values(by=['year', 'symbol']).reset_index(drop=True)

    return data_df


@click.command()
@click.argument(
    'strategy_name',
    metavar='STRATEGY',
    type=click.Choice(list(STRATEGY_COLUMNS.keys()), case_sensitive=False),
)
@click.argument(
    'output_path',
    required=False,
    type=click.Path(dir_okay=False, writable=True)
)
@click.option(
    "--start-year", "-s",
    default=None,
    type=int,
    help=f"Start year for the data. Defaults to {DEFAULT_START_YEAR_OFFSET} years before end_year."
)
@click.option(
    "--end-year", "-e",
    default=None,
    type=int,
    help="End year for the data. Defaults to the year before the current year."
)
@click.option(
    "--num-companies", "-n",
    default=DEFAULT_NUM_COMPANIES,
    type=click.IntRange(min=1),
    show_default=True,
    help="Number of companies to generate."
)
@click.option(
    "--seed",
    type=int,
    default=None,
    help="Random seed for reproducibility."
)
def cli(
        strategy_name: str,
        output_path: str = None,
        start_year: int = None,
        end_year: int = None,
        num_companies: int = DEFAULT_NUM_COMPANIES,
        seed: int = None
):
    """Creates a CSV file with synthetic market data for backtesting a STRATEGY.

    This command generates financial market data tailored for the specified
    strategy and saves it to a CSV file.

    If OUTPUT_PATH is not provided, it defaults to
    './data/[strategy_name].example_data.[counter].csv'.

    Args:
        strategy_name: The name of the strategy for which to generate data.
        output_path: Optional path to save the generated CSV file.
        start_year: The first year for data generation.
        end_year: The last year for data generation.
        num_companies: The number of unique companies to generate data for.
        seed: An optional seed for the random number generator.
    """
    if output_path is None:
        default_dir = "data"
        if not os.path.exists(default_dir):
            try:
                os.makedirs(default_dir, exist_ok=True)
            except OSError as e:
                click.echo(f"Error creating default directory '{default_dir}': {e}", err=True)
                sys.exit(1)

        counter = 1
        while True:
            default_filename = f"{strategy_name}.example_data.{counter:02d}.csv"
            temp_output_path = os.path.join(default_dir, default_filename)
            if not os.path.exists(temp_output_path):
                output_path = temp_output_path
                break
            counter += 1

    click.echo(f"Generating synthetic market data for strategy '{strategy_name}'...")
    try:
        data_df = generate_example_data(
            strategy_name=strategy_name,
            start_year=start_year,
            end_year=end_year,
            num_companies=num_companies,
            seed=seed
        )
    except ValueError as e:
        click.echo(f"Error generating data: {e}", err=True)
        sys.exit(1)
    except NotImplementedError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    try:
        output_directory = os.path.dirname(os.path.abspath(output_path))
        if output_directory and not os.path.exists(output_directory):
            os.makedirs(output_directory, exist_ok=True)

        data_df.to_csv(output_path, index=False)
        click.echo(f"Success: Example data saved to {output_path}")
    except IOError as e:
        click.echo(f"Error: Failed to save data to {output_path}. {e}", err=True)
        sys.exit(1)
    except Exception as e:  # pylint: disable=broad-except
        click.echo(f"An unexpected error occurred while saving data: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    cli()  # pylint: disable=no-value-for-parameter