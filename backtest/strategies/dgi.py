"""Implementation of the Dividend Growth Investing (DGI) strategy.

This strategy focuses on companies with a strong history of dividend growth,
financial stability, and aims for long-term compounding of income and capital.
"""
from typing import Any, Dict

import pandas as pd

from .base import Strategy
from backtest.utils.quality_utils import (
    map_sp_quality_to_numeric,
    MIN_SP_QUALITY_NUMERIC
)


class DgiStrategy(Strategy):
    """Implements the Dividend Growth Investing (DGI) strategy.

    Selects stocks based on dividend growth history, payout ratios, earnings
    growth, ROE, debt levels, and S&P Quality Rank. Portfolio weights are
    determined by a hybrid approach combining dividend yield, dividend growth
    rate, and a quality score.
    """

    def execute_strategy(
            self, data_for_period: pd.DataFrame, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Executes the DGI strategy for a single period.

        Args:
            data_for_period: DataFrame containing market data for the period.
                             Expected columns include 'div_growth_streak',
                             'payout_ratio', 'eps_cagr_3y', 'roe', 'debt_equity',
                             'industry_debt_equity', 'sp_quality', 'dividend_yield',
                             'div_growth_5y', and 'quality_score'.
            params: A dictionary of strategy parameters (currently unused by DGI).

        Returns:
            A dictionary with 'portfolio' (DataFrame of selected stocks with
            weights and 'share_price') and 'metrics' (DGI-specific metrics for
            the period).
        """
        candidates_df = data_for_period.copy()

        # Use the imported utility function and constant
        if 'sp_quality' in candidates_df.columns:
            candidates_df['sp_quality_numeric'] = map_sp_quality_to_numeric(
                candidates_df['sp_quality']
            )
        else:
            print("Warning: 'sp_quality' column not found for DgiStrategy. "
                  "Skipping S&P Quality Rank filter.")
            # Assign a value that will fail the quality check if column is missing
            candidates_df['sp_quality_numeric'] = MIN_SP_QUALITY_NUMERIC - 1 # Or np.nan, but a low number is safer for comparison

        # Ensure sp_quality_numeric exists before using it in the condition
        if 'sp_quality_numeric' not in candidates_df.columns:
             candidates_df['sp_quality_numeric'] = MIN_SP_QUALITY_NUMERIC - 1 # Ensure it exists and fails filter

        screened_df = candidates_df[
            (candidates_df.get('div_growth_streak', pd.Series(dtype=float)).fillna(0) >= 10) & # Added fillna for robustness
            (candidates_df.get('payout_ratio', pd.Series(dtype=float)).fillna(1.0) <= 0.60) & # Added fillna
            (candidates_df.get('eps_cagr_3y', pd.Series(dtype=float)).fillna(float('-inf')) >= 0.05) & # Added fillna
            (candidates_df.get('roe', pd.Series(dtype=float)).fillna(float('-inf')) >= 0.15) & # Added fillna
            (candidates_df.get('debt_equity', pd.Series(dtype=float)).fillna(float('inf')) <=
             candidates_df.get('industry_debt_equity', pd.Series(dtype=float)).fillna(float('inf'))) & # Added fillna
            (candidates_df['sp_quality_numeric'] >= MIN_SP_QUALITY_NUMERIC) # Use imported constant
        ].copy()

        if screened_df.empty:
            return {
                'portfolio': pd.DataFrame(columns=['symbol', 'weight', 'share_price']),
                'metrics': DgiStrategy.calculate_metrics(pd.DataFrame())
            }

        required_weighting_cols = ['dividend_yield', 'div_growth_5y', 'quality_score']
        # Check if required weighting columns are present AND have non-NaN values for calculation
        missing_weighting_cols = [
            col for col in required_weighting_cols
            if col not in screened_df.columns or screened_df[col].isnull().all()
        ]


        if missing_weighting_cols:
            # Fallback to equal weighting if weighting columns are missing or all NaN
            print(f"Warning: Missing or all-NaN columns for DGI weighting: {missing_weighting_cols}. "
                  "Falling back to equal weight for selected stocks.")
            screened_df['weight'] = 1.0 / len(screened_df) if len(screened_df) > 0 else 0.0
        else:
             # Use fillna(0) for weighting calculation to treat missing/NaN values as 0 contribution
            screened_df['raw_weight_score'] = (
                    0.50 * screened_df['dividend_yield'].fillna(0) +
                    0.30 * screened_df['div_growth_5y'].fillna(0) +
                    0.20 * screened_df['quality_score'].fillna(0)
            )

            if screened_df['raw_weight_score'].sum() <= 0:
                print("Warning: Raw weight scores are not positive. Falling back to equal weight for selected DGI stocks.")
                screened_df['weight'] = 1.0 / len(screened_df) if len(screened_df) > 0 else 0.0
            else:
                # Ensure no negative weights before normalizing
                screened_df.loc[screened_df['raw_weight_score'] < 0, 'raw_weight_score'] = 0
                screened_df['weight'] = screened_df['raw_weight_score'] / screened_df['raw_weight_score'].sum()


        output_columns = ['symbol', 'weight']
        if 'share_price' in screened_df.columns:
            output_columns.append('share_price')
        else:
            # Add a placeholder if share_price is missing, though it's required by constants
            screened_df['share_price'] = 1.0
            output_columns.append('share_price')

        # Add other relevant columns for metrics/reporting if they exist
        for col in ['year', 'annual_return', 'market_cap', 'dividend_yield', 'div_growth_5y']: # Added yield and growth for metrics
            if col in screened_df.columns and col not in output_columns:
                output_columns.append(col)

        final_portfolio_df = screened_df[output_columns].copy()
        # Drop rows where weight calculation resulted in NaN (shouldn't happen with fillna, but as a safeguard)
        final_portfolio_df.dropna(subset=['weight'], inplace=True)

        # Re-normalize weights after dropping NaNs, if any were dropped
        if not final_portfolio_df.empty and final_portfolio_df['weight'].sum() > 0:
             # Check if sum is close to 1.0, normalize if not
            if abs(final_portfolio_df['weight'].sum() - 1.0) > 1e-6:
                 final_portfolio_df['weight'] /= final_portfolio_df['weight'].sum()
        elif not final_portfolio_df.empty:
             # Fallback to equal weight if sum is zero after dropping NaNs
             final_portfolio_df['weight'] = 1.0 / len(final_portfolio_df)


        return {
            'portfolio': final_portfolio_df,
            'metrics': DgiStrategy.calculate_metrics(final_portfolio_df)
        }

    @staticmethod
    def calculate_metrics(portfolio_for_period: pd.DataFrame) -> Dict[str, float]:
        """Calculates DGI-specific metrics for the portfolio in a given period.

        Args:
            portfolio_for_period: DataFrame representing the selected portfolio
                                  for the period, including 'weight',
                                  'dividend_yield', and 'div_growth_5y'.

        Returns:
            A dictionary of DGI-specific metrics like portfolio size,
            average dividend yield, and average 5-year dividend growth.
        """
        avg_yield = 0.0
        avg_div_growth = 0.0
        # Ensure weight column exists and is not all NaN/zero before calculating weighted sums
        if not portfolio_for_period.empty and 'weight' in portfolio_for_period.columns and portfolio_for_period['weight'].sum() > 0:
            if 'dividend_yield' in portfolio_for_period.columns:
                # Use fillna(0) for the metric itself in case of NaNs in the data
                avg_yield = (portfolio_for_period['dividend_yield'].fillna(0) * portfolio_for_period['weight']).sum()
            if 'div_growth_5y' in portfolio_for_period.columns:
                 # Use fillna(0) for the metric itself
                avg_div_growth = (portfolio_for_period['div_growth_5y'].fillna(0) * portfolio_for_period['weight']).sum()

        return {
            'dgi_portfolio_size': len(portfolio_for_period),
            'dgi_avg_dividend_yield': avg_yield,
            'dgi_avg_div_growth_5y': avg_div_growth,
        }