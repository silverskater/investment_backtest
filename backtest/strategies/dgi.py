"""Implementation of the Dividend Growth Investing (DGI) strategy."""

from typing import Any, Dict

import numpy as np
import pandas as pd

from .base import Strategy


class DgiStrategy(Strategy):
    """
    Dividend Growth Investing (DGI) strategy.

    This strategy focuses on companies with a strong history of dividend growth,
    financial stability, and aims for long-term compounding of income and
    capital. The selection and weighting criteria are based on the principles
    outlined in the dgi.md documentation.
    """

    # Define a mapping for S&P Quality Ranks to numerical values for comparison
    # Higher number is better.
    SP_QUALITY_RANK_MAPPING = {
        'A+': 6, 'A': 5, 'A-': 4,
        'B+': 3, 'B': 2, 'B-': 1,
        # Add lower ranks if necessary, or handle missing/unmapped values
    }
    MIN_SP_QUALITY_NUMERIC = SP_QUALITY_RANK_MAPPING['B+']

    def _map_sp_quality_to_numeric(self, quality_series: pd.Series) -> pd.Series:
        """Maps S&P quality string ranks to numerical values."""
        return quality_series.map(self.SP_QUALITY_RANK_MAPPING).fillna(0)  # Default to 0 if unmapped

    def execute_strategy(
            self, data_for_period: pd.DataFrame, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Executes the DGI strategy for a single period.

        Args:
            data_for_period: DataFrame containing market data for the specific
                             period. Expected columns include:
                             'symbol', 'share_price', 'div_growth_streak',
                             'payout_ratio', 'eps_cagr_3y', 'roe',
                             'debt_equity', 'industry_debt_equity',
                             'sp_quality', 'dividend_yield', 'div_growth_5y',
                             'quality_score' (composite).
            params: A dictionary of strategy parameters from the CLI.
                    This DGI strategy primarily uses its own fixed criteria
                    but could be extended to use some generic params like 'top_n'.

        Returns:
            A dictionary containing the resulting 'portfolio' (DataFrame of
            selected stocks with weights) and 'metrics' (empty for now).
        """
        # Make a copy to avoid modifying the original DataFrame
        candidates_df = data_for_period.copy()

        # --- 1. Stock Selection Criteria ---
        # Apply S&P Quality Rank mapping
        if 'sp_quality' in candidates_df.columns:
            candidates_df['sp_quality_numeric'] = self._map_sp_quality_to_numeric(
                candidates_df['sp_quality']
            )
        else:
            # If sp_quality is missing, we can't apply this filter.
            # Option 1: Exclude all stocks (conservative)
            # Option 2: Skip this filter with a warning (as done here)
            # Option 3: Require the column and raise an error
            print("Warning: 'sp_quality' column not found. Skipping S&P Quality Rank filter.")
            candidates_df['sp_quality_numeric'] = np.nan  # Will not pass the filter if it's NaN

        # Ensure required columns exist, using .get() for safety
        screened_df = candidates_df[
            (candidates_df.get('div_growth_streak', pd.Series(dtype=float)) >= 10) &
            (candidates_df.get('payout_ratio', pd.Series(dtype=float)) <= 0.60) &
            (candidates_df.get('eps_cagr_3y', pd.Series(dtype=float)) >= 0.05) &
            (candidates_df.get('roe', pd.Series(dtype=float)) >= 0.15) &
            (candidates_df.get('debt_equity', pd.Series(dtype=float)) <= candidates_df.get('industry_debt_equity', pd.Series(dtype=float))) &
            (candidates_df.get('sp_quality_numeric', pd.Series(dtype=float)) >= self.MIN_SP_QUALITY_NUMERIC)
        ].copy()  # Use .copy() to ensure screened_df is a new DataFrame

        if screened_df.empty:
            return {
                'portfolio': pd.DataFrame(columns=['symbol', 'weight', 'share_price']),  # Ensure schema
                'metrics': self.calculate_metrics(pd.DataFrame())
            }

        # --- 2. Portfolio Construction - Weighting ---
        # Apply hybrid approach:
        # - 50% dividend yield weighting
        # - 30% dividend growth rate (5-year CAGR) -> 'div_growth_5y'
        # - 20% quality score (ROE + earnings stability) -> assume 'quality_score' column

        # Normalize components for weighting to avoid scale issues if they are not already 0-1
        # For simplicity, we assume these metrics are already in a comparable scale or
        # that their raw values are appropriate for direct weighting.
        # A more robust approach might involve ranking or scaling these components.

        # Ensure all necessary columns for weighting exist
        required_weighting_cols = ['dividend_yield', 'div_growth_5y', 'quality_score']
        missing_weighting_cols = [col for col in required_weighting_cols if col not in screened_df.columns]

        if missing_weighting_cols:
            print(f"Warning: Missing columns for DGI weighting: {missing_weighting_cols}. "
                  "Cannot calculate weights. Returning empty portfolio.")
            return {
                'portfolio': pd.DataFrame(columns=['symbol', 'weight', 'share_price']),
                'metrics': self.calculate_metrics(pd.DataFrame())
            }

        # Calculate raw weighted score
        # Fill NaN with 0 for weighting calculation to avoid issues,
        # or handle missing data more sophisticatedly (e.g., imputation, exclusion)
        screened_df['raw_weight_score'] = (
                0.50 * screened_df['dividend_yield'].fillna(0) +
                0.30 * screened_df['div_growth_5y'].fillna(0) +
                0.20 * screened_df['quality_score'].fillna(0)
        )

        # Handle cases where all raw_weight_scores are zero or negative
        if screened_df['raw_weight_score'].sum() <= 0:
            # Fallback to equal weighting if scores are not positive
            print("Warning: Raw weight scores are not positive. Falling back to equal weight for selected DGI stocks.")
            screened_df['weight'] = 1.0 / len(screened_df) if len(screened_df) > 0 else 0.0
        else:
            # Set negative scores to 0 before normalization to avoid negative weights
            screened_df.loc[screened_df['raw_weight_score'] < 0, 'raw_weight_score'] = 0
            screened_df['weight'] = screened_df['raw_weight_score'] / screened_df['raw_weight_score'].sum()

        # Select necessary columns for the output portfolio
        # Ensure 'share_price' is included as cli.py might use it.
        # 'symbol' and 'weight' are essential.
        output_columns = ['symbol', 'weight']
        if 'share_price' in screened_df.columns:
            output_columns.append('share_price')
        else:
            # If share_price is somehow missing at this stage, add a placeholder
            screened_df['share_price'] = 1.0  # Not ideal, but for schema consistency
            output_columns.append('share_price')

        # Carry over other potentially useful columns if they exist
        for col in ['year', 'annual_return', 'market_cap']:
            if col in screened_df.columns and col not in output_columns:
                output_columns.append(col)

        final_portfolio_df = screened_df[output_columns].copy()

        # Drop rows where weight might have become NaN due to all-zero sums or other issues
        final_portfolio_df.dropna(subset=['weight'], inplace=True)
        # Ensure weights sum to 1 after any drops, though the above logic should prevent this if sum > 0
        if not final_portfolio_df.empty and final_portfolio_df['weight'].sum() > 0:
            if abs(final_portfolio_df['weight'].sum() - 1.0) > 1e-6:  # Check if not already normalized
                final_portfolio_df['weight'] = final_portfolio_df['weight'] / final_portfolio_df['weight'].sum()
        elif not final_portfolio_df.empty:  # All weights are zero or NaN
            final_portfolio_df['weight'] = 1.0 / len(final_portfolio_df)

        return {
            'portfolio': final_portfolio_df,
            'metrics': self.calculate_metrics(final_portfolio_df)  # Placeholder
        }

    def calculate_metrics(self, portfolio_for_period: pd.DataFrame) -> Dict[str, float]:
        """
        Calculates (placeholder) performance metrics for the selected portfolio.
        This can be expanded to include DGI-specific metrics for the period.
        """
        # Example DGI-specific metrics for the period (optional)
        avg_yield = 0.0
        avg_div_growth = 0.0
        if not portfolio_for_period.empty and 'weight' in portfolio_for_period.columns:
            if 'dividend_yield' in portfolio_for_period.columns:
                avg_yield = (portfolio_for_period['dividend_yield'] * portfolio_for_period['weight']).sum()
            if 'div_growth_5y' in portfolio_for_period.columns:
                avg_div_growth = (portfolio_for_period['div_growth_5y'] * portfolio_for_period['weight']).sum()

        return {
            'dgi_portfolio_size': len(portfolio_for_period),
            'dgi_avg_dividend_yield': avg_yield,
            'dgi_avg_div_growth_5y': avg_div_growth,
        }
