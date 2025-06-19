"""Implementation of the Value Play Investing strategy.

This strategy focuses on identifying undervalued companies with strong
fundamentals, a margin of safety, and long-term compounding potential.
"""
from typing import Any, Dict

import numpy as np
import pandas as pd

from .base import Strategy
# Import shared quality utilities
from backtest.utils.quality_utils import (
    map_sp_quality_to_numeric,
    MIN_SP_QUALITY_NUMERIC
)


class ValuePlayStrategy(Strategy):
    """Implements the Value Investing strategy.

    Selects stocks based on a combination of undervaluation metrics (P/E, P/B),
    financial health (current ratio, debt-to-equity, ROE), and quality filters
    (S&P Quality Rank, EPS growth, margin of safety). Portfolio weights are
    determined by a hybrid approach.

    Attributes:
        DEFAULT_TOP_N_VALUE: Default number of stocks if 'top_n' not in params.
    """

    DEFAULT_TOP_N_VALUE = 20

    def execute_strategy(
            self, data_for_period: pd.DataFrame, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Executes the Value Play strategy for a single period.

        Args:
            data_for_period: DataFrame with market data for the period.
            params: Dictionary of strategy parameters, e.g., 'top_n'.

        Returns:
            A dictionary with 'portfolio' (DataFrame of selected stocks with
            weights and 'share_price') and 'metrics' (strategy-specific
            metrics for the period).
        """
        candidates_df = data_for_period.copy()
        top_n = params.get('top_n', self.DEFAULT_TOP_N_VALUE)

        # Use the imported utility function and constant
        if 'sp_quality' in candidates_df.columns:
            candidates_df['sp_quality_numeric'] = map_sp_quality_to_numeric(
                candidates_df['sp_quality']
            )
        else:
            print("Warning: 'sp_quality' column not found for ValuePlayStrategy. "
                  "Skipping S&P Quality Rank filter.")
            # Assign a value that will fail the quality check if column is missing
            candidates_df['sp_quality_numeric'] = MIN_SP_QUALITY_NUMERIC - 1 # Or np.nan, but a low number is safer for comparison

        # Ensure sp_quality_numeric exists before using it in the condition
        if 'sp_quality_numeric' not in candidates_df.columns:
             candidates_df['sp_quality_numeric'] = MIN_SP_QUALITY_NUMERIC - 1 # Ensure it exists and fails filter


        # Added fillna for robustness in screening conditions
        pe_condition = (
                candidates_df.get('pe_ratio', pd.Series(dtype=float)).fillna(float('inf')) < # Fill NaN PE with inf to fail < check
                candidates_df.get('sector_median_pe', pd.Series(dtype=float)).fillna(0) * 0.4 # Fill NaN median PE with 0
        )
        pb_condition = (candidates_df.get('pb_ratio', pd.Series(dtype=float)).fillna(float('inf')) < 1.0) # Fill NaN PB with inf
        current_ratio_condition = (candidates_df.get('current_ratio', pd.Series(dtype=float)).fillna(0) > 1.5) # Fill NaN CR with 0
        debt_equity_condition = (
                candidates_df.get('debt_equity', pd.Series(dtype=float)).fillna(float('inf')) < # Fill NaN DE with inf
                candidates_df.get('industry_avg_debt_equity', pd.Series(dtype=float)).fillna(float('inf')) # Fill NaN industry avg DE with inf
        )
        roe_condition = (candidates_df.get('roe', pd.Series(dtype=float)).fillna(float('-inf')) > 0.15) # Fill NaN ROE with neg inf
        eps_growth_condition = (candidates_df.get('eps_growth_5y', pd.Series(dtype=float)).fillna(float('-inf')) > 0) # Fill NaN EPS growth with neg inf
        # Use imported constant
        sp_quality_condition = (
            candidates_df['sp_quality_numeric'] >= MIN_SP_QUALITY_NUMERIC
        )
        margin_of_safety_condition = (candidates_df.get('margin_of_safety', pd.Series(dtype=float)).fillna(float('-inf')) > 0.25) # Fill NaN MOS with neg inf


        screened_df = candidates_df[
            pe_condition & pb_condition & current_ratio_condition &
            debt_equity_condition & roe_condition & eps_growth_condition &
            sp_quality_condition & margin_of_safety_condition
        ].copy()

        if screened_df.empty:
            return {
                'portfolio': pd.DataFrame(columns=['symbol', 'weight', 'share_price']),
                'metrics': ValuePlayStrategy.calculate_metrics(pd.DataFrame())
            }

        # Sorting logic remains the same
        if 'composite_value_score' in screened_df.columns and not screened_df['composite_value_score'].isnull().all():
            screened_df = screened_df.sort_values('composite_value_score', ascending=False)
        elif 'margin_of_safety' in screened_df.columns and not screened_df['margin_of_safety'].isnull().all():
            print("Warning: 'composite_value_score' not found or all NaN. Sorting by 'margin_of_safety'.")
            screened_df = screened_df.sort_values('margin_of_safety', ascending=False)
        else:
            print("Warning: Neither 'composite_value_score' nor 'margin_of_safety' found or all NaN. "
                  "Cannot sort for ValuePlayStrategy. Selecting top N arbitrarily.")
            # If sorting isn't possible, just take the head(top_n) as is.

        selected_stocks_df = screened_df.head(top_n).copy()

        if selected_stocks_df.empty:
            return {
                'portfolio': pd.DataFrame(columns=['symbol', 'weight', 'share_price']),
                'metrics': ValuePlayStrategy.calculate_metrics(pd.DataFrame())
            }

        # Weighting logic remains the same, added fillna(0) for robustness in sums
        num_selected = len(selected_stocks_df)
        equal_weight_component = 0.5 * (1 / num_selected if num_selected > 0 else 0)

        mos_weight_component = 0.0
        if 'margin_of_safety' in selected_stocks_df.columns:
            mos_sum = selected_stocks_df['margin_of_safety'].fillna(0).sum() # Fill NaN MOS with 0 for sum
            if mos_sum > 0:
                mos_normalized = selected_stocks_df['margin_of_safety'].fillna(0) / mos_sum # Fill NaN MOS with 0 for normalization
                mos_weight_component = 0.25 * mos_normalized
            else:
                 print("Warning: Sum of 'margin_of_safety' is not positive. Using 0 for this component.")
        else:
            print("Warning: 'margin_of_safety' column not found for weighting. Using 0 for this component.")


        quality_weight_component = 0.0
        if 'quality_score' in selected_stocks_df.columns:
            quality_sum = selected_stocks_df['quality_score'].fillna(0).sum() # Fill NaN quality score with 0 for sum
            if quality_sum > 0:
                quality_normalized = selected_stocks_df['quality_score'].fillna(0) / quality_sum # Fill NaN quality score with 0 for normalization
                quality_weight_component = 0.25 * quality_normalized
            else:
                 print("Warning: Sum of 'quality_score' is not positive. Using 0 for this component.")
        else:
            print("Warning: 'quality_score' column not found for weighting. Using 0 for this component.")


        selected_stocks_df['weight'] = equal_weight_component + mos_weight_component + quality_weight_component

        # Normalize weights to sum to 1.0
        if 'weight' in selected_stocks_df.columns and selected_stocks_df['weight'].sum() > 0:
             # Check if sum is close to 1.0, normalize if not
            if abs(selected_stocks_df['weight'].sum() - 1.0) > 1e-6:
                 selected_stocks_df['weight'] /= selected_stocks_df['weight'].sum()
        elif not selected_stocks_df.empty:
             # Fallback to equal weight if sum is zero
             selected_stocks_df['weight'] = 1.0 / len(selected_stocks_df)
        #else:
             # If empty, weight is already 0.0

        output_columns = ['symbol', 'weight']
        if 'share_price' in selected_stocks_df.columns:
            output_columns.append('share_price')
        else:
            # Add a placeholder if share_price is missing, though it's required by constants
            selected_stocks_df['share_price'] = 1.0
            output_columns.append('share_price')

        # Add other relevant columns for metrics/reporting if they exist
        for col in ['year', 'annual_return', 'market_cap', 'pe_ratio', 'pb_ratio', 'roe', 'margin_of_safety']:
            if col in selected_stocks_df.columns and col not in output_columns:
                output_columns.append(col)

        final_portfolio_df = selected_stocks_df[output_columns].copy()
        # Drop rows where weight calculation resulted in NaN (shouldn't happen with fillna, but as a safeguard)
        final_portfolio_df.dropna(subset=['weight'], inplace=True)

        # Re-normalize weights after dropping NaNs, if any were dropped
        if not final_portfolio_df.empty and 'weight' in final_portfolio_df.columns and final_portfolio_df['weight'].sum() > 0:
             # Check if sum is close to 1.0, normalize if not
            if abs(final_portfolio_df['weight'].sum() - 1.0) > 1e-6:
                 final_portfolio_df['weight'] /= final_portfolio_df['weight'].sum()
        elif not final_portfolio_df.empty:
             # Fallback to equal weight if sum is zero after dropping NaNs
             final_portfolio_df['weight'] = 1.0 / len(final_portfolio_df)


        return {
            'portfolio': final_portfolio_df,
            'metrics': ValuePlayStrategy.calculate_metrics(final_portfolio_df)
        }

    @staticmethod
    def calculate_metrics(portfolio_for_period: pd.DataFrame) -> Dict[str, float]:
        """Calculates ValuePlay-specific metrics for the portfolio in a given period.

        Args:
            portfolio_for_period: DataFrame representing the selected portfolio.

        Returns:
            A dictionary of metrics like portfolio size, average margin of safety,
            and average ROE.
        """
        avg_margin_of_safety = 0.0
        avg_roe = 0.0
         # Ensure weight column exists and is not all NaN/zero before calculating weighted sums
        if not portfolio_for_period.empty and 'weight' in portfolio_for_period.columns and portfolio_for_period['weight'].sum() > 0:
            # Ensure the metric columns exist before trying to access them
            weighted_sum_exists = portfolio_for_period['weight'].sum() > 0
            if 'margin_of_safety' in portfolio_for_period.columns and weighted_sum_exists:
                # Use fillna(0) for the metric itself in case of NaNs in the data
                avg_margin_of_safety = (
                        portfolio_for_period['margin_of_safety'].fillna(0) * portfolio_for_period['weight']
                ).sum()
            if 'roe' in portfolio_for_period.columns and weighted_sum_exists:
                 # Use fillna(0) for the metric itself
                avg_roe = (portfolio_for_period['roe'].fillna(0) * portfolio_for_period['weight']).sum()
        return {
            'value_portfolio_size': len(portfolio_for_period),
            'value_avg_margin_of_safety': avg_margin_of_safety,
            'value_avg_roe': avg_roe,
        }