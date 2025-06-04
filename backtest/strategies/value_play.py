"""Implementation of the Value Play Investing strategy."""

from typing import Any, Dict

import numpy as np
import pandas as pd

from .base import Strategy


class ValuePlayStrategy(Strategy):
    """
    Value Investing strategy.

    This strategy focuses on identifying undervalued companies with strong
    fundamentals, a margin of safety, and long-term compounding potential,
    based on the principles outlined in the value_play.md documentation.
    """

    # Define a mapping for S&P Quality Ranks to numerical values for comparison
    # Higher number is better.
    SP_QUALITY_RANK_MAPPING = {
        'A+': 6, 'A': 5, 'A-': 4,
        'B+': 3, 'B': 2, 'B-': 1,
    }
    MIN_SP_QUALITY_NUMERIC = SP_QUALITY_RANK_MAPPING['B+']  # B+ or better

    # Default top_n if not provided in params, aligning with portfolio size 10-30
    DEFAULT_TOP_N_VALUE = 20

    def _map_sp_quality_to_numeric(self, quality_series: pd.Series) -> pd.Series:
        """Maps S&P quality string ranks to numerical values."""
        return quality_series.map(self.SP_QUALITY_RANK_MAPPING).fillna(0)  # Default to 0 if unmapped

    def execute_strategy(
            self, data_for_period: pd.DataFrame, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Executes the Value Play strategy for a single period.

        Args:
            data_for_period: DataFrame containing market data for the specific
                             period. Expected columns are defined by the
                             value_play.md screening criteria.
            params: A dictionary of strategy parameters from the CLI.
                    'top_n' can be used to limit portfolio size.

        Returns:
            A dictionary containing the resulting 'portfolio' (DataFrame of
            selected stocks with weights) and 'metrics'.
        """
        candidates_df = data_for_period.copy()
        top_n = params.get('top_n', self.DEFAULT_TOP_N_VALUE)

        # --- 1. Stock Selection Criteria (adapted from value_screener in md) ---
        # Ensure necessary columns exist for screening, using .get() for safety
        # and providing default Series that will make conditions false if column is missing.

        # S&P Quality Rank
        if 'sp_quality' in candidates_df.columns:
            candidates_df['sp_quality_numeric'] = self._map_sp_quality_to_numeric(
                candidates_df['sp_quality']
            )
        else:
            print("Warning: 'sp_quality' column not found for ValuePlayStrategy. Skipping S&P Quality Rank filter.")
            candidates_df['sp_quality_numeric'] = np.nan  # Will fail the filter

        # Undervaluation: P/E
        # P/E ratio below sector median OR less than 40% of the stock's highest P/E over the previous five years
        # Simplified to: P/E ratio below sector median (as per screener example)
        # The screener example uses pe_ratio < sector_median_pe * 0.4, which is more stringent.
        pe_condition = (
                candidates_df.get('pe_ratio', pd.Series(dtype=float)) <
                candidates_df.get('sector_median_pe', pd.Series(dtype=float)) * 0.4
        )
        # Undervaluation: P/B
        # P/B ratio below 1.0 OR below sector median
        # Screener example uses pb_ratio < 1.0
        pb_condition = (candidates_df.get('pb_ratio', pd.Series(dtype=float)) < 1.0)

        # Financial Health: Current Ratio
        current_ratio_condition = (candidates_df.get('current_ratio', pd.Series(dtype=float)) > 1.5)

        # Financial Health: Debt-to-Equity
        # Debt-to-equity ratio below industry average
        # (The OR total debt < twice net current asset value is more complex for this step)
        debt_equity_condition = (
                candidates_df.get('debt_equity', pd.Series(dtype=float)) <
                candidates_df.get('industry_avg_debt_equity', pd.Series(dtype=float))
        )
        # Financial Health: ROE
        roe_condition = (candidates_df.get('roe', pd.Series(dtype=float)) > 0.15)

        # Quality: Earnings Growth
        # Consistent earnings growth with positive earnings per share growth over five years
        # Renamed earnings_growth_5y to eps_growth_5y to match typical data column names
        eps_growth_condition = (candidates_df.get('eps_growth_5y', pd.Series(dtype=float)) > 0)

        # Quality: S&P Rank
        sp_quality_condition = (
                    candidates_df.get('sp_quality_numeric', pd.Series(dtype=float)) >= self.MIN_SP_QUALITY_NUMERIC)

        # Margin of Safety
        # Estimated intrinsic value at least 20-30% above current market price. Screener uses > 0.25
        margin_of_safety_condition = (candidates_df.get('margin_of_safety', pd.Series(dtype=float)) > 0.25)

        # Additional criteria from markdown not in screener example (can be added if data exists):
        # - Free cash flow yield > sector median FCF yield
        # - Price < 67% of tangible per-share book value
        # - Positive net income for at least 5 consecutive years (positive_ni_5y_streak > 4)
        # - Company's total book value > total debt
        # - Management with significant insider ownership (significant_insider_activity == True)

        screened_df = candidates_df[
            pe_condition &
            pb_condition &
            current_ratio_condition &
            debt_equity_condition &
            roe_condition &
            eps_growth_condition &
            sp_quality_condition &
            margin_of_safety_condition
            ].copy()  # Use .copy() to avoid SettingWithCopyWarning

        if screened_df.empty:
            return {
                'portfolio': pd.DataFrame(columns=['symbol', 'weight', 'share_price']),  # Ensure schema
                'metrics': self.calculate_metrics(pd.DataFrame())
            }

        # Sort by composite_value_score (assuming this column is generated in example_data)
        # If not, we might need a proxy or fallback to equal weighting of top N by another metric.
        if 'composite_value_score' in screened_df.columns:
            screened_df = screened_df.sort_values('composite_value_score', ascending=False)
        else:
            print("Warning: 'composite_value_score' not found. Cannot sort for ValuePlayStrategy.")
            # Fallback: if no composite score, perhaps sort by margin_of_safety or another key metric
            if 'margin_of_safety' in screened_df.columns:
                screened_df = screened_df.sort_values('margin_of_safety', ascending=False)

        # Portfolio size: Optimal range of 10-30 stocks
        selected_stocks_df = screened_df.head(top_n).copy()

        if selected_stocks_df.empty:
            return {
                'portfolio': pd.DataFrame(columns=['symbol', 'weight', 'share_price']),
                'metrics': self.calculate_metrics(pd.DataFrame())
            }

        # --- 2. Portfolio Construction - Weighting ---
        # 50% equal weight allocation
        # 25% based on margin of safety (greater discount to intrinsic value)
        # 25% based on quality score combining ROE, earnings stability, and insider ownership
        num_selected = len(selected_stocks_df)
        equal_weight_component = 0.5 * (1 / num_selected if num_selected > 0 else 0)

        # Margin of Safety component (needs normalization)
        if 'margin_of_safety' in selected_stocks_df.columns and selected_stocks_df['margin_of_safety'].sum() > 0:
            mos_normalized = selected_stocks_df['margin_of_safety'] / selected_stocks_df['margin_of_safety'].sum()
            mos_weight_component = 0.25 * mos_normalized
        else:
            print("Warning: 'margin_of_safety' not suitable for weighting in ValuePlay. Using 0 for this component.")
            mos_weight_component = 0.0

        # Quality Score component (needs normalization)
        if 'quality_score' in selected_stocks_df.columns and selected_stocks_df['quality_score'].sum() > 0:
            quality_normalized = selected_stocks_df['quality_score'] / selected_stocks_df['quality_score'].sum()
            quality_weight_component = 0.25 * quality_normalized
        else:
            print("Warning: 'quality_score' not suitable for weighting in ValuePlay. Using 0 for this component.")
            quality_weight_component = 0.0

        selected_stocks_df['weight'] = equal_weight_component + mos_weight_component + quality_weight_component

        # Final normalization of weights
        if selected_stocks_df['weight'].sum() > 0:
            selected_stocks_df['weight'] = selected_stocks_df['weight'] / selected_stocks_df['weight'].sum()
        elif not selected_stocks_df.empty:  # All weights are zero or NaN
            selected_stocks_df['weight'] = 1.0 / len(selected_stocks_df)

        # Ensure essential columns for portfolio history
        output_columns = ['symbol', 'weight']
        if 'share_price' in selected_stocks_df.columns:
            output_columns.append('share_price')
        else:
            selected_stocks_df['share_price'] = 1.0  # Placeholder if missing
            output_columns.append('share_price')

        # Add other useful columns if they exist
        for col in ['year', 'annual_return', 'market_cap', 'pe_ratio', 'pb_ratio', 'roe', 'margin_of_safety']:
            if col in selected_stocks_df.columns and col not in output_columns:
                output_columns.append(col)

        final_portfolio_df = selected_stocks_df[output_columns].copy()
        final_portfolio_df.dropna(subset=['weight'], inplace=True)  # Should not be needed with fallbacks

        return {
            'portfolio': final_portfolio_df,
            'metrics': self.calculate_metrics(final_portfolio_df)
        }

    def calculate_metrics(self, portfolio_for_period: pd.DataFrame) -> Dict[str, float]:
        """
        Calculates (placeholder) performance metrics for the selected portfolio.
        """
        avg_margin_of_safety = 0.0
        avg_roe = 0.0
        if not portfolio_for_period.empty and 'weight' in portfolio_for_period.columns:
            weighted_sum_exists = portfolio_for_period['weight'].sum() > 0
            if 'margin_of_safety' in portfolio_for_period.columns and weighted_sum_exists:
                avg_margin_of_safety = (portfolio_for_period['margin_of_safety'] * portfolio_for_period['weight']).sum()
            if 'roe' in portfolio_for_period.columns and weighted_sum_exists:
                avg_roe = (portfolio_for_period['roe'] * portfolio_for_period['weight']).sum()
        return {
            'value_portfolio_size': len(portfolio_for_period),
            'value_avg_margin_of_safety': avg_margin_of_safety,
            'value_avg_roe': avg_roe,
        }
