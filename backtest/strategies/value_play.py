"""Implementation of the Value Play Investing strategy.

This strategy focuses on identifying undervalued companies with strong
fundamentals, a margin of safety, and long-term compounding potential.
"""
from typing import Any, Dict

import numpy as np
import pandas as pd

from .base import Strategy


class ValuePlayStrategy(Strategy):
    """Implements the Value Investing strategy.

    Selects stocks based on a combination of undervaluation metrics (P/E, P/B),
    financial health (current ratio, debt-to-equity, ROE), and quality filters
    (S&P Quality Rank, EPS growth, margin of safety). Portfolio weights are
    determined by a hybrid approach.

    Attributes:
        SP_QUALITY_RANK_MAPPING: Maps S&P quality ranks to numerical values.
        MIN_SP_QUALITY_NUMERIC: Minimum numerical S&P quality rank required.
        DEFAULT_TOP_N_VALUE: Default number of stocks if 'top_n' not in params.
    """

    SP_QUALITY_RANK_MAPPING = {
        'A+': 6, 'A': 5, 'A-': 4,
        'B+': 3, 'B': 2, 'B-': 1,
    }
    MIN_SP_QUALITY_NUMERIC = SP_QUALITY_RANK_MAPPING['B+']
    DEFAULT_TOP_N_VALUE = 20

    def _map_sp_quality_to_numeric(self, quality_series: pd.Series) -> pd.Series:
        """Maps S&P quality string ranks to numerical values.

        Args:
            quality_series: A pandas Series containing S&P quality ranks.

        Returns:
            A pandas Series with numerical quality ranks. Unmapped values default to 0.
        """
        return quality_series.map(self.SP_QUALITY_RANK_MAPPING).fillna(0)

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

        if 'sp_quality' in candidates_df.columns:
            candidates_df['sp_quality_numeric'] = self._map_sp_quality_to_numeric(
                candidates_df['sp_quality']
            )
        else:
            print("Warning: 'sp_quality' column not found for ValuePlayStrategy. "
                  "Skipping S&P Quality Rank filter.")
            candidates_df['sp_quality_numeric'] = np.nan

        pe_condition = (
                candidates_df.get('pe_ratio', pd.Series(dtype=float)) <
                candidates_df.get('sector_median_pe', pd.Series(dtype=float)) * 0.4
        )
        pb_condition = (candidates_df.get('pb_ratio', pd.Series(dtype=float)) < 1.0)
        current_ratio_condition = (candidates_df.get('current_ratio', pd.Series(dtype=float)) > 1.5)
        debt_equity_condition = (
                candidates_df.get('debt_equity', pd.Series(dtype=float)) <
                candidates_df.get('industry_avg_debt_equity', pd.Series(dtype=float))
        )
        roe_condition = (candidates_df.get('roe', pd.Series(dtype=float)) > 0.15)
        eps_growth_condition = (candidates_df.get('eps_growth_5y', pd.Series(dtype=float)) > 0)
        sp_quality_condition = (
            candidates_df.get('sp_quality_numeric', pd.Series(dtype=float)) >= self.MIN_SP_QUALITY_NUMERIC
        )
        margin_of_safety_condition = (candidates_df.get('margin_of_safety', pd.Series(dtype=float)) > 0.25)

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

        if 'composite_value_score' in screened_df.columns:
            screened_df = screened_df.sort_values('composite_value_score', ascending=False)
        elif 'margin_of_safety' in screened_df.columns:
            print("Warning: 'composite_value_score' not found. Sorting by 'margin_of_safety'.")
            screened_df = screened_df.sort_values('margin_of_safety', ascending=False)
        else:
            print("Warning: Neither 'composite_value_score' nor 'margin_of_safety' found. "
                  "Cannot sort for ValuePlayStrategy.")

        selected_stocks_df = screened_df.head(top_n).copy()

        if selected_stocks_df.empty:
            return {
                'portfolio': pd.DataFrame(columns=['symbol', 'weight', 'share_price']),
                'metrics': ValuePlayStrategy.calculate_metrics(pd.DataFrame())
            }

        num_selected = len(selected_stocks_df)
        equal_weight_component = 0.5 * (1 / num_selected if num_selected > 0 else 0)

        mos_weight_component = 0.0
        if 'margin_of_safety' in selected_stocks_df.columns and \
           selected_stocks_df['margin_of_safety'].sum() > 0:
            mos_normalized = selected_stocks_df['margin_of_safety'] / selected_stocks_df['margin_of_safety'].sum()
            mos_weight_component = 0.25 * mos_normalized
        else:
            print("Warning: 'margin_of_safety' not suitable for weighting. Using 0 for this component.")

        quality_weight_component = 0.0
        if 'quality_score' in selected_stocks_df.columns and \
           selected_stocks_df['quality_score'].sum() > 0:
            quality_normalized = selected_stocks_df['quality_score'] / selected_stocks_df['quality_score'].sum()
            quality_weight_component = 0.25 * quality_normalized
        else:
            print("Warning: 'quality_score' not suitable for weighting. Using 0 for this component.")

        selected_stocks_df['weight'] = equal_weight_component + mos_weight_component + quality_weight_component

        if selected_stocks_df['weight'].sum() > 0:
            selected_stocks_df['weight'] /= selected_stocks_df['weight'].sum()
        elif not selected_stocks_df.empty:
            selected_stocks_df['weight'] = 1.0 / len(selected_stocks_df)

        output_columns = ['symbol', 'weight']
        if 'share_price' in selected_stocks_df.columns:
            output_columns.append('share_price')
        else:
            selected_stocks_df['share_price'] = 1.0
            output_columns.append('share_price')

        for col in ['year', 'annual_return', 'market_cap', 'pe_ratio', 'pb_ratio', 'roe', 'margin_of_safety']:
            if col in selected_stocks_df.columns and col not in output_columns:
                output_columns.append(col)

        final_portfolio_df = selected_stocks_df[output_columns].copy()
        final_portfolio_df.dropna(subset=['weight'], inplace=True)

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
        if not portfolio_for_period.empty and 'weight' in portfolio_for_period.columns:
            weighted_sum_exists = portfolio_for_period['weight'].sum() > 0
            if 'margin_of_safety' in portfolio_for_period.columns and weighted_sum_exists:
                avg_margin_of_safety = (
                        portfolio_for_period['margin_of_safety'] * portfolio_for_period['weight']
                ).sum()
            if 'roe' in portfolio_for_period.columns and weighted_sum_exists:
                avg_roe = (portfolio_for_period['roe'] * portfolio_for_period['weight']).sum()
        return {
            'value_portfolio_size': len(portfolio_for_period),
            'value_avg_margin_of_safety': avg_margin_of_safety,
            'value_avg_roe': avg_roe,
        }