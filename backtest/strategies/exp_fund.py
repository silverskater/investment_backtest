"""Implementation of the Exponential Fund strategy.

This strategy focuses on identifying high-growth companies and applies
weighting based on market capitalization or a hybrid model. It also
includes optional risk management features.
"""
from typing import Any, Dict

import pandas as pd

from .base import Strategy


class ExpFundStrategy(Strategy):
    """Implements the Exponential Fund investment strategy.

    This strategy selects companies based on sales growth and market cap rank.
    It supports market-cap weighting, hybrid weighting (market-cap and P/S ratio),
    and a risk overlay based on portfolio average P/S ratio.
    """

    def execute_strategy(
        self, data_for_period: pd.DataFrame, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Executes the Exponential Fund strategy for a single period.

        Filters companies based on sales growth and market cap rank, applies
        weighting, and optionally a risk overlay.

        Args:
            data_for_period: DataFrame with market data for the period.
                             Expected columns: 'sales_growth_5y', 'market_cap_rank',
                             'share_price', and optionally 'ps_ratio', 'market_cap'.
            params: Dictionary of strategy parameters including:
                'growth_threshold': Minimum sales growth.
                'top_n': Number of top companies by market cap rank.
                'hybrid_weighting': If True, use hybrid weighting.
                'risk_overlay': If True, apply P/S-based risk overlay.
                'ps_threshold': P/S threshold for risk overlay.

        Returns:
            A dictionary with 'portfolio' (DataFrame of selected stocks with
            weights and 'share_price') and 'metrics' (strategy-specific
            metrics for the period).
        """
        growth_threshold = params.get('growth_threshold', 0.2)
        top_n = params.get('top_n', 10)
        hybrid_weighting = params.get('hybrid_weighting', False)
        risk_overlay = params.get('risk_overlay', False)
        ps_threshold = params.get('ps_threshold', 10.0)

        current_data = data_for_period.copy()

        growth_filter_condition = (
            (current_data.get('sales_growth_5y', pd.Series(dtype=float)) >= growth_threshold) &
            (current_data.get('market_cap_rank', pd.Series(dtype=int)) <= top_n)
        )

        columns_to_keep = [
            col for col in data_for_period.columns
            if col in ['symbol', 'year', 'market_cap', 'sales_growth_5y',
                       'ps_ratio', 'annual_return', 'share_price', 'market_cap_rank']
        ]
        if not columns_to_keep:
            growth_stocks = pd.DataFrame(columns=['weight']) # Should have at least symbol
        else:
            growth_stocks = current_data.loc[growth_filter_condition, columns_to_keep].copy()

        if growth_stocks.empty:
            portfolio_columns = columns_to_keep + ['weight'] if columns_to_keep else ['symbol', 'weight', 'share_price']
            if 'symbol' not in portfolio_columns: portfolio_columns.append('symbol')
            if 'share_price' not in portfolio_columns: portfolio_columns.append('share_price')
            return {
                'portfolio': pd.DataFrame(columns=list(set(portfolio_columns))), # Ensure unique columns
                'metrics': ExpFundStrategy.calculate_metrics(pd.DataFrame())
            }

        if 'share_price' not in growth_stocks.columns:
            growth_stocks['share_price'] = 1.0  # Placeholder.

        if hybrid_weighting and 'ps_ratio' in growth_stocks.columns:
            ExpFundStrategy.apply_hybrid_weighting(growth_stocks)
        else:
            ExpFundStrategy.apply_market_cap_weighting(growth_stocks)

        if risk_overlay and 'ps_ratio' in growth_stocks.columns:
            ExpFundStrategy.apply_risk_overlay(growth_stocks, ps_threshold)

        if not growth_stocks.empty:
            if 'weight' in growth_stocks.columns and growth_stocks['weight'].sum() > 0:
                growth_stocks.loc[:, 'weight'] = growth_stocks['weight'] / growth_stocks['weight'].sum()
            elif 'weight' in growth_stocks.columns:
                growth_stocks.loc[:, 'weight'] = 1.0 / len(growth_stocks) if len(growth_stocks) > 0 else 0.0
            else:
                growth_stocks.loc[:, 'weight'] = 1.0 / len(growth_stocks) if len(growth_stocks) > 0 else 0.0

        return {
            'portfolio': growth_stocks,
            'metrics': ExpFundStrategy.calculate_metrics(growth_stocks)
        }

    @staticmethod
    def apply_market_cap_weighting(data: pd.DataFrame) -> None:
        """Applies market cap weighting to the stocks.

        Modifies the DataFrame in place by adding/updating a 'weight' column.
        If total market capitalization is not positive, it falls back to
        equal weighting.

        Args:
            data: DataFrame of stocks; must include 'market_cap'.
                  Modified in-place.
        """
        if data.empty or 'market_cap' not in data.columns:
            if not data.empty:
                data.loc[:, 'weight'] = 0.0
            return

        total_market_cap = pd.to_numeric(data['market_cap'], errors='coerce').sum()
        if total_market_cap > 0:
            data.loc[:, 'weight'] = pd.to_numeric(
                data['market_cap'], errors='coerce'
            ) / total_market_cap
        else:
            data.loc[:, 'weight'] = 1.0 / len(data) if len(data) > 0 else 0.0

    @staticmethod
    def apply_hybrid_weighting(data: pd.DataFrame) -> None:
        """Applies hybrid weighting (50% market cap, 50% inverse P/S ratio).

        Modifies the DataFrame in place by adding/updating a 'weight' column.
        Falls back to market cap weighting if 'ps_ratio' is unavailable or
        if P/S based weighting cannot be determined.

        Args:
            data: DataFrame of stocks; must include 'market_cap' and 'ps_ratio'.
                  Modified in-place.
        """
        if data.empty:
            return

        if 'market_cap' not in data.columns or 'ps_ratio' not in data.columns:
            ExpFundStrategy.apply_market_cap_weighting(data)
            return

        market_cap_series = pd.to_numeric(data['market_cap'], errors='coerce')
        total_market_cap = market_cap_series.sum()

        market_cap_w = (market_cap_series / total_market_cap
                        if total_market_cap > 0
                        else pd.Series([1.0 / len(data) if len(data) > 0 else 0.0] * len(data), index=data.index))

        ps_ratios_numeric = pd.to_numeric(data['ps_ratio'], errors='coerce').fillna(float('inf'))
        inv_ps_values = 1.0 / ps_ratios_numeric.replace(to_replace=0, value=float('inf'))
        inv_ps_values[ps_ratios_numeric <= 0] = 0  # Negative or zero P/S gets zero inverse weight.

        total_inv_ps = inv_ps_values.sum()
        ps_w = (inv_ps_values / total_inv_ps
                if total_inv_ps > 0 and total_inv_ps != float('inf')
                else pd.Series([1.0 / len(data) if len(data) > 0 else 0.0] * len(data), index=data.index))

        data.loc[:, 'weight'] = 0.5 * market_cap_w + 0.5 * ps_w

    @staticmethod
    def apply_risk_overlay(data: pd.DataFrame, ps_threshold: float = 10.0) -> None:
        """Applies a risk overlay based on the portfolio's average P/S ratio.

        Reduces exposure (by scaling down weights) if the weighted average
        P/S ratio of the portfolio exceeds the specified `ps_threshold`.
        Modifies the DataFrame in-place.

        Args:
            data: DataFrame of stocks; must include 'ps_ratio' and 'weight'.
                  Modified in-place.
            ps_threshold: The P/S ratio threshold. If the portfolio's
                          average P/S exceeds this, exposure is reduced.
        """
        if data.empty or 'ps_ratio' not in data.columns or 'weight' not in data.columns:
            return

        ps_ratios_numeric = pd.to_numeric(data['ps_ratio'], errors='coerce')
        weights_numeric = pd.to_numeric(data['weight'], errors='coerce')

        valid_entries = ps_ratios_numeric.notna() & weights_numeric.notna()
        if not valid_entries.any():
            return

        weighted_ps_sum = (ps_ratios_numeric[valid_entries] * weights_numeric[valid_entries]).sum()
        total_valid_weight = weights_numeric[valid_entries].sum()

        if total_valid_weight == 0:
            return

        average_weighted_ps = weighted_ps_sum / total_valid_weight

        if average_weighted_ps > ps_threshold:
            risk_factor = 1.0
            if average_weighted_ps > 0 and ps_threshold > 0:
                risk_factor = ps_threshold / average_weighted_ps
            risk_factor = min(risk_factor, 1.0)
            data.loc[:, 'weight'] = weights_numeric * risk_factor

    @staticmethod
    def calculate_metrics(portfolio_for_period: pd.DataFrame) -> Dict[str, float]:
        """Calculates strategy-specific metrics for the portfolio in a given period.

        Args:
            portfolio_for_period: DataFrame representing the selected portfolio
                                  for the period.

        Returns:
            A dictionary of calculated metrics like portfolio size, average
            sales growth, and average P/S ratio.
        """
        metrics = {
            'portfolio_size': len(portfolio_for_period),
            'average_growth': 0.0,
            'average_ps': 0.0,
        }
        if not portfolio_for_period.empty:
            if 'sales_growth_5y' in portfolio_for_period.columns:
                metrics['average_growth'] = pd.to_numeric(
                    portfolio_for_period['sales_growth_5y'], errors='coerce'
                ).mean()
            if 'ps_ratio' in portfolio_for_period.columns:
                metrics['average_ps'] = pd.to_numeric(
                    portfolio_for_period['ps_ratio'], errors='coerce'
                ).mean()
        return metrics