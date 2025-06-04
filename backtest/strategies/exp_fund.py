"""Implementation of the Exponential Fund strategy."""

from typing import Dict, Any

import pandas as pd

from .base import Strategy


class ExpFundStrategy(Strategy):
    """
    The Exponential Fund investment strategy.

    This strategy focuses on companies with strong sales growth and applies
    various weighting and risk management techniques.
    """

    def execute_strategy(self, data_for_period: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes the Exponential Fund strategy for a single period.

        The method filters the provided period's data based on growth criteria,
        applies weighting (market-cap or hybrid), and optionally applies a risk
        overlay. The output portfolio includes 'share_price'.

        Args:
            data_for_period: DataFrame containing market data for the specific
                             period (e.g., one year). Expected columns
                             include 'sales_growth_5y', 'market_cap_rank',
                             'share_price', and optionally 'ps_ratio'.
            params: A dictionary of strategy parameters:
                growth_threshold: Minimum annual sales growth (e.g., 0.2 for 20%).
                top_n: Number of top-ranked companies to consider.
                hybrid_weighting: Boolean, if True, use hybrid weighting.
                risk_overlay: Boolean, if True, apply P/S-based risk overlay.
                ps_threshold: P/S ratio threshold for the risk overlay.

        Returns:
            A dictionary containing the resulting 'portfolio' (DataFrame of selected
            stocks with weights and share_price for the period) and 'metrics' (Dict, placeholder).
        """
        # Extract parameters with sensible defaults
        growth_threshold = params.get('growth_threshold', 0.2)
        top_n = params.get('top_n', 10)
        hybrid_weighting = params.get('hybrid_weighting', False)
        risk_overlay = params.get('risk_overlay', False)
        ps_threshold = params.get('ps_threshold', 10.0)

        # Work on a copy of the data for this period to avoid side effects
        current_data = data_for_period.copy()

        # Filter for companies meeting sales growth and market cap rank criteria.
        # A .copy() is used to ensure 'growth_stocks' is an independent DataFrame,
        # preventing potential SettingWithCopyWarning when modified later.
        growth_filter_condition = (
            (current_data.get('sales_growth_5y', pd.Series(dtype=float)) >= growth_threshold) &
            (current_data.get('market_cap_rank', pd.Series(dtype=int)) <= top_n)
        )

        # Select all necessary columns, including share_price
        columns_to_keep = [
            col for col in data_for_period.columns
            if col in ['symbol', 'year', 'market_cap', 'sales_growth_5y',
                       'ps_ratio', 'annual_return', 'share_price', 'market_cap_rank'] # Add other relevant cols
        ]
        if not columns_to_keep: # Should not happen if data_for_period is valid
             growth_stocks = pd.DataFrame(columns=['weight'])
        else:
            growth_stocks = current_data.loc[growth_filter_condition, columns_to_keep].copy()


        if growth_stocks.empty:
            # If no stocks meet criteria, return an empty portfolio (or 100% cash)
            # For simplicity, returning an empty DataFrame. The rebalance logic in cli.py
            # will handle rebalancing to cash if the portfolio is empty.
            portfolio_columns = columns_to_keep + ['weight'] if columns_to_keep else ['symbol', 'weight', 'share_price']
            # Minimal required columns if columns_to_keep was empty
            if 'symbol' not in portfolio_columns: portfolio_columns.append('symbol')
            if 'share_price' not in portfolio_columns: portfolio_columns.append('share_price')

            return {
                'portfolio': pd.DataFrame(columns=data_for_period.columns.tolist() + ['weight']), # Ensure schema consistency
                'metrics': self.calculate_metrics(pd.DataFrame())
            }

        if 'share_price' not in growth_stocks.columns:
            # This should not happen if data_for_period has share_price
            # and columns_to_keep includes it.
            growth_stocks['share_price'] = 1.0 # Placeholder, not ideal

        # Apply the selected weighting strategy
        if hybrid_weighting and 'ps_ratio' in growth_stocks.columns:
            ExpFundStrategy.apply_hybrid_weighting(growth_stocks)
        else:
            # Default to standard market cap weighting
            ExpFundStrategy.apply_market_cap_weighting(growth_stocks)

        # Apply risk overlay if enabled and P/S ratio is available
        if risk_overlay and 'ps_ratio' in growth_stocks.columns:
            ExpFundStrategy.apply_risk_overlay(growth_stocks, ps_threshold)

        # Normalize weights if they don't sum to 1 (or handle as needed)
        # This step was previously in cli.py, makes sense to ensure strategy output is normalized
        if not growth_stocks.empty:
            if 'weight' in growth_stocks.columns and growth_stocks['weight'].sum() > 0:
                growth_stocks.loc[:, 'weight'] = growth_stocks['weight'] / growth_stocks['weight'].sum()
            elif 'weight' in growth_stocks.columns: # Weights sum to 0 or are all NaN
                 # Fallback to equal weight if sum is 0 but stocks exist
                growth_stocks.loc[:, 'weight'] = 1.0 / len(growth_stocks) if len(growth_stocks) > 0 else 0.0
            else: # 'weight' column was not created
                growth_stocks.loc[:, 'weight'] = 1.0 / len(growth_stocks) if len(growth_stocks) > 0 else 0.0


        results = {
            'portfolio': growth_stocks,
            'metrics': self.calculate_metrics(growth_stocks)  # Placeholder metrics
        }
        return results

    @staticmethod
    def apply_market_cap_weighting(data: pd.DataFrame) -> None:
        """
        Applies market cap weighting to the stocks.

        Modifies the DataFrame in place by adding/updating a 'weight' column.
        If total market capitalization is not positive, it falls back to
        equal weighting.

        Args:
            data: DataFrame of stocks; must include 'market_cap'.
                  Modified in-place.
        """
        if data.empty or 'market_cap' not in data.columns:
            if not data.empty: # If not empty but missing column, assign 0 weight
                 data.loc[:, 'weight'] = 0.0
            return

        total_market_cap = pd.to_numeric(data['market_cap'], errors='coerce').sum()
        if total_market_cap > 0:
            data.loc[:, 'weight'] = pd.to_numeric(data['market_cap'], errors='coerce') / total_market_cap
        else:
            data.loc[:, 'weight'] = 1.0 / len(data) if len(data) > 0 else 0.0


    @staticmethod
    def apply_hybrid_weighting(data: pd.DataFrame) -> None:
        """
        Applies hybrid weighting (50% market cap, 50% inverse P/S ratio).

        Modifies the DataFrame in place by adding/updating a 'weight' column.
        Falls back to market cap weighting if 'ps_ratio' is unavailable.

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

        if total_market_cap > 0:
            market_cap_w = market_cap_series / total_market_cap
        else:
            market_cap_w = pd.Series([1.0 / len(data) if len(data) > 0 else 0.0] * len(data), index=data.index)


        ps_ratios_numeric = pd.to_numeric(data['ps_ratio'], errors='coerce').fillna(float('inf'))
        inv_ps_values = 1.0 / ps_ratios_numeric.replace(to_replace=0, value=float('inf'))
        inv_ps_values[ps_ratios_numeric <= 0] = 0 # Negative or zero P/S gets zero inverse weight

        total_inv_ps = inv_ps_values.sum()
        if total_inv_ps > 0 and total_inv_ps != float('inf'):
            ps_w = inv_ps_values / total_inv_ps
        else:
            ps_w = pd.Series([1.0 / len(data) if len(data) > 0 else 0.0] * len(data), index=data.index)

        data.loc[:, 'weight'] = 0.5 * market_cap_w + 0.5 * ps_w

    @staticmethod
    def apply_risk_overlay(data: pd.DataFrame, ps_threshold: float = 10.0) -> None:
        """
        Applies a risk overlay based on the portfolio's average P/S ratio.

        Reduces exposure (by scaling down weights) if the weighted average
        P/S ratio of the portfolio exceeds the specified `ps_threshold`.
        Modifies the DataFrame in place.

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
            if average_weighted_ps > 0 and ps_threshold > 0: # Ensure ps_threshold is positive
                risk_factor = ps_threshold / average_weighted_ps
            # If average_weighted_ps is not positive, or ps_threshold is not positive,
            # risk_factor remains 1.0 (no reduction) unless ps_threshold is very small.
            # This logic implies reduction only if avg P/S is high and positive.

            risk_factor = min(risk_factor, 1.0) # Cap factor at 1 (no increase in exposure)
            data.loc[:, 'weight'] = weights_numeric * risk_factor


    def calculate_metrics(self, portfolio_for_period: pd.DataFrame) -> Dict[str, float]:
        """
        Calculates (placeholder) performance metrics for the selected portfolio for a period.

        Args:
            portfolio_for_period: DataFrame representing the selected portfolio for the period.

        Returns:
            A dictionary of calculated metrics.
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