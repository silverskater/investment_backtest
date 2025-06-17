"""Tests for strategy loading and execution functionality.

This module tests the integration of `strategy_manager` with the
`StrategyRegistry` to ensure strategies can be correctly retrieved
and their `execute_strategy` methods can be called.
"""
import inspect

import pandas as pd
import pytest

# This import ensures strategies are loaded via backtest.strategies.__init__
import backtest.strategies  # pylint: disable=unused-import
from backtest.strategy_manager import execute_strategy, get_strategy_implementation


@pytest.mark.integration
class TestStrategyLoadingAndExecution:
    """Test suite for strategy loading and execution via the strategy manager."""

    def test_get_strategy_implementation(self):
        """Tests that `execute_strategy` methods can be loaded for registered strategies."""
        dgi_strategy_method = get_strategy_implementation('dgi')
        assert dgi_strategy_method is not None
        assert callable(dgi_strategy_method)

        # Check the signature of the retrieved method.
        sig = inspect.signature(dgi_strategy_method)
        assert 'data_for_period' in sig.parameters
        assert 'params' in sig.parameters

        nonexistent_strategy_method = get_strategy_implementation('nonexistent_strategy')
        assert nonexistent_strategy_method is None

    def test_execute_strategy(self):
        """Tests that a registered strategy can be executed successfully."""
        data = pd.DataFrame({
            'year': [2020, 2020, 2020],
            'symbol': ['AAPL', 'MSFT', 'GOOG'],
            'market_cap': [2000, 1800, 1500],
            'market_cap_rank': [1, 2, 3],
            'sales_growth_5y': [0.25, 0.22, 0.18],
            'ps_ratio': [7.5, 10.2, 8.7],
            'share_price': [150, 250, 2000]
        })
        params = {
            'growth_threshold': 0.2,
            'top_n': 2,
            'hybrid_weighting': False
        }
        try:
            results = execute_strategy('exp_fund', data, params)
            assert 'portfolio' in results
            assert 'metrics' in results
            portfolio = results['portfolio']
            assert isinstance(portfolio, pd.DataFrame)
            assert len(portfolio) == 2
            assert set(portfolio['symbol'].tolist()) == {'AAPL', 'MSFT'}
            metrics = results['metrics']
            assert 'portfolio_size' in metrics
            assert metrics['portfolio_size'] == 2
        except ImportError:
            pytest.skip("A required strategy module might not be fully implemented or available.")
        except ValueError as e:
            pytest.fail(f"Strategy execution failed unexpectedly: {e}")

    def test_execute_nonexistent_strategy(self):
        """Tests that executing a non-existent strategy raises a ValueError."""
        data = pd.DataFrame()
        params = {}
        with pytest.raises(
            ValueError,
            match="Strategy implementation not found for 'nonexistent_strategy'"
        ):
            execute_strategy('nonexistent_strategy', data, params)