"""Tests for strategy loading functionality."""
import inspect
import pytest
import pandas as pd

from backtest.strategy_manager import get_strategy_implementation, execute_strategy

@pytest.mark.integration # Tests interaction with strategy_manager and potentially registered strategies
class TestStrategyLoadingAndExecution:
    def test_get_strategy_implementation(self):
        """Test that strategy implementations can be loaded."""
        # Test with an existing strategy
        dgi_strategy_method = get_strategy_implementation('dgi')
        assert dgi_strategy_method is not None
        assert callable(dgi_strategy_method)
        sig = inspect.signature(dgi_strategy_method)
        # The method returned is already bound to an instance or is a static/class method
        # that doesn't take 'self' as its first arg in the way execute_strategy is called.
        # It should accept 'data_for_period' and 'params'.
        assert 'data_for_period' in sig.parameters
        assert 'params' in sig.parameters

        # Test with a non-existent strategy
        nonexistent_strategy_method = get_strategy_implementation('nonexistent_strategy')
        assert nonexistent_strategy_method is None

    def test_execute_strategy(self):
        """Test that strategies can be executed."""
        # Create sample data
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
            # Check results structure
            assert 'portfolio' in results
            assert 'metrics' in results
            # Check that only the top 2 companies with growth >= 20% are included
            portfolio = results['portfolio']
            assert len(portfolio) == 2
            assert set(portfolio['symbol'].tolist()) == {'AAPL', 'MSFT'}
            # Check metrics
            metrics = results['metrics']
            assert 'portfolio_size' in metrics
            assert metrics['portfolio_size'] == 2
        except ImportError:
            pytest.skip("Strategy module not yet implemented")
        except ValueError as e:
            pytest.fail(f"Strategy execution failed: {e}")

    def test_execute_nonexistent_strategy(self):
        """Test that executing a non-existent strategy raises an error."""
        data = pd.DataFrame()
        params = {}
        with pytest.raises(ValueError, match="Strategy implementation not found for 'nonexistent_strategy'"):
            execute_strategy('nonexistent_strategy', data, params)
