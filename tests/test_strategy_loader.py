"""Tests for strategy loading functionality."""
import pytest
import pandas as pd
from backtest.strategy_manager import get_strategy_implementation, execute_strategy


def test_get_strategy_implementation():
    """Test that strategy implementations can be loaded."""
    # Test with an existing strategy
    dgi_strategy = get_strategy_implementation('dgi')
    assert dgi_strategy is not None
    assert callable(dgi_strategy)

    # Verify the strategy has the correct signature
    import inspect
    sig = inspect.signature(dgi_strategy)
    # Should have 'self' (for class methods) or accept at least data and params
    assert len(sig.parameters) >= 2

    # Test with a non-existent strategy
    nonexistent_strategy = get_strategy_implementation('nonexistent_strategy')
    assert nonexistent_strategy is None


def test_execute_strategy():
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

    # Define parameters
    params = {
        'growth_threshold': 0.2,
        'top_n': 2,
        'hybrid_weighting': False
    }

    # Execute strategy
    try:
        results = execute_strategy('exp_fund', data, params)

        # Check results structure
        assert 'portfolio' in results
        assert 'metrics' in results

        # Check that only top 2 companies with growth >= 20% are included
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


def test_execute_nonexistent_strategy():
    """Test that executing a non-existent strategy raises an error."""
    data = pd.DataFrame()
    params = {}

    with pytest.raises(ValueError):
        execute_strategy('nonexistent_strategy', data, params)
