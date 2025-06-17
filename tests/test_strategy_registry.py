import pytest
from backtest.strategy_registry import StrategyRegistry
from backtest.strategies.base import Strategy
from typing import Dict, Any
import pandas as pd


# Dummy strategy for testing registration
class MockStrategy(Strategy):
    def execute_strategy(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        return {"portfolio": pd.DataFrame(), "metrics": {"mock_metric": 1.0}}


class AnotherMockStrategy(Strategy):
    """Another mock strategy description."""

    def execute_strategy(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        return {"portfolio": pd.DataFrame(), "metrics": {"another_metric": 1.0}}


@pytest.mark.unit
class TestStrategyRegistry:

    @pytest.fixture(autouse=True)
    def clear_registry_before_each_test(self):
        """Ensures the registry is clean before each test."""
        StrategyRegistry._strategies = {}
        yield  # Test runs here
        StrategyRegistry._strategies = {}  # Clean up after

    def test_register_and_get_class(self):
        StrategyRegistry.register("mock_strat", MockStrategy, "A mock strategy.")
        assert StrategyRegistry.get_strategy_class("mock_strat") == MockStrategy
        assert StrategyRegistry.get_strategy_class("nonexistent") is None

    def test_create_strategy(self):
        StrategyRegistry.register("mock_strat", MockStrategy)
        instance = StrategyRegistry.create_strategy("mock_strat")
        assert isinstance(instance, MockStrategy)
        assert StrategyRegistry.create_strategy("nonexistent") is None

    def test_get_strategy_description(self):
        StrategyRegistry.register("mock_strat1", MockStrategy, "Explicit description.")
        StrategyRegistry.register("mock_strat2", AnotherMockStrategy)  # Uses docstring

        assert StrategyRegistry.get_strategy_description("mock_strat1") == "Explicit description."
        assert StrategyRegistry.get_strategy_description("mock_strat2") == "Another mock strategy description."
        assert StrategyRegistry.get_strategy_description("nonexistent") is None

    def test_list_strategies(self):
        assert StrategyRegistry.list_strategies() == []
        StrategyRegistry.register("mock_strat1", MockStrategy, "Desc1")
        StrategyRegistry.register("mock_strat2", AnotherMockStrategy, "Desc2")

        strategies = StrategyRegistry.list_strategies()
        assert len(strategies) == 2
        # Convert list of dicts to a set of tuples for easier comparison (order doesn't matter)
        strategy_set = {(s['name'], s['description']) for s in strategies}
        expected_set = {('mock_strat1', 'Desc1'), ('mock_strat2', 'Desc2')}
        assert strategy_set == expected_set

    def test_is_strategy_registered(self):
        assert not StrategyRegistry.is_strategy_registered("mock_strat")
        StrategyRegistry.register("mock_strat", MockStrategy)
        assert StrategyRegistry.is_strategy_registered("mock_strat")

    def test_execute_strategy(self):
        StrategyRegistry.register("mock_strat", MockStrategy)
        dummy_data = pd.DataFrame()
        dummy_params = {}
        result = StrategyRegistry.execute_strategy("mock_strat", dummy_data, dummy_params)
        assert "portfolio" in result
        assert result["metrics"]["mock_metric"] == 1.0

    def test_execute_unregistered_strategy(self):
        with pytest.raises(ValueError, match="Strategy implementation not found for 'unregistered'"):
            StrategyRegistry.execute_strategy("unregistered", pd.DataFrame(), {})

    def test_register_uses_class_docstring_if_no_description(self):
        class DocstringStrategy(Strategy):
            """This is the docstring description."""

            def execute_strategy(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
                return {}

        StrategyRegistry.register("doc_strat", DocstringStrategy)
        assert StrategyRegistry.get_strategy_description("doc_strat") == "This is the docstring description."

    def test_register_default_description_if_no_docstring_or_explicit(self):
        class NoDocstringStrategy(Strategy):
            def execute_strategy(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
                return {}

        StrategyRegistry.register("no_doc_strat", NoDocstringStrategy)
        assert StrategyRegistry.get_strategy_description("no_doc_strat") == "Strategy: no_doc_strat"
