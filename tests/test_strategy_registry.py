"""Unit tests for the StrategyRegistry class.

These tests verify the functionality of the `StrategyRegistry` for registering,
retrieving, and managing investment strategy implementations.
"""
from typing import Any, Dict

import pandas as pd
import pytest

from backtest.strategies.base import Strategy
from backtest.strategy_registry import StrategyRegistry


# Dummy strategies for testing registration.
class MockStrategy(Strategy):
    """A simple mock strategy for testing purposes."""
    def execute_strategy(
        self, data: pd.DataFrame, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Executes the mock strategy."""
        return {"portfolio": pd.DataFrame(), "metrics": {"mock_metric": 1.0}}


class AnotherMockStrategy(Strategy):
    """Another mock strategy with a specific description."""
    def execute_strategy(
        self, data: pd.DataFrame, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Executes another mock strategy."""
        return {"portfolio": pd.DataFrame(), "metrics": {"another_metric": 1.0}}


@pytest.mark.unit
class TestStrategyRegistry:
    """Test suite for the `StrategyRegistry` class."""

    @pytest.fixture(autouse=True)
    def _clear_registry_before_each_test(self):
        """Ensures the registry is clean before each test method.

        This fixture uses `autouse=True` to automatically apply to all tests
        in this class, clearing the `_strategies` attribute of the
        `StrategyRegistry` before and after each test.
        """
        StrategyRegistry._strategies = {}  # pylint: disable=protected-access
        yield
        StrategyRegistry._strategies = {}  # pylint: disable=protected-access

    def test_register_and_get_class(self):
        """Tests strategy registration and class retrieval."""
        StrategyRegistry.register("mock_strat", MockStrategy, "A mock strategy.")
        assert StrategyRegistry.get_strategy_class("mock_strat") == MockStrategy
        assert StrategyRegistry.get_strategy_class("nonexistent") is None

    def test_create_strategy(self):
        """Tests the creation of strategy instances from the registry."""
        StrategyRegistry.register("mock_strat", MockStrategy)
        instance = StrategyRegistry.create_strategy("mock_strat")
        assert isinstance(instance, MockStrategy)
        assert StrategyRegistry.create_strategy("nonexistent") is None

    def test_get_strategy_description(self):
        """Tests retrieval of strategy descriptions."""
        StrategyRegistry.register("mock_strat1", MockStrategy, "Explicit description.")
        StrategyRegistry.register("mock_strat2", AnotherMockStrategy)  # Uses docstring.

        assert StrategyRegistry.get_strategy_description("mock_strat1") == "Explicit description."
        assert StrategyRegistry.get_strategy_description("mock_strat2") == \
               "Another mock strategy with a specific description." # From class docstring.
        assert StrategyRegistry.get_strategy_description("nonexistent") is None

    def test_list_strategies(self):
        """Tests listing of all registered strategies."""
        assert not StrategyRegistry.list_strategies()

        StrategyRegistry.register("mock_strat1", MockStrategy, "Desc1")
        StrategyRegistry.register("mock_strat2", AnotherMockStrategy, "Desc2_override")

        strategies = StrategyRegistry.list_strategies()
        assert len(strategies) == 2
        # Convert list of dicts to a set of tuples for easier comparison (order doesn't matter).
        strategy_set = {(s['name'], s['description']) for s in strategies}
        expected_set = {('mock_strat1', 'Desc1'), ('mock_strat2', 'Desc2_override')}
        assert strategy_set == expected_set

    def test_is_strategy_registered(self):
        """Tests checking if a strategy is registered."""
        assert not StrategyRegistry.is_strategy_registered("mock_strat")
        StrategyRegistry.register("mock_strat", MockStrategy)
        assert StrategyRegistry.is_strategy_registered("mock_strat")

    def test_execute_strategy(self):
        """Tests execution of a registered strategy."""
        StrategyRegistry.register("mock_strat", MockStrategy)
        dummy_data = pd.DataFrame()
        dummy_params = {}
        result = StrategyRegistry.execute_strategy("mock_strat", dummy_data, dummy_params)
        assert "portfolio" in result
        assert result["metrics"]["mock_metric"] == 1.0

    def test_execute_unregistered_strategy(self):
        """Tests that executing an unregistered strategy raises a ValueError."""
        with pytest.raises(
            ValueError,
            match="Strategy implementation not found for 'unregistered'"
        ):
            StrategyRegistry.execute_strategy("unregistered", pd.DataFrame(), {})

    def test_register_uses_class_docstring_if_no_description(self):
        """Tests that class docstring is used if no explicit description is provided."""
        class DocstringStrategy(Strategy):
            """This is the docstring description."""
            def execute_strategy(
                self, data: pd.DataFrame, params: Dict[str, Any]
            ) -> Dict[str, Any]:
                """Executes the docstring strategy."""
                return {}

        StrategyRegistry.register("doc_strat", DocstringStrategy)
        assert StrategyRegistry.get_strategy_description("doc_strat") == \
               "This is the docstring description."

    def test_register_default_description_if_no_docstring_or_explicit(self):
        """Tests default description generation when no docstring or explicit one is given."""
        class NoDocstringStrategy(Strategy): # No docstring here.
            def execute_strategy(
                self, data: pd.DataFrame, params: Dict[str, Any]
            ) -> Dict[str, Any]:
                """Executes the no-docstring strategy."""
                return {}

        StrategyRegistry.register("no_doc_strat", NoDocstringStrategy)
        assert StrategyRegistry.get_strategy_description("no_doc_strat") == \
               "Strategy: no_doc_strat"