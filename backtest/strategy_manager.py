"""Manages different investment strategy implementations.

This module provides a facade that delegates to the StrategyRegistry,
decoupling strategy management from the specific strategy implementations.
It ensures strategies are registered upon import and offers functions
to validate, list, retrieve, and execute strategies.
"""
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

# Ensure strategies are registered when this manager is imported.
# This executes backtest/strategies/__init__.py, which should
# scan and register all available strategies.
import backtest.strategies  # pylint: disable=unused-import
from backtest.strategy_registry import StrategyRegistry


def is_strategy_available(strategy_name: str) -> bool:
    """Checks if a strategy is registered and available.

    Args:
        strategy_name: The name of the strategy.

    Returns:
        True if the strategy is registered, False otherwise.
    """
    return StrategyRegistry.is_strategy_registered(strategy_name)


def list_available_strategies() -> List[Dict[str, str]]:
    """Lists all available and registered investment strategies.

    Returns:
        A list of dictionaries, where each dictionary contains the
        'name' and 'description' of a strategy.
    """
    return StrategyRegistry.list_strategies()


def validate_strategy(strategy_name: str) -> bool:
    """Validates that a strategy exists in the registry.

    Args:
        strategy_name: The name of the strategy to validate.

    Returns:
        True if the strategy is registered, False otherwise.
    """
    return StrategyRegistry.is_strategy_registered(strategy_name)


def get_strategy_description(strategy_name: str) -> Optional[str]:
    """Gets the description of a registered strategy.

    Args:
        strategy_name: The name of the strategy.

    Returns:
        The strategy's description string if found, None otherwise.
    """
    return StrategyRegistry.get_strategy_description(strategy_name)


def get_strategy_implementation(strategy_name: str) -> Optional[Callable]:
    """Gets the callable `execute_strategy` method for a strategy.

    This function instantiates the strategy class from the registry and
    returns its `execute_strategy` method, ready to be called.

    Args:
        strategy_name: The name of the strategy.

    Returns:
        The `execute_strategy` method of the instantiated strategy if found,
        None otherwise.
    """
    strategy = StrategyRegistry.create_strategy(strategy_name)
    return strategy.execute_strategy if strategy else None


def execute_strategy(
    strategy_name: str, data: pd.DataFrame, params: Dict[str, Any]
) -> Dict[str, Any]:
    """Executes a specific investment strategy on the provided market data.

    Args:
        strategy_name: The name of the strategy to execute.
        data: A pandas DataFrame containing the market data for the period.
        params: A dictionary of parameters to be passed to the strategy.

    Returns:
        A dictionary containing the strategy's results, typically including
        a 'portfolio' DataFrame and any strategy-specific 'metrics'.

    Raises:
        ValueError: If the strategy implementation cannot be found in the registry.
    """
    return StrategyRegistry.execute_strategy(strategy_name, data, params)