"""Module for managing different investment strategy implementations.

This module provides a thin facade that delegates to the StrategyRegistry,
decoupling strategy management from strategy implementation.
"""
from typing import Dict, List, Optional, Any, Callable

import pandas as pd

# Ensure strategies are registered when this manager is imported.
# This will execute backtest/strategies/__init__.py which should
# contain the logic to scan and register all available strategies.
import backtest.strategies

from backtest.strategy_registry import StrategyRegistry


def is_strategy_available(strategy_name: str) -> bool:
    """Check if a strategy module is available.

    Args:
        strategy_name: Name of the strategy

    Returns:
        True if the strategy module exists, False otherwise
    """
    return StrategyRegistry.is_strategy_registered(strategy_name)


def list_available_strategies() -> List[Dict[str, str]]:
    """List all available investment strategies.

    Returns:
        List of dictionaries with strategy information (name, description)
    """
    return StrategyRegistry.list_strategies()


def validate_strategy(strategy_name: str) -> bool:
    """Validate that a strategy exists and is valid.

    Args:
        strategy_name: Name of the strategy to validate

    Returns:
        True if the strategy exists and is valid, False otherwise
    """
    return StrategyRegistry.is_strategy_registered(strategy_name)


def get_strategy_description(strategy_name: str) -> Optional[str]:
    """Get the description of a strategy.

    Args:
        strategy_name: Name of the strategy

    Returns:
        Strategy description if found, None otherwise
    """
    return StrategyRegistry.get_strategy_description(strategy_name)


def get_strategy_implementation(strategy_name: str) -> Optional[Callable]:
    """Get the implementation function for a strategy.

    This function instantiates the strategy class and returns its
    execute_strategy method.

    Args:
        strategy_name: Name of the strategy

    Returns:
        Strategy execute_strategy method if found, None otherwise
    """
    strategy = StrategyRegistry.create_strategy(strategy_name)
    return strategy.execute_strategy if strategy else None


def execute_strategy(strategy_name: str, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a specific investment strategy on the provided market data.

    Args:
        strategy_name: Name of the strategy to execute
        data: DataFrame containing market data
        params: Dictionary of strategy parameters

    Returns:
        Dictionary containing strategy results and metrics

    Raises:
        ValueError: If the strategy implementation cannot be found
    """
    return StrategyRegistry.execute_strategy(strategy_name, data, params)
