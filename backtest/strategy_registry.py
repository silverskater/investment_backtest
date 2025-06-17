"""Module for centralized strategy registration and management.

This module defines the `StrategyRegistry` class, which acts as a central
repository for all investment strategy implementations. It allows strategies
to be registered by name and provides methods to look up, instantiate,
and execute these strategies.
"""
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING, Type

import pandas as pd

if TYPE_CHECKING:
    # This import is only for static type checkers and doesn't run at runtime,
    # thus avoiding a potential circular import with backtest.strategies.base.
    from backtest.strategies.base import Strategy


class StrategyRegistry:
    """Central registry for investment strategy implementations.

    This class uses class methods to manage a dictionary of registered strategies.
    Strategies are stored with their class and description.

    Attributes:
        _strategies: A dictionary mapping strategy names to their class and
                     description.
    """

    _strategies: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def register(
        cls, name: str, strategy_class: Type['Strategy'], description: str = ""
    ) -> None:
        """Registers a strategy with the registry.

        The strategy's description is taken from the provided `description`
        argument, or falls back to the class's docstring, or a default
        description if neither is available.

        Args:
            name: The unique name for the strategy.
            strategy_class: The strategy class (subclass of `Strategy`) to register.
            description: An optional description of the strategy.
        """
        cls._strategies[name] = {
            "class": strategy_class,
            "description": description or strategy_class.__doc__ or f"Strategy: {name}"
        }

    @classmethod
    def get_strategy_class(cls, name: str) -> Optional[Type['Strategy']]:
        """Retrieves the class for a registered strategy.

        Args:
            name: The name of the strategy to retrieve.

        Returns:
            The strategy class if found, otherwise None.
        """
        strategy_info = cls._strategies.get(name)
        return strategy_info["class"] if strategy_info else None

    @classmethod
    def create_strategy(cls, name: str) -> Optional['Strategy']:
        """Creates an instance of a registered strategy.

        Args:
            name: The name of the strategy to instantiate.

        Returns:
            A new instance of the strategy if found, otherwise None.
        """
        strategy_class = cls.get_strategy_class(name)
        return strategy_class() if strategy_class else None

    @classmethod
    def get_strategy_description(cls, name: str) -> Optional[str]:
        """Gets the description of a registered strategy.

        Args:
            name: The name of the strategy.

        Returns:
            The strategy's description string if found, otherwise None.
        """
        strategy_info = cls._strategies.get(name)
        return strategy_info["description"] if strategy_info else None

    @classmethod
    def list_strategies(cls) -> List[Dict[str, str]]:
        """Lists all registered strategies.

        Returns:
            A list of dictionaries, where each dictionary contains the
            'name' and 'description' of a registered strategy.
        """
        return [
            {"name": name, "description": info["description"]}
            for name, info in cls._strategies.items()
        ]

    @classmethod
    def is_strategy_registered(cls, name: str) -> bool:
        """Checks if a strategy is registered.

        Args:
            name: The name of the strategy to check.

        Returns:
            True if the strategy is registered, False otherwise.
        """
        return name in cls._strategies

    @classmethod
    def execute_strategy(
        cls, name: str, data: pd.DataFrame, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Executes a registered strategy on the provided market data.

        Args:
            name: The name of the strategy to execute.
            data: A pandas DataFrame containing the market data for the period.
            params: A dictionary of parameters to be passed to the strategy.

        Returns:
            A dictionary containing the strategy's results, typically including
            a 'portfolio' DataFrame and any strategy-specific 'metrics'.

        Raises:
            ValueError: If the strategy is not registered.
        """
        strategy = cls.create_strategy(name)
        if not strategy:
            raise ValueError(f"Strategy implementation not found for '{name}'")
        return strategy.execute_strategy(data, params)