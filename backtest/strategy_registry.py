"""Module for centralized strategy registration and management."""
from typing import Dict, Type, Callable, List, Optional, Any, TYPE_CHECKING
import pandas as pd

if TYPE_CHECKING:
    # This import is only for static type checkers and doesn't run at runtime,
    # thus avoiding the circular import.
    from backtest.strategies.base import Strategy


class StrategyRegistry:
    """Central registry for investment strategy implementations.

    This registry allows strategies to register themselves and provides
    methods to look up, instantiate, and execute strategies by name.
    """

    _strategies: Dict[str, Dict[str, Any]] = {}

    @classmethod
    def register(cls, name: str, strategy_class: Type['Strategy'], description: str = "") -> None: # Use string literal 'Strategy'
        """Register a strategy with the registry.

        Args:
            name: Unique name for the strategy
            strategy_class: The strategy class to register. Note: Type hint uses a string.
            description: Optional description of the strategy
        """
        cls._strategies[name] = {
            "class": strategy_class,
            "description": description or strategy_class.__doc__ or f"Strategy: {name}"
        }

    @classmethod
    def get_strategy_class(cls, name: str) -> Optional[Type['Strategy']]: # Use string literal 'Strategy'
        """Get the class for a registered strategy.

        Args:
            name: Name of the strategy to retrieve

        Returns:
            The strategy class if found, None otherwise
        """
        strategy_info = cls._strategies.get(name)
        return strategy_info["class"] if strategy_info else None

    @classmethod
    def create_strategy(cls, name: str) -> Optional['Strategy']: # Use string literal 'Strategy'
        """Create an instance of a registered strategy.

        Args:
            name: Name of the strategy to instantiate

        Returns:
            A new strategy instance if found, None otherwise
        """
        strategy_class = cls.get_strategy_class(name)
        return strategy_class() if strategy_class else None

    @classmethod
    def get_strategy_description(cls, name: str) -> Optional[str]:
        """Get the description of a registered strategy.

        Args:
            name: Name of the strategy

        Returns:
            Strategy description if found, None otherwise
        """
        strategy_info = cls._strategies.get(name)
        return strategy_info["description"] if strategy_info else None

    @classmethod
    def list_strategies(cls) -> List[Dict[str, str]]:
        """List all registered strategies.

        Returns:
            List of dictionaries with strategy information (name, description)
        """
        return [
            {"name": name, "description": info["description"]}
            for name, info in cls._strategies.items()
        ]

    @classmethod
    def is_strategy_registered(cls, name: str) -> bool:
        """Check if a strategy is registered.

        Args:
            name: Name of the strategy to check

        Returns:
            True if the strategy is registered, False otherwise
        """
        return name in cls._strategies

    @classmethod
    def execute_strategy(cls, name: str, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a registered strategy on the provided market data.

        Args:
            name: Name of the strategy to execute
            data: DataFrame containing market data
            params: Dictionary of strategy parameters

        Returns:
            Dictionary containing strategy results and metrics

        Raises:
            ValueError: If the strategy is not registered
        """
        strategy = cls.create_strategy(name)
        if not strategy:
            raise ValueError(f"Strategy implementation not found for '{name}'")

        return strategy.execute_strategy(data, params)