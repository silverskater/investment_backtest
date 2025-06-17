"""Initializes the strategies package and registers available strategies.

This module is responsible for automatically discovering and registering all
strategy implementations found within the `backtest.strategies` package.
It scans for Python modules, imports them, and registers any classes
that are subclasses of `backtest.strategies.base.Strategy`.
"""
import os
import importlib
import inspect

from backtest.strategies.base import Strategy


def _register_strategies() -> None:
    """Automatically discovers and registers all strategy implementations.

    This function dynamically imports Python modules from the `strategies`
    package directory. It then inspects these modules for classes that
    inherit from the `Strategy` base class and registers them with the
    `StrategyRegistry`.
    """
    # Import the registry here to avoid circular imports at the module level.
    from backtest.strategy_registry import StrategyRegistry

    strategies_dir = os.path.dirname(os.path.abspath(__file__))
    strategy_files = [
        f for f in os.listdir(strategies_dir)
        if f.endswith('.py') and not f.startswith('__')
    ]

    for file_name in strategy_files:
        module_name = os.path.splitext(file_name)[0]
        try:
            module = importlib.import_module(f"backtest.strategies.{module_name}")
            for name, obj in inspect.getmembers(module):
                if (inspect.isclass(obj) and
                        issubclass(obj, Strategy) and
                        obj is not Strategy):
                    description = ""
                    if obj.__doc__:
                        # Use the first line of the docstring as the description.
                        description = obj.__doc__.split('\n')[0].strip()
                    StrategyRegistry.register(module_name, obj, description)
        except (ImportError, AttributeError) as e:
            print(f"Error importing strategy module {module_name}: {e}")


# Automatically register all strategies when the package is imported.
_register_strategies()