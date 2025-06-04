"""Package containing strategy implementations for the backtest tool."""
import os
import importlib
import inspect

from backtest.strategies.base import Strategy


def _register_strategies() -> None:
    """Automatically discover and register all strategy implementations.

    This function dynamically imports all Python modules in the strategies package
    and registers any Strategy subclasses they contain with the StrategyRegistry.
    """
    # Import the registry here to avoid circular imports
    from backtest.strategy_registry import StrategyRegistry

    # Get the strategies directory path
    strategies_dir = os.path.dirname(os.path.abspath(__file__))

    # Find all potential strategy modules (Python files in the strategies directory)
    strategy_files = [f for f in os.listdir(strategies_dir)
                      if f.endswith('.py') and not f.startswith('__')]

    # Import each module and register its Strategy classes
    for file_name in strategy_files:
        # Get the module name (file name without .py extension)
        module_name = os.path.splitext(file_name)[0]

        try:
            # Import the module
            module = importlib.import_module(f"backtest.strategies.{module_name}")

            # Find all Strategy subclasses in the module
            for name, obj in inspect.getmembers(module):
                # Check if it's a class, is a Strategy subclass, and is not Strategy itself
                if (inspect.isclass(obj) and 
                        issubclass(obj, Strategy) and 
                        obj is not Strategy):
                    # Extract description from class docstring
                    description = ""
                    if obj.__doc__:
                        description = obj.__doc__.split('\n')[0].strip()

                    # Register the strategy with its module name
                    StrategyRegistry.register(module_name, obj, description)
        except (ImportError, AttributeError) as e:
            # Log the error but continue with other modules
            print(f"Error importing strategy module {module_name}: {e}")


# Automatically register all strategies when the package is imported
_register_strategies()