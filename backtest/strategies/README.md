# Strategy Implementations

This package contains the individual implementations of investment strategies used in the backtest tool.

Individual strategies are documented under _./[strategy].md_.

## Adding a New Strategy

To add a new strategy:

1. Create a new Python module in this package named `[strategy].py`
2. Implement a class that inherits from the `Strategy` abstract base class:

```python
from .base import Strategy

class NewStrategy(Strategy):
    """New strategy description."""

   def execute_strategy(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the strategy on the provided market data."""
        # Your implementation here
        return results
```

## Strategy Module Structure

Each strategy module should follow this structure:

1. Import necessary dependencies including the base `Strategy` class
2. Implement a strategy class that inherits from `Strategy` and implements the `execute_strategy` method
3. Implement any helper methods needed for the strategy
4. The `execute_strategy` method should return a dictionary with results and metrics
5. Include a meaningful docstring for your strategy class to provide a description

## Automatic Registration

All strategy implementations are automatically discovered and registered when the `backtest.strategies` package is imported. You don't need to manually register your strategy or add a factory function.

The strategy registration process:
1. Scans all Python modules in the `backtest.strategies` package
2. Identifies classes that inherit from the `Strategy` base class
3. Registers each strategy with the central `StrategyRegistry` using the module name as the strategy name
4. Extracts the strategy description from the class docstring

## Example

See the `exp_fund.py` module for a complete example implementation.
