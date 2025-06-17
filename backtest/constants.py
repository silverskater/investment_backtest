"""
Shared constants for the Investment Strategy Backtest Tool.
"""

# Notional portfolio value for calculating monetary buys/sells from weights.
# This assumes the portfolio is notionally this size at each rebalance point
# for determining trade values.
NOTIONAL_PORTFOLIO_VALUE_FOR_TRADES: float = 1_000_000.0

# Default annual risk-free rate for calculations like Sharpe ratio.
ANNUAL_RISK_FREE_RATE: float = 0.02

# Default transaction cost rate for rebalancing.
DEFAULT_TRANSACTION_COST: float = 0.005

# Default deviation threshold for dynamic rebalancing.
DEFAULT_DEVIATION_THRESHOLD: float = 0.05

# Global DEBUG flag
# Set to True to enable more verbose error logging, like full tracebacks.
# Set to False for production or normal operation.
DEBUG: bool = False