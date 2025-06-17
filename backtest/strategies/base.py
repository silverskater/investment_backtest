"""Defines the abstract base class for all investment strategies."""
import abc
from typing import Any, Dict

import pandas as pd


class Strategy(abc.ABC):
    """Abstract base class for all investment strategies.

    This class defines the common interface that every concrete strategy
    implementation must adhere to. Specifically, strategies must implement
    the `execute_strategy` method.
    """

    @abc.abstractmethod
    def execute_strategy(
        self, data: pd.DataFrame, params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Executes the strategy on the provided market data for a single period.

        This method must be implemented by all concrete strategy classes.

        Args:
            data: A pandas DataFrame containing market data relevant to the
                  current period of the backtest.
            params: A dictionary of parameters that can be used to configure
                    the strategy's behavior (e.g., thresholds, top_n).

        Returns:
            A dictionary containing the results of the strategy execution.
            Typically, this includes a 'portfolio' (DataFrame of selected
            stocks with their weights) and potentially 'metrics' specific
            to the strategy's internal calculations for that period.
        """
        raise NotImplementedError(
            "Each strategy must implement the 'execute_strategy' method."
        )