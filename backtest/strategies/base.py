import abc
from typing import Dict, Any
import pandas as pd


class Strategy(abc.ABC):
    """
    Abstract base class for all investment strategies.

    This class defines the common interface that every strategy implementation
    must provide.
    """

    @abc.abstractmethod
    def execute_strategy(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the strategy on the provided market data.

        This method must be implemented by all concrete strategy classes.

        Args:
            data: DataFrame containing market data.
            params: Dictionary of strategy parameters.

        Returns:
            A dictionary containing strategy results and calculated metrics.
        """
        raise NotImplementedError("Each strategy must implement the 'execute_strategy' method.")
