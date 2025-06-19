from abc import ABC, abstractmethod
import pandas as pd
from typing import List, Dict, Any, Optional

class FetcherInterface(ABC):
    """Abstract Base Class for data fetchers."""

    @abstractmethod
    def __init__(self, api_key: Optional[str] = None, **kwargs):
        """
        Initialize the fetcher.
        Args:
            api_key: Optional API key if required by the provider.
            **kwargs: Additional provider-specific configurations.
        """
        self.api_key = api_key

    @abstractmethod
    def fetch_data(
        self,
        symbols: List[str],
        start_date: str,  # Format: YYYY-MM-DD
        end_date: str,    # Format: YYYY-MM-DD
        # interval: str = '1d' # Optional: '1wk', '1mo'
    ) -> pd.DataFrame:
        """
        Fetches historical market data.

        Returns:
            pd.DataFrame: A DataFrame in a "long" format with columns like
                          ['date', 'symbol', 'open', 'high', 'low', 'close', 'volume', ... (other API specific fields)].
                          'date' should be a datetime.date object or string 'YYYY-MM-DD'.
        """
        pass

    @abstractmethod
    def get_supported_fields(self) -> List[str]:
        """
        Returns a list of raw field names this fetcher can provide (e.g., 'Adj Close', 'Volume' for yfinance).
        """
        pass