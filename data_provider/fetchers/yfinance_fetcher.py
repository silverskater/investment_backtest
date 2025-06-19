import pandas as pd
from typing import List, Optional
from ..base_fetcher import FetcherInterface

try:
    import yfinance as yf
except ImportError:
    yf = None  # Will be checked in __init__


class YFinanceFetcher(FetcherInterface):
    def __init__(self, api_key: Optional[str] = None, **kwargs):
        super().__init__(api_key, **kwargs)
        if yf is None:
            raise ImportError(
                "yfinance library is not installed. "
                "Please install it with: poetry install --extras provider_yfinance"
            )

    def fetch_data(self, symbols: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        try:
            # yf.download can return a DataFrame with MultiIndex columns
            data_yf = yf.download(symbols, start=start_date, end=end_date, progress=False)
            if data_yf.empty:
                return pd.DataFrame()

            # Reshape to long format: Date, Symbol, Open, High, Low, Close, Adj Close, Volume
            # If only one symbol, stack() might behave differently or not be needed.
            if isinstance(data_yf.columns, pd.MultiIndex):
                data_long = data_yf.stack(level=1).reset_index()
            else:  # Single symbol case, yf.download might not return MultiIndex columns
                data_long = data_yf.reset_index()
                # Add symbol column if it's missing (for single symbol download)
                if 'symbol' not in data_long.columns and len(symbols) == 1:
                    data_long['symbol'] = symbols[0]

            data_long = data_long.rename(columns={
                'Date': 'date',
                'level_1': 'symbol',  # From stacking
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Adj Close': 'adj_close',
                'Volume': 'volume'
            })

            # Ensure 'symbol' column exists if not created by stack (e.g. single symbol)
            if 'symbol' not in data_long.columns and len(symbols) == 1:
                data_long['symbol'] = symbols[0]

            # Ensure date is just date, not datetime, and select relevant columns
            if 'date' in data_long.columns:
                data_long['date'] = pd.to_datetime(data_long['date']).dt.date

            # Select only columns that are typically present and renamed
            # This helps standardize before mapping
            cols_to_keep = ['date', 'symbol', 'open', 'high', 'low', 'close', 'adj_close', 'volume']
            existing_cols = [col for col in cols_to_keep if col in data_long.columns]

            return data_long[existing_cols]

        except Exception as e:
            print(f"Error fetching data from yfinance for {symbols}: {e}")
            return pd.DataFrame()  # Return empty DataFrame on error

    def get_supported_fields(self) -> List[str]:
        return ['open', 'high', 'low', 'close', 'adj_close', 'volume']