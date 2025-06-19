import pandas as pd
import numpy as np
from typing import Dict, List

from .constants import STRATEGY_COLUMNS


class DataMapper:
    # Maps API-specific field names to our canonical internal names
    # These internal names will then be mapped to STRATEGY_COLUMNS names
    API_FIELD_MAPPINGS: Dict[str, Dict[str, str]] = {
        "yfinance": {
            "adj_close": "share_price",  # yfinance 'adj_close' is a good candidate for 'share_price'
            "open": "open_price",
            "high": "high_price",
            "low": "low_price",
            "volume": "trade_volume",
            # yfinance doesn't directly provide things like sales_growth_5y, ps_ratio in basic download
        },
        "alphavantage": {
            # Alpha Vantage daily adjusted keys
            "1. open": "open_price",
            "2. high": "high_price",
            "3. low": "low_price",
            "4. close": "unadjusted_close_price",  # Might need to decide if this or adjusted is 'share_price'
            "5. adjusted close": "share_price",
            "6. volume": "trade_volume",
            # TODO: Alpha Vantage might have other endpoints for fundamental data
        }
        # Add mappings for other providers
    }

    def map_to_strategy_format(
            self,
            raw_data_df: pd.DataFrame,
            strategy_template_name: str,
            provider_name: str
    ) -> pd.DataFrame:
        if raw_data_df.empty:
            return pd.DataFrame()

        if strategy_template_name not in STRATEGY_COLUMNS:
            raise ValueError(f"Unknown strategy template: {strategy_template_name}")

        df = raw_data_df.copy()

        if provider_name == 'demo':
            # DemoFetcher output is already structured with 'year', 'symbol',
            # and strategy-specific columns named correctly.
            # We just need to ensure 'year' is int and all target columns are present.
            if 'year' in df.columns:
                df['year'] = df['year'].astype(int)
            else:
                # This should not happen if DemoFetcher works correctly
                raise ValueError("DemoFetcher output missing 'year' column.")
            if 'symbol' not in df.columns:
                raise ValueError("DemoFetcher output missing 'symbol' column.")

        elif provider_name in self.API_FIELD_MAPPINGS:
            provider_mapping = self.API_FIELD_MAPPINGS[provider_name]
            # 1. Rename API columns to canonical internal names
            rename_dict = {
                api_col: canonical_col
                for api_col, canonical_col in provider_mapping.items()
                if api_col in df.columns
            }
            df.rename(columns=rename_dict, inplace=True)

            if 'date' not in df.columns:
                raise ValueError(f"Raw data from API provider {provider_name} is missing 'date' column.")

            # 2. Ensure 'date' is string 'YYYY-MM-DD' or datetime.date before extracting year
            try:
                df['year'] = pd.to_datetime(df['date']).dt.year
            except Exception as e:
                raise ValueError(f"Could not parse 'date' column from {provider_name} to extract year: {e}")
            # TODO: The original 'date' column from API can be dropped if not needed in final output
            # or kept if it's part of STRATEGY_COLUMNS (unlikely for yearly data).
            # For now, we assume 'date' itself is not a target column for yearly data.

        else:
            raise ValueError(f"No field mappings or special handling defined for provider: {provider_name}")

        # 3. Calculate 'annual_return'
        # Might need calculation or be NaN.
        if 'annual_return' not in df.columns:
            # 'share_price' is our canonical name for the price used for returns
            if 'share_price' in df.columns:
                df = df.sort_values(by=['symbol', 'date'])
                # Calculate daily returns first, then compound to annual if needed, or simply use year-end prices.
                # For simplicity matching example_data.py, we'll aim for annual returns based on year-end prices.
                # This requires more sophisticated handling: group by symbol, then by year, pick last price of year.
                # For now, let's placeholder this and assume 'annual_return' might be directly mapped or NaN.
                # A more robust approach would be to calculate it from daily/monthly 'share_price'.
                # For now, we'll create the column and it will be NaN if not directly mapped.

                # Simplified annual return calculation (end of year to end of year)
                # This is a complex step if data is not already yearly.
                # For now, we'll assume 'share_price' is the daily adjusted close.
                # The backtest logic itself calculates returns based on portfolio composition and these prices.
                # The 'annual_return' column in example_data.py is more of a *result* for a stock for that year.
                # Let's make it NaN here, as it's usually calculated *by* the backtest or a more complex process.
                df['annual_return'] = np.nan  # TODO: derive from 'share_price'
            else:
                df['annual_return'] = np.nan
        else:  # Ensure it's float
            df['annual_return'] = pd.to_numeric(df['annual_return'], errors='coerce')

        # 4. Prepare final DataFrame based on STRATEGY_COLUMNS
        target_columns_spec = STRATEGY_COLUMNS[strategy_template_name]
        required_cols = target_columns_spec['required']
        optional_cols = target_columns_spec['optional']
        all_target_cols_ordered = required_cols + optional_cols  # This defines the desired order

        final_df_data = {}
        # Ensure 'year' and 'symbol' are handled first if they are part of the spec
        # (which they always should be for the target format)
        if 'year' not in df.columns:  # Should have been created by now
            raise InternalError("Year column was not created during mapping.")
        final_df_data['year'] = df['year']

        if 'symbol' not in df.columns:
            raise InternalError("Symbol column was not created/present during mapping.")
        final_df_data['symbol'] = df['symbol']

        # Process other columns defined in the strategy template
        for col_name in all_target_cols_ordered:
            if col_name in ['year', 'symbol']:  # Already handled
                continue
            if col_name in df.columns:
                final_df_data[col_name] = df[col_name]
            else:
                # If a target column is not in our mapped df, add it with NaNs
                final_df_data[col_name] = pd.Series([np.nan] * len(df), index=df.index, name=col_name)

        # Create DataFrame with columns in the desired order
        # The order of keys in final_df_data might not be guaranteed, so explicitly use all_target_cols_ordered
        # Filter all_target_cols_ordered to only include columns actually present in final_df_data
        # (which should be all of them, some possibly filled with NaN)

        # Reconstruct final_cols_in_order based on STRATEGY_COLUMNS spec
        final_cols_in_order = []
        if 'year' in all_target_cols_ordered: final_cols_in_order.append('year')
        if 'symbol' in all_target_cols_ordered: final_cols_in_order.append('symbol')
        for col in all_target_cols_ordered:
            if col not in final_cols_in_order:
                final_cols_in_order.append(col)

        final_df = pd.DataFrame(final_df_data)[final_cols_in_order]

        final_df = final_df.sort_values(by=['year', 'symbol']).reset_index(drop=True)

        return final_df
