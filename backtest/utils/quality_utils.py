"""Utility functions and constants related to S&P Quality Ranks."""

import pandas as pd
from typing import Dict

# Maps S&P quality ranks (strings) to numerical values for easier comparison and filtering.
SP_QUALITY_RANK_MAPPING: Dict[str, int] = {
    'A+': 6, 'A': 5, 'A-': 4,
    'B+': 3, 'B': 2, 'B-': 1,
    # Add lower ranks if needed, mapping to 0 or negative values
    'C': 0, 'D': -1, 'NR': 0 # NR = Not Rated
}

# Minimum numerical S&P quality rank required by strategies like DGI and Value Play.
MIN_SP_QUALITY_NUMERIC: int = SP_QUALITY_RANK_MAPPING['B+']

def map_sp_quality_to_numeric(quality_series: pd.Series) -> pd.Series:
    """Maps S&P quality string ranks to numerical values.

    Args:
        quality_series: A pandas Series containing S&P quality ranks as strings.

    Returns:
        A pandas Series with numerical quality ranks. Unmapped values (including NaN)
        are filled with 0, representing a low quality score.
    """
    # Use .get() for robustness in case the column is missing entirely,
    # though the DataMapper should ensure it's present if required.
    # Map the values and fill any resulting NaNs (from unmapped strings or original NaNs) with 0.
    return quality_series.map(SP_QUALITY_RANK_MAPPING).fillna(0).astype(int)
