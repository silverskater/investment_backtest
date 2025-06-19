from typing import Dict, List, Union

# Global DEBUG flag.
DEBUG: bool = False

# Defines the required and optional data columns for each investment strategy template.
# This structure is used by the DataMapper to ensure that fetched/generated data
# conforms to the expectations of a given strategy.
STRATEGY_COLUMNS: Dict[str, Dict[str, List[str]]] = {
    'dgi': {  # Dividend Growth Investing
        # Core columns for the 'Dividend Growth Investing' strategy,
        # emphasizing dividend sustainability, growth, and company quality.
        "required": [
            "year",                     # Year of the data
            "symbol",                   # Stock symbol
            "share_price",              # Share price
            "div_growth_streak",        # Dividend growth streak in years
            "payout_ratio",             # Dividend Payout Ratio
            "eps_cagr_3y",              # 3-year EPS Compound Annual Growth Rate
            "roe",                      # Return on Equity
            "debt_equity",              # Debt-to-Equity Ratio
            "industry_debt_equity",     # Industry average Debt-to-Equity Ratio
            "sp_quality",               # S&P Quality Rating or similar financial strength indicator
            "dividend_yield",           # Dividend Yield
            "div_growth_5y",            # 5-year average dividend growth rate
            "quality_score",            # A composite score reflecting overall quality
        ],
        # Additional data points that can be used for DGI strategy refinement.
        "optional": [
            "market_cap",               # Market capitalization
            "market_cap_rank",          # Rank based on market capitalization
            "sales_growth_5y",          # 5-year average sales growth rate
            "annual_return",            # Annual return of the stock (for performance tracking)
            "ps_ratio",                 # Price-to-Sales ratio
            "shares_outstanding",       # Number of shares outstanding
        ],
    },
    "exp_fund": {
        # Columns essential for the 'Experimental Fund' strategy,
        # typically focusing on growth, market perception, and basic financial health.
        "required": [
            "year",                     # Year of the data
            "symbol",                   # Stock symbol
            "market_cap",               # Market capitalization
            "market_cap_rank",          # Rank based on market capitalization
            "sales_growth_5y",          # 5-year average sales growth rate
            "annual_return",            # Annual return of the stock
            "share_price",              # Share price at the time of data
        ],
        # Optional columns that can provide additional context or refinement
        # for the 'Experimental Fund' strategy.
        "optional": [
            "ps_ratio",                 # Price-to-Sales ratio
            "shares_outstanding",       # Number of shares outstanding
        ],
    },
    "value_play": {
        # Comprehensive list of required columns for the 'Value Play' strategy,
        # covering various aspects of value assessment, financial health, and quality.
        "required": [
            # Basic identification and performance tracking
            "year",                     # Year of the data
            "symbol",                   # Stock symbol
            "share_price",              # Share price
            "annual_return",            # Annual return of the stock

            # Undervaluation Metrics (often compared to sector/industry medians)
            "pe_ratio",                 # Price-to-Earnings ratio
            "sector_median_pe",         # Sector median Price-to-Earnings ratio
            "pb_ratio",                 # Price-to-Book ratio
            "sector_median_pb",         # Sector median Price-to-Book ratio

            # Cash Flow and Asset Value
            "fcf_yield",                # Free Cash Flow Yield
            "sector_median_fcf_yield",  # Sector median Free Cash Flow Yield
            "tangible_book_value_per_share", # Tangible Book Value per Share

            # Financial Health & Solvency (often compared to industry averages)
            "debt_equity",              # Debt-to-Equity Ratio
            "industry_avg_debt_equity", # Industry average Debt-to-Equity Ratio
            "total_debt",               # Total Debt
            "net_current_asset_value",  # Net Current Asset Value (NCAV)

            # Profitability and Stability
            "current_ratio",            # Current Ratio (liquidity)
            "positive_ni_5y_streak",    # Streak of positive Net Income over 5 years
            "roe",                      # Return on Equity

            # Margin of Safety Indicators
            "margin_of_safety",         # Calculated Margin of Safety
            "total_book_value",         # Total Book Value

            # Quality Indicators
            "sp_quality",               # S&P Quality Rating or similar financial strength indicator
            "eps_growth_5y",            # 5-year Earnings Per Share growth rate
            "significant_insider_activity", # Flag for notable insider trading activity

            # Composite Scores (for ranking and weighting within the strategy)
            "composite_value_score",    # A custom score reflecting overall value
            "quality_score",            # A custom score reflecting overall quality

            # Sector Information (crucial for sector-relative comparisons)
            "sector"                    # Industry sector of the company
        ],
        # Optional columns that can provide further context or refinement for value analysis.
        "optional": [
            "market_cap",               # Market capitalization
            "market_cap_rank",          # Rank based on market capitalization
            "ps_ratio",                 # Price-to-Sales ratio
            "shares_outstanding",       # Number of shares outstanding
            # "is_delisted"             # Example: Flag if company was delisted (if relevant)
            # "dividend_yield"          # Example: Dividend yield (can be a value signal)
        ],
    }
}

# Define default parameters for demo data generation.
DEFAULT_DEMO_START_YEAR_OFFSET = 10  # Default to 10 years before the end year
DEFAULT_DEMO_NUM_COMPANIES = 50
DEFAULT_MARKET_CAP_MIN = 1_000_000_000  # 1 Billion
DEFAULT_MARKET_CAP_MAX = 2_000_000_000_000  # 2 Trillion
DEFAULT_SHARE_PRICE_MIN = 10.0
DEFAULT_SHARE_PRICE_MAX = 500.0
DEFAULT_ANNUAL_RETURN_MIN = -0.50  # -50%
DEFAULT_ANNUAL_RETURN_MAX = 0.80  # +80%
NUM_IDEAL_DGI_STOCKS_PER_YEAR = 3
NUM_IDEAL_VALUE_STOCKS_PER_YEAR = 4
# Default symbols to use for demo fetcher if num_companies is not specified
# or if the 'symbols' argument to fetch_data is empty.
# This is more of a conceptual placeholder as num_companies will be used directly.
DEFAULT_DEMO_SYMBOLS_COUNT = DEFAULT_DEMO_NUM_COMPANIES

AVAILABLE_FETCHERS: List[str] = [
    "demo",
    "yfinance",
    #"alphavantage",
    #"ibkr",
]
# Default data fetcher for the `data` command.
DEFAULT_DATA_FETCHER: str = "demo"

# Sanity check: ensure the default fetcher is available
if DEFAULT_DATA_FETCHER not in AVAILABLE_FETCHERS:
    raise ValueError(
        f"DEFAULT_DATA_FETCHER '{DEFAULT_DATA_FETCHER}' is not in AVAILABLE_FETCHERS list."
    )