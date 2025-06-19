# Data Provider Package

This package is responsible for fetching and preparing market data for the investment strategy backtesting tool.

## Overview

The core functionality is exposed through the `backtest data` command-line tool (accessible via `backtest data --help`). This tool allows users to:

*   **Fetch data from various providers**: Run `backtest list` to see the available providers.
*   **Generate synthetic (demo) data**: Useful for testing and demonstration purposes without relying on external APIs or historical data availability.
*   **Map data to strategy-specific formats**: Ensures the fetched or generated data has the correct columns and structure required by different investment strategy templates (e.g., `exp_fund`, `dgi`, `value_play`).

## Key Components

*   **Fetchers**: Classes that implement the `FetcherInterface` to retrieve data from specific sources (e.g., `YFinanceFetcher`, `DemoFetcher`).
*   **DataMapper**: A class responsible for transforming raw data from fetchers into the standardized format required by strategy templates defined in `STRATEGY_COLUMNS`.
*   **CLI (`cli.py`)**: Defines the `data` command, its arguments, and options for controlling data fetching and generation.
*   **Constants (`constants.py`)**: Contains shared constants, including `STRATEGY_COLUMNS` which defines the data schema for each strategy.

## <a name="usage"></a>Basic Usage

To generate or fetch data, use the `backtest data` command.

**Example: Fetch data for Apple (AAPL) from Yahoo Finance for the 'dgi' strategy template**


```bash
backtest data <strategy> --output <OUTPUT_FILE_PATH> [OPTIONS]
```

**Examples for `backtest data`:**

```bash
# Help
backtest data --help

# Generate data for backtesting the Dividend Growth Investing (DGI) strategy using the default data fetcher
backtest data dgi --output data/dgi.market_data.csv

# Generate demo data for the value investing strategy
backtest data value_play --output data/value_play.demo_market_data.csv --fetcher demo --demo-start-year 2015 --demo-end-year 2023
```

The structure of the data file depends on the `STRATEGY` template chosen during data generation. The `STRATEGY_COLUMNS` constant in `/data_provider/constants.py` defines the required and optional columns for each strategy template.
The `backtest data` command will fetch data using the specified provider (e.g., `yfinance`) and then map it to the column structure required by the chosen `STRATEGY` template. If using the `demo` fetcher, it generates synthetic data already conforming to the strategy template.

**Arguments:**
- `strategy`: The name of the strategy template for which to format the data (e.g., `exp_fund`, `dgi`, `value_play`).

**Key Options for `backtest data`:**
-   `--fetcher [demo|yfinance|alphavantage]`: Specifies the data source.
    -   `demo`: Generates synthetic data.
    -   `yfinance`: Fetches data from Yahoo Finance.
    -   `alphavantage`: Fetches data from Alpha Vantage (requires API key).
-   `--output, -o`: Path to save the fetched/generated data (e.g., `data/market_data.csv`). **Required.**

Run `backtest data --help` for a list of available options.