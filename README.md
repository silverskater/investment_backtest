# Investment Strategy Backtest Tool


A Python tool for backtesting specific investment strategies. [^1]

Beyond basic portfolio rebalancing, the tool can also apply risk management features to reduce exposure and improve performance.

## Disclaimer
**IMPORTANT**: This tool is for educational and research purposes only. It is not intended to provide investment advice. Past performance is not indicative of future results. The backtesting results should not be considered as financial advice or a recommendation to invest. Always consult with a qualified financial advisor before making investment decisions. The authors and contributors of this project assume no responsibility for any financial losses or damages resulting from the use of this tool.

## Requirements

- [Python](https://wiki.python.org/moin/BeginnersGuide/Download) 3.12+
- [Poetry](https://python-poetry.org/docs/#installation) (package manager)

## Installation

```bash
poetry install
```

## <a name="usage"></a>Usage

```bash
# Display help information and all available options
backtest --help
```

The `backtest` command has the following subcommands:

### `list`
Lists all available investment strategies.

```bash
# List all available investment strategies
backtest list
```

### `run <strategy> <data_file>`
Runs a backtest for the specified `strategy` using the provided `data_file` — see _[Market Data](#market-data)_

**Arguments:**
- `strategy`: The name of the strategy to run (e.g., `value_play`).
- `data_file`: Path to CSV or JSON file containing historical market data.

**Options for `run` subcommand:**

#### Basic Options
- `--start-year, -s`: Start year of the backtest period (default: last year - 10)
- `--end-year, -e`: End year of the backtest period (default: last year)
- `--growth-threshold, -g`: Minimum annual sales growth threshold (default: 0.2 or 20%)
- `--top-n, -n`: Number of top companies to include (default: 10)
- `--output, -o`: Output file for detailed results (JSON or CSV). The file will include the performance metrics plus the final portfolio composition.

#### Weighting Options
- `--hybrid-weighting, -hw`: Use hybrid weighting (50% market cap, 50% P/S ratio)
- `--benchmark, -b`: Benchmark symbol for comparison (default: SPY)

#### Risk Management Options
- `--dynamic-rebalance, -dr`: Enable dynamic rebalancing when weights deviate >5% from target
- `--risk-overlay, -ro`: Reduce exposure when portfolio's average P/S exceeds the threshold
- `--ps-threshold, -pt`: P/S ratio threshold for risk overlay (default: 10.0)
- `--rebalance-frequency, -rf`: Portfolio rebalancing frequency (annual, quarterly, monthly)

#### Realistic Testing Options
- `--transaction-cost, -tc`: Transaction cost as slippage percentage (default: 0.5%)
- `--include-delisted, -id`: Include delisted companies (survivorship bias adjustment)
- `--stress-test, -st`: Run specific stress test scenario (none, 2008crisis, 2022ratehike)


```bash
# Backtest
backtest run <strategy> <data_file> [OPTIONS...]:

# Basic usage with a specific strategy (dgi) and data file
backtest run dgi data/market_data.csv

# Specify custom parameters
backtest run exp_fund data/market_data.csv --start-year 2015 --end-year 2023 --growth-threshold 0.15 --top-n 15 --hybrid-weighting

# Backtest with risk management features
backtest run dgi data/market_data.csv --risk-overlay --ps-threshold 8.5 --dynamic-rebalance

# Backtest a stress test scenario
backtest run exp_fund data/market_data.csv --stress-test 2008crisis

# Use different rebalancing frequency
backtest run exp_fund data/market_data.csv --rebalance-frequency quarterly

# Include transaction costs and delisted companies
backtest run dgi data/market_data.csv --transaction-cost 0.005 --include-delisted

# Save detailed results to a JSON file
backtest run value_play data/market_data.csv --output results.json

# Save results to a CSV file
backtest run value_play data/market_data.csv --output results.csv
```

## <a name="market-data"></a>Market Data

The market data file should be in CSV or JSON format.

See `STRATEGY_COLUMNS` constant from _backtest/example_data.py_ for required and optional columns. 

### <a name="generate-example-data"></a>Example Data Generation

To simplify testing and demonstration, this tool includes a command to generate synthetic market data that matches the required format:

```bash
generate_example_data <strategy> [output_path] [OPTIONS]

# For the Dividend Growth Investing (DGI) strategy
generate_example_data dgi data/dgi.demo_content.csv
```

This command creates a CSV file with the specified strategy's data for simulated companies (labeled STOCK01-STOCK50).

E.g. `generate_example_data exp_fund` creates a file at `data/exp_fund.example_data.01.csv` with the following characteristics:
- Years: 2015–2023 (default)
- 50 simulated companies (labeled STOCK01-STOCK50)
- Realistic market cap values with proper ranking
- Growth rates between 10-40%
- P/S ratios that correlate with company size
- Annual returns that correlate with growth rates

Once generated, you can immediately use this data file with the backtest tool, see the _[Usage](#usage)_.


## Metrics

The backtest calculates and reports several key performance metrics:

- Sharpe Ratio: Risk-adjusted returns (target: >1.0)
- Maximum Drawdown: Worst historical loss (target: <25%)
- Outperformance vs. Benchmark: Comparison against specified benchmark (default: S&P 500)
- Portfolio Turnover: Tax/cost efficiency (target: <30%)
- Transaction Costs: Total costs from trading activities (as % of portfolio)
- Average Annual Return: Overall performance
- Portfolio Size: Number of stocks in the final portfolio

## Stress Test Scenarios

The tool includes predefined stress test scenarios to evaluate strategy robustness:

- **2008 Financial Crisis**: Simulates the 2008–2009 market conditions where value outperformed growth by 35% [^2]
- **2022 Rate Hike Cycle**: Simulates the 2022 interest rate increases where high-growth tech stocks (like ARK Innovation ETF) fell 67% compared to S&P 500's 19% decline [^3] [^4]

## Risk Management Features

### Dynamic Rebalancing

Instead of rigid annual rebalancing, this feature triggers portfolio adjustments only when individual positions deviate by more than 5% from their target weights. This approach can reduce unnecessary trading costs while still maintaining the desired asset allocation.

### Risk Overlay

This feature monitors the portfolio's average Price-to-Sales (P/S) ratio and reduces exposure when it exceeds a specified threshold (default: 10.0). High P/S ratios historically indicate overvaluation and increased downside risk. The risk overlay gradually increases cash allocation as the P/S ratio rises above the threshold.


## Technical Notes

### Performance Optimizations

The implementation is optimized for computational efficiency through the following techniques:

1. **Vectorized Operations**: Using NumPy's vectorized operations instead of loops for data transformations:
   - Calculating portfolio returns across multiple stocks simultaneously
   - Applying stress test scenarios to entire datasets at once
   - Generating large datasets of simulated market data efficiently

2. **Memory Efficiency**:
   - Creating arrays with pre-allocated sizes where possible
   - Using boolean masks for filtering instead of creating multiple intermediate dataframes
   - Optimizing data copying operations

3. **Calculation Improvements**:
   - Vectorized statistical calculations for metrics like the Sharpe ratio and maximum drawdown
   - Efficient handling of large datasets through NumPy's broadcasting capabilities
   - Using NumPy's statistical functions for portfolio risk analysis

These optimizations result in significantly faster execution times, especially when processing large historical datasets with many companies and multiple years of data.

For huge datasets (>1000 companies or >20 years), consider using a machine with at least 8GB RAM to ensure optimal performance.

## Code Style

This project follows the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html).

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Run tests (`pytest`)
4. Commit your changes (`git commit -m 'Add some amazing feature'`)
5. Push to the branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

---

## License

MIT


<div style="text-align: center">⁂</div>

 [^1]: https://www.investopedia.com/terms/b/backtesting.asp
 [^2]: https://www.investopedia.com/articles/economics/09/financial-crisis-review.asp
 [^3]: https://www.stlouisfed.org/on-the-economy/2023/jan/many-interest-rates-2022
 [^4]: https://www.forbes.com/advisor/investing/fed-funds-rate-history/