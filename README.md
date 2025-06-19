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

The project provides the `backtest` command-line interface.
You can get help for the main command and its subcommands:

```bash
# Display help information and all available options
backtest --help
# Basic example - generate some data and backtest the Dividend Growth Investing (DGI) strategy:
backtest data dgi --output ./data/dgi.market_data.csv
backtest run dgi --input ./data/dgi.market_data.csv
```

The `backtest` command has the following subcommands:

### `list`
Lists all available investment strategies.

```bash
backtest list
```

### `run <strategy>`
Runs a backtest for the specified `strategy` using a pre-generated market data file — see the [data](#market-data) sub-command

```bash
# Backtest
backtest run <strategy> --input <INPUT_FILE_PATH> [OPTIONS...]
```

**Arguments:**
- `strategy`: The name of the strategy to run (e.g., `value_play`).

```bash
# Help
backtest run --help

# Basic usage with a specific strategy (dgi) and data file
backtest run dgi --input data/market_data.csv

# Specify custom parameters
backtest run exp_fund --input data/market_data.csv --start-year 2015 --end-year 2023 --growth-threshold 0.15 --top-n 15 --hybrid-weighting

# Backtest with risk management features
backtest run dgi --input data/market_data.csv --risk-overlay --ps-threshold 8.5 --dynamic-rebalance

# Backtest a stress test scenario
backtest run exp_fund --input data/market_data.csv --stress-test 2008crisis

# Use different rebalancing frequency
backtest run exp_fund --input data/market_data.csv --rebalance-frequency quarterly

# Include transaction costs and delisted companies
backtest run dgi --input data/market_data.csv --transaction-cost 0.005 --include-delisted

# Save detailed results to a JSON file
backtest run value_play --input data/market_data.csv --output results.json

# Save results to a CSV file
backtest run value_play --input data/market_data.csv --output results.csv
```


### <a name="market-data"></a>`data <strategy_template_name>`
Fetches or generates market data suitable for the backtesting tool. It is the primary way to create the input data files used by `backtest run`.

```bash
backtest data <strategy> --output <OUTPUT_FILE_PATH> [OPTIONS]
```

**Examples for `backtest data`:**

```bash
# Help
backtest data --help

# For the Dividend Growth Investing (DGI) strategy
backtest data dgi --output data/dgi.demo_content.csv
```

See [./data_provider/README.md](./data_provider/README.md#usage) for more.

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