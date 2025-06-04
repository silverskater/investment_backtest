# Investment Strategy: The Exponential Fund

The strategy called ["The Exponential Fund"](https://x.com/Hutton_Richard/status/1741601623988650410) focuses on high-growth companies with market cap weighting.

## Core Strategy Overview

1. Invest in the top 10 largest companies that have grown their sales by at least 20%/year for the last 5 years
2. The weight of each stock is proportional to its market cap relative to the total sum of their market caps
3. Re-balance yearly in April when all the annual reports are out

---

## **Potential Weak Spots**

This investment strategy, while systematic, has several potential weaknesses rooted in market dynamics, growth stock behavior, and portfolio construction. Historical backtesting can reveal vulnerabilities using specific methodologies and metrics.

This systematic growth strategy targets companies demonstrating exceptional revenue expansion while addressing inherent biases through enhanced risk controls. The approach combines market-cap weighting with fundamental filters to balance momentum and valuation concerns.


### **1. Market Cap Weighting Risks**

- **Momentum bias**: Market cap-weighted portfolios inherently overweight stocks that have recently outperformed, creating exposure to overvalued companies. This can lead to significant drawdowns if growth expectations reverse.
- **Concentration risk**: The top 10 companies may dominate the portfolio (e.g., tech giants in the 2020s), amplifying sector-specific risks.

### **2. Growth Stock Vulnerabilities**

- **High valuations**: Companies with 20%+ sales growth often trade at elevated P/E ratios, making them sensitive to earnings disappointments. For example, a stock with a P/E of 60 could collapse 50% if growth slows.
- **Volatility**: Growth stocks tend to have 30–50% higher volatility than value stocks, increasing portfolio swings during market downturns.
- **Survivorship bias**: The strategy excludes companies that failed to maintain 20% growth, potentially overstating historical returns.

### **3. Rebalancing Challenges**

- **Annual lag**: Yearly rebalancing in April may miss mid-year earnings surprises or macroeconomic shifts (e.g., rate hikes in March 2023).
- **Tax inefficiency**: Rebalancing triggers capital gains taxes when selling winners, reducing net returns by ~1–2% annually in taxable accounts.

### **4. Macroeconomic Sensitivity**

- **Interest rate risk**: Growth stocks underperform when rates rise. A 1% rate increase historically correlates with 15–20% declines in high-growth sectors.
- **Economic cycles**: The strategy may struggle in recessions when investors favor value stocks with stable cash flows.

---

## Implementation Improvements

Based on identified vulnerabilities in the original strategy, the following improvements have been implemented:

### 1. Enhanced Weighting Options

- **Hybrid Weighting**: Combines 50% market cap with 50% fundamental factors (using the P/S ratio) to reduce momentum bias
- **Custom Top-N Selection**: Configurable number of companies to include (default: 10)
- **Flexible Growth Threshold**: Adjustable minimum sales growth threshold (default: 20%)

### 2. Dynamic Risk Management

- **Risk Overlay**: Reduces exposure when the portfolio's average P/S exceeds a configurable threshold (default: 10.0)
- **Dynamic Rebalancing**: Triggers adjustments when individual weights deviate >5% from target
- **Flexible Rebalancing Frequency**: Support for annual, quarterly, or monthly rebalancing

### 3. Realistic Backtesting

- **Transaction Cost Modeling**: Accounts for slippage costs (default: 0.5% per trade)
- **Survivorship Bias Adjustment**: Option to include delisted companies in the analysis
- **Stress Test Scenarios**: Predefined scenarios to evaluate strategy performance:
  - 2008 Financial Crisis: Value outperformed growth by 35%
  - 2022 Rate Hike Cycle: High-growth tech stocks fell significantly

---

## **Historical Testing Methodology**

Historical backtesting can reveal vulnerabilities using specific methodologies and metrics.

### Backtest Design

```python
# Enhanced backtest framework with optimized NumPy vectorization
import pandas as pd
import numpy as np
from datetime import datetime

def backtest_strategy(data, start_year=None, end_year=None, growth_threshold=0.2, top_n=10,
                     hybrid_weighting=False, dynamic_rebalance=False, risk_overlay=False,
                     ps_threshold=10.0, transaction_cost=0.005, rebalance_frequency='annual',
                     include_delisted=False, stress_test='none'):
    # Set default years if not provided
    if end_year is None:
        end_year = datetime.now().year - 1  # Last year
    if start_year is None:
        start_year = end_year - 10  # Last year - 10

    # Apply stress test if requested
    if stress_test != 'none':
        data = apply_stress_test(data, stress_test, start_year, end_year)

    # Initialize portfolio tracking
    portfolio_history = []
    yearly_returns = []

    # Run the backtest for each year
    for year in range(start_year, end_year + 1):
        # Filter data for current year
        year_data = data[data['year'] == year]

        # Filter companies with required sales growth
        growth_stocks = year_data[
            (year_data['sales_growth_5y'] >= growth_threshold) & 
            (year_data['market_cap_rank'] <= top_n)
        ]

        # Apply weighting strategy
        if hybrid_weighting:
            # 50% market cap, 50% P/S-based weighting
            apply_hybrid_weighting(growth_stocks)
        else:
            # Standard market cap weighting
            apply_market_cap_weighting(growth_stocks)

        # Apply risk overlay if enabled
        if risk_overlay:
            apply_risk_overlay(growth_stocks, ps_threshold)

        # Determine if rebalancing is needed
        should_rebalance = determine_rebalance_timing(
            year_data, rebalance_frequency)

        # Rebalance portfolio if needed
        if should_rebalance:
            portfolio_history = rebalance(
                portfolio_history, growth_stocks, 
                dynamic=dynamic_rebalance, 
                transaction_cost=transaction_cost
            )

        # Calculate and store yearly returns
        if 'annual_return' in growth_stocks.columns:
            portfolio_return = (growth_stocks['annual_return'] * growth_stocks['weight']).sum()
            yearly_returns.append({'year': year, 'return': portfolio_return})

    # Calculate final performance metrics including total return
    metrics = calculate_metrics(portfolio_history)

    # Export total return over the entire period
    print(f"Total Return Over Period: {metrics['total_return']:.2f}%")

    return metrics
```

### Key Metrics Evaluation

| Metric                         | Target         | Implementation                                          |
|:-------------------------------|:---------------|:--------------------------------------------------------|
| Sharpe ratio                   | >1.0           | Calculates excess returns divided by standard deviation |
| Maximum drawdown               | <25%           | Tracks worst peak-to-trough decline                     |
| Growth vs. Benchmark           | Outperformance | Compares returns against specified benchmark            |
| Portfolio Turnover Ratio (PTR) | <30%           | Monitors portfolio changes and associated costs         |
| Total Return                   | Maximized      | Calculates cumulative return over the entire period     |
| Transaction costs              | Minimized      | Accounts for trading costs in total returns             |

### Critical Tests Implementation

1. **Stress Scenarios**:
   - 2008 Financial Crisis: Applies return adjustments to simulate value outperforming growth
   - 2022 Rate Hike Cycle: Simulates differential impact on stocks based on growth rates

2. **Survivorship Bias Adjustment**: 
   - Includes an option to retain delisted companies in the analysis

3. **Parameter Sensitivity**: 
   - Configurable growth thresholds and rebalancing frequencies

4. **Transaction Cost Modeling**: 
   - Accounts for slippage in each rebalance operation

---

## **Recommended Improvements**

1. **Hybrid weighting**: Combine 50% market cap with 50% fundamental factors (e.g., P/S ratio) to reduce momentum bias.
2. **Dynamic rebalancing**: Trigger adjustments when individual weights deviate >5% from the target.
3. **Risk overlay**: Reduce exposure when the portfolio's average P/S exceeds 10 (historically a danger zone for growth stocks).

This implementation addresses the key vulnerabilities of the original strategy while maintaining its core focus on high-growth companies. The enhanced risk management features help protect against market downturns, while realistic backtesting provides more accurate performance expectations.
