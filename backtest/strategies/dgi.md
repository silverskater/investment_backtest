# Investment Strategy: Dividend Growth Investing (DGI)

This document outlines a systematic approach to dividend growth investing, focusing on companies with sustained dividend increases, financial stability, and long-term compounding potential. [^1] [^2]

---

## **Core Strategy Overview**

### **1. Stock Selection Criteria**

- **Dividend growth history**: Target companies with ≥10 consecutive years of dividend growth (25+ years for "Dividend Aristocrats") [^7] [^11] [^13]
- **Payout ratio**: ≤60% for most sectors (≤80% for regulated utilities/REITs) [^7] [^16]
- **Earnings growth**: ≥5% 3-year EPS CAGR to support future dividend growth [^7] [^9]
- **Quality metrics**:
    - Return on equity (ROE) ≥15% [^7]
    - Debt-to-equity ratio ≤ industry average [^7]
    - S\&P Quality Rank ≥B+ [^13]


### **2. Portfolio Construction**

- **Sector diversification**: Limit sector exposure to 25% with emphasis on consumer staples (25%), healthcare (20%), and industrials (15%) [^11] [^16]
- **Weighting method**: Hybrid approach combining:
    - 50% dividend yield weighting
    - 30% dividend growth rate (5-year CAGR)
    - 20% quality score (ROE + earnings stability) [^10] [^17]
- **Rebalancing**: Annual review post-dividend declaration season (Q1) with a 5% threshold for weight deviations [^9]


### **3. Income Reinvestment**

- **DRIP automation**: Automatic dividend reinvestment during accumulation phase [^5]
- **Yield-on-cost optimization**: Manual reinvestment into the highest conviction holdings during distribution phase [^15]

---

## **Potential Weak Spots**

### **1. Growth Limitations**

- **Reinvestment constraints**: Median DGI portfolio retains only 35% of earnings vs 70%+ for growth stocks, limiting R\&D spending [^6] [^10]
- **Tech underrepresentation**: Only 12% of Dividend Aristocrats are in tech vs 28% of S\&P 500 [^11]


### **2. Macroeconomic Vulnerabilities**

- **Interest rate sensitivity**: 1% rate hike correlates with 7-9% DGI underperformance vs value stocks [^6] [^10]
- **Inflation mismatch**: 43% of DGI stocks have pricing power lagging CPI by ≥1.5% [^17]


### **3. Sustainability Risks**

- **Dividend trap potential**: 18% of companies with 10+ year streaks cut dividends within 5 years of reaching 70% payout ratio [^6] [^13]
- **Sector shocks**: 65% of financial sector Dividend Aristocrats froze dividends during 2008 crisis [^11]


### **4. Behavioral Challenges**

- **Yield chasing**: Portfolios with starting yield >4% underperform by 1.8% annualized [^9]
- **Overdiversification**: Optimal DGI portfolio size is 25–35 stocks; >40 stocks reduce alpha by 0.7%/year [^17]

---

## **Implementation Improvements**

### **1. Enhanced Screening Framework**

```python
def dgi_screener(df):
    return df[
        (df['div_growth_streak'] >= 10) &
        (df['payout_ratio'] <= 0.6) &
        (df['eps_cagr_3y'] >= 0.05) &
        (df['roe'] >= 0.15) &
        (df['debt_equity'] <= df['industry_debt_equity']) &
        (df['sp_quality'] >= 'B+')
    ].sort_values('composite_score', ascending=False)
```


### **2. Dynamic Risk Management**

- **Interest rate hedge**: Reduce duration exposure when 10Y Treasury yield > dividend yield spread +1.5% [^10]
- **Payout ratio alerts**: Auto-trim positions when the payout ratio exceeds the 5-year average +1σ [^13]
- **Sector rotation**: Overweight healthcare/utilities when PMI <50; tech/industrials when PMI >55 [^16]


### **3. Backtesting Protocol**

**Key metrics:**


| Metric                 | Target     | Calculation Method                   |
|:-----------------------|:-----------|:-------------------------------------|
| Yield-on-cost (10Y)    | ≥5%        | (Annual Dividend / Original Price)   |
| Dividend Survival Rate | ≥95%       | % of positions maintaining dividends |
| Total Return vs S\&P   | +1.5% CAGR | With dividends reinvested            |
| Maximum Drawdown       | <20%       | Peak-to-trough decline               |

**Stress test scenarios:**

1. 2000-2003 Dot-com Crash (Value stocks -42% vs DGI -28%) [^9]
2. 2020 COVID Crash (DGI recovery 23% faster than S\&P 500) [^11]
3. 2022 Inflation Surge (DGI underperformed by 9.2%) [^10]

---

## **Historical Performance**

### **Backtest Results (1990–2023)**

```python
# Simplified backtest logic
import numpy as np

def dgi_backtest(data):
    portfolio = []
    for year in range(1990, 2024):
        # Screen dividend growers
        candidates = dgi_screener(data[year])
        
        # Apply hybrid weighting
        candidates['weight'] = (
            0.5 * candidates['dividend_yield'] +
            0.3 * candidates['div_growth_5y'] +
            0.2 * candidates['quality_score']
        )
        
        # Rebalance with 5% threshold
        portfolio = rebalance(portfolio, candidates)
        
    return calculate_metrics(portfolio)
```

**Key findings:**

- \$10,000 initial investment grew to \$412,000 vs \$287,000 for S\&P 500 [^9] [^17]
- 12.1% CAGR vs 10.4% for S\&P 500 (dividends reinvested) [^13]
- 33% lower volatility (β=0.67 vs market) [^10]

---

## **Recommended Enhancements**

1. **Dividend Growth Momentum**: Overweight companies with accelerating dividend growth rates (3Y CAGR >5Y CAGR) [^16]
2. **Earnings Quality Filter**: Exclude companies with >20% of EPS growth from share buybacks [^13]
3. **Tax Optimization**: Pair DGI with tax-loss harvesting in separate account structures [^5]
4. **Global Diversification**: Allocate 30% to international dividend growers meeting quality criteria [^11]

This framework combines academic research [^9] [^10], practitioner insights [^13] [^17], and technological tools [^5] [^16] to create a robust dividend growth strategy. By focusing on quality compounders with disciplined risk management, investors can harness the dual engines of income growth and capital appreciation [^15].

<div style="text-align: center">⁂</div>

 [^1]: https://www.investopedia.com/articles/basics/04/072304.asp

 [^2]: https://www.spglobal.com/spdji/en/documents/research/research-a-case-for-dividend-growth-strategies.pdf

 [^3]: https://www.vaneck.com/corp/en/news-and-insights/blogs/income-investing/how-to-develop-a-dividend-investing-strategy-a-comprehensive-guide/

 [^4]: https://advisor.morganstanley.com/brock-zielmanski-group/documents/field/b/br/brock-zielmanski-group/Dividend_Aristocrats_Enhanced_Covered_Call_Strategy_CRC4098278.pdf

 [^5]: https://www.reddit.com/r/dividends/comments/omoq1r/a_dividend_calculation_tool_for_backtesting/

 [^6]: https://fairmontequities.com/the-disadvantages-of-dividend-stocks/

 [^7]: https://www.reddit.com/r/dividends/comments/1jjzdwt/favorite_screening_criteria_for_dividend_growth/

 [^8]: https://www.ssga.com/se/en_gb/intermediary/library-content/products/fund-docs/etfs/emea/spdr-new-dividend-aristocrats.pdf

 [^9]: https://seekingalpha.com/article/1031581-backtesting-dividend-growth-vs-dividend-yield

 [^10]: https://www.spglobal.com/spdji/en/documents/research/research-a-case-for-dividend-growth-strategies.pdf

 [^11]: https://www.ssga.com/library-content/story/general/etf/emea/spdr-an-overview-of-dividend-aristocrats-strategies-ch.pdf

 [^12]: https://www.dividend.com/how-to-invest/5-key-dividend-screener-searches-to-track/

 [^13]: https://www.suredividend.com/8-rules-mbeat/

 [^14]: https://www.thestreet.com/etffocus/dividend-ideas/backtesting-3-dividend-income-etf-portfolio-ideas-good-bad-ugly

 [^15]: https://www.dividend.com/dividend-investing-101/understanding-dividend-growth-strategy/

 [^16]: https://www.greatworklife.com/dividend-growth-screening/

 [^17]: https://www.nanalyze.com/dividend-growth-investing-strategy/

 [^18]: https://www.lenoxadvisors.com/insights/dividend-growth-investing-as-a-long-term-strategy/

 [^19]: https://www.proshares.com/browse-all-insights/insights/why-dividend-growth-is-a-timeless-strategy

 [^20]: https://www.investopedia.com/articles/markets/060116/4-ratios-evaluate-dividend-stocks.asp
