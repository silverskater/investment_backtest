# Investment Strategy: Value Investing

This document outlines a systematic approach to value investing, focusing on identifying undervalued companies with strong fundamentals, margin of safety, and long-term compounding potential[^3][^4].

---

## **Core Strategy Overview**

### **1. Stock Selection Criteria**

- **Undervaluation metrics**[^4][^5]:
    - Price-to-earnings (P/E) ratio below sector median or less than 40% of the stock's highest P/E over the previous five years[^8][^9]
    - Price-to-book (P/B) ratio below 1.0 or below sector median[^8][^9]
    - Free cash flow yield above sector median[^3][^4]
    - Price less than 67% of tangible per-share book value[^8][^9]
- **Financial health**[^7][^9]:
    - Debt-to-equity ratio below industry average or total debt less than twice net current asset value[^7][^8]
    - Current ratio above 1.5, preferably above 2.0[^7][^9]
    - Positive net income for at least 5 consecutive years with no earnings deficits[^9][^10]
    - Return on equity (ROE) above 15%[^7][^11]
- **Margin of safety**[^6][^10]:
    - Estimated intrinsic value at least 20-30% above current market price using discounted cash flow analysis[^6][^11]
    - Company's total book value greater than total debt[^8][^9]
- **Quality filters**[^5][^7]:
    - S\&P Quality Rank of B+ or better[^7][^9]
    - Consistent earnings growth with positive earnings per share growth over five years[^9][^10]
    - Management with significant insider ownership or recent insider buying[^3][^7]


### **2. Portfolio Construction**

- **Sector diversification**[^14][^15]: Limit sector exposure to 25% per sector, with targeted exposure to traditionally value-oriented industries (financials, industrials, consumer staples)[^14][^16]
- **Weighting method**[^14][^22]:
    - 50% equal weight allocation
    - 25% based on margin of safety (greater discount to intrinsic value)[^6][^11]
    - 25% based on quality score combining ROE, earnings stability, and insider ownership[^7][^11]
- **Portfolio size**: Optimal range of 10-30 stocks for adequate diversification while maintaining conviction[^9][^10]
- **Rebalancing**: Annual review with rebalancing triggered if sector weights deviate by more than 5% or individual stock weights exceed 8%[^14][^15]


### **3. Risk Management**

- **Dynamic position sizing**: Reduce position size if P/E or P/B rises above sector median without fundamental improvement[^15][^22]
- **Stop-loss discipline**: Trim positions if price falls below 80% of estimated intrinsic value without fundamental deterioration[^3][^15]
- **Peer-relative analysis**: Control for sector and country-level allocation effects to focus on stock selection rather than broad allocation decisions[^22][^14]

---

## **Potential Weak Spots**

### **1. Value Traps**

- **Fundamental deterioration**: Companies may be cheap for legitimate reasons including declining business models, regulatory challenges, or structural industry headwinds[^23][^19]
- **Cyclical timing**: Value stocks in cyclical industries may face extended periods of poor performance during economic downturns[^23][^24]
- **Quality issues**: Low-quality companies with deteriorating fundamentals may never recover to fair value despite appearing statistically cheap[^23][^26]


### **2. Macroeconomic Vulnerabilities**

- **Interest rate sensitivity**: Value stocks can underperform during periods of rising interest rates, particularly when growth stocks are favored[^24][^31]
- **Economic cycle dependence**: Value strategies may struggle during extended growth phases when investors favor momentum and innovation over traditional metrics[^30][^31]
- **Inflation impact**: Many value companies lack pricing power and may struggle during inflationary periods[^17][^24]


### **3. Behavioral Challenges**

- **Extended underperformance periods**: Value investing has experienced prolonged periods of underperformance, including 12+ years of trailing growth stocks[^31][^34]
- **Contrarian psychology**: Requires discipline to buy unpopular stocks and hold through negative sentiment[^5][^17]
- **Market efficiency concerns**: Increasing market efficiency may reduce the frequency and magnitude of mispricings[^4][^22]


### **4. Implementation Risks**

- **Survivorship bias**: Historical backtests may overstate returns by excluding companies that failed[^18][^13]
- **Style drift**: Tendency to chase performance by modifying criteria during underperformance periods[^31][^22]
- **Concentration risk**: Value strategies may become concentrated in certain sectors or geographies[^14][^19]

---

## **Implementation Improvements**

### **1. Enhanced Screening Framework**

```python
def value_screener(df):
    return df[
        (df['pe_ratio'] < df['sector_median_pe'] * 0.4) &
        (df['pb_ratio'] < 1.0) &
        (df['current_ratio'] > 1.5) &
        (df['debt_equity'] < df['industry_avg_debt_equity']) &
        (df['roe'] > 0.15) &
        (df['earnings_growth_5y'] > 0) &
        (df['sp_quality_rank'] >= 'B+') &
        (df['margin_of_safety'] > 0.25)
    ].sort_values('composite_value_score', ascending=False)
```


### **2. Dynamic Risk Management**

- **Peer-relative valuation**: Focus on within-sector comparisons rather than absolute metrics to avoid sector concentration[^22][^14]
- **Quality overlay**: Exclude companies with deteriorating fundamentals using earnings quality metrics and cash flow analysis[^22][^13]
- **Cyclical awareness**: Reduce exposure to cyclical sectors when economic indicators suggest downturn[^24][^27]
- **Interest rate hedge**: Monitor duration exposure and adjust when yield spreads indicate unfavorable conditions[^24][^19]


### **3. Backtesting Protocol**

**Key metrics**[^18][^28]:


| Metric | Target | Calculation Method |
| :-- | :-- | :-- |
| Sharpe ratio | >0.8 | Excess returns / standard deviation |
| Maximum drawdown | <30% | Worst peak-to-trough decline |
| Total Return vs S\&P | +1.5% CAGR | With dividends reinvested |
| Value Premium Capture | >75% | % of historical value premium captured |
| Downside Protection | <-15% | Performance during market crashes |

**Stress test scenarios**[^27][^28]:

1. **2008 Financial Crisis**: Value stocks historically provided better downside protection during credit crises[^30][^34]
2. **2000-2002 Tech Bubble**: Value significantly outperformed growth during this period[^30][^34]
3. **2010-2020 Growth Regime**: Extended period of value underperformance to test discipline[^31][^34]
4. **Interest Rate Shock**: Rapid rate increases testing sensitivity to monetary policy changes[^24][^27]

---

## **Historical Performance**

**Backtest Results (1927–2023)**[^30][^31]:

```python
def value_backtest(data):
    portfolio = []
    for year in range(1927, 2024):
        # Screen for value stocks using Graham criteria
        candidates = value_screener(data[year])
        
        # Apply margin of safety weighting
        candidates['weight'] = (
            0.5 * (1 / len(candidates)) +
            0.25 * candidates['margin_of_safety'] +
            0.25 * candidates['quality_score']
        )
        
        # Include delisted companies to avoid survivorship bias
        portfolio = rebalance(portfolio, candidates, include_delisted=True)
        
    return calculate_metrics(portfolio)
```

**Key findings**[^30][^34]:

- **Long-term outperformance**: Value stocks have historically outperformed growth stocks by 4.4% annually since 1927[^30]
- **Periodic underperformance**: Extended periods of underperformance including 1926-1941 and 2007-2020[^31][^34]
- **Quick reversals**: Value premiums often appear rapidly with average outperformance of nearly 15% in positive years[^30][^34]

---

## **Recommended Enhancements**

1. **Multi-factor integration**: Combine value signals with quality and momentum factors to improve signal-to-noise ratio[^22][^13]
2. **Earnings quality focus**: Emphasize free cash flow and exclude companies relying heavily on accounting accruals[^13][^21]
3. **Global diversification**: Allocate 30-40% to international value stocks where the premium remains more robust[^22][^30]
4. **Dynamic sector allocation**: Adjust sector weights based on relative valuations and economic cycle positioning[^14][^24]

---

This framework combines Benjamin Graham's foundational principles with modern risk management techniques to create a robust value investing strategy[^3][^10]. By focusing on undervalued companies with strong fundamentals and maintaining disciplined risk controls, investors can harness the long-term benefits of value investing while mitigating its inherent vulnerabilities[^4][^17].

<div style="text-align: center">⁂</div>

[^1]: exp_fund.md

[^2]: dgi.md

[^3]: https://www.investopedia.com/terms/v/valueinvesting.asp

[^4]: https://www.home.saxo/learn/guides/trading-strategies/value-investing-what-it-is-and-how-it-works

[^5]: https://www.investopedia.com/articles/fundamental-analysis/09/value-investing.asp

[^6]: https://www.wallstreetprep.com/knowledge/value-investing-101/

[^7]: https://www.heartlandadvisors.com/Philosophy-Process/10-Principles-of-Value-Investing

[^8]: https://corporatefinanceinstitute.com/resources/capital-markets/a-guide-to-value-investing/

[^9]: https://www.cabotwealth.com/daily/value-stocks/benjamin-grahams-value-stock-criteria

[^10]: https://www.investopedia.com/articles/basics/07/grahamprinciples.asp

[^11]: https://tradeforgood.com.au/learn/buffet-investment-model/

[^12]: https://www.equitymaster.com/timeless-reading/12/basics-of-value-investing

[^13]: https://www.ijfmr.com/papers/2025/2/38662.pdf

[^14]: https://www.mackenzieinvestments.com/content/dam/final/corporate/mackenzie/docs/marketing-materials/mi-wp-portfolio-construction-value-vs-growth-en.pdf

[^15]: https://www.schwab.com/learn/story/value-stock-investments-build-durable-portfolio

[^16]: https://investordiary.com/value-investing-portfolio

[^17]: https://www.mintos.com/blog/value-investing/

[^18]: https://alphaarchitect.com/2014/10/value-investing-backtests-our-analysis-of-13-aaii-value-strategies/

[^19]: https://www.schwab.com/learn/story/4-weak-spots-current-market

[^20]: https://www.soa.org/493862/globalassets/assets/files/research/projects/research-cker-value-investing-erm.pdf

[^21]: https://fbe.unimelb.edu.au/__data/assets/pdf_file/0006/2565591/Sloan_-_ajms_-_20130610.pdf

[^22]: https://www.acadian-asset.com/-/media/files/thematic-research-paper-pdfs/value-papers/acadian---acadians-approach-to-value-investing.pdf

[^23]: https://corporatefinanceinstitute.com/resources/career-map/sell-side/capital-markets/value-trap/

[^24]: https://www.yieldstreet.com/blog/article/economic-rhythms-investing-in-cyclical-stocks/

[^25]: https://www.indeed.com/career-advice/career-development/margin-of-safety-formula

[^26]: https://business.columbia.edu/sites/default/files-efs/pubfiles/18184/Penman_value_trap.pdf

[^27]: https://www.phoenixstrategy.group/blog/5-scenarios-to-stress-test-portfolio-volatility

[^28]: https://www.morningstar.com/business/insights/blog/portfolio-construction/portfolio-stress-testing

[^29]: https://www.investopedia.com/ask/answers/061515/what-stress-testing-value-risk-var.asp

[^30]: https://www.dimensional.com/ca-en/insights/when-its-value-versus-growth-history-is-on-values-side

[^31]: https://osam.com/Commentary/value-is-dead-long-live-value

[^32]: https://blog.portfolio123.com/a-stock-pickers-guide-to-benjamin-grahams-screening-rules/

[^33]: https://www.thewealthmosaic.com/vendors/jacobi/blogs/the-importance-of-stress-testing-and-scenario-anal/

[^34]: https://www.dimensional.com/de-de/insights/value-judgments-viewing-the-premiums-performance-through-historys-lens

[^35]: https://rpc.cfainstitute.org/topics/portfolio-construction-and-investment

[^36]: https://caia.org/blog/2025/06/03/beyond-divide-value-continuum

[^37]: https://www.youtube.com/watch?v=fBIC3Nf0hPU

[^38]: https://www.nb.com/handlers/documents.ashx?id=c7474dd6-9009-42a0-a4cb-8e044f98e299

[^39]: https://www.investopedia.com/terms/s/stresstesting.asp

[^40]: https://www.blackrock.com/us/financial-professionals/tools/analyze-portfolio-risk

