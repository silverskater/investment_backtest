import pytest
import pandas as pd

from backtest.utils.portfolio_calculations import calculate_portfolio_entry_return


@pytest.mark.unit
class TestPortfolioCalculations:

    def test_empty_or_none_entry(self):
        assert calculate_portfolio_entry_return(None) == 0.0
        assert calculate_portfolio_entry_return({}) == 0.0
        assert calculate_portfolio_entry_return({"stocks": []}) == 0.0

    def test_empty_stocks_dataframe(self):
        entry = {"stocks": pd.DataFrame(columns=['symbol', 'weight', 'annual_return']).to_dict('records')}
        assert calculate_portfolio_entry_return(entry) == 0.0

    def test_missing_required_columns(self):
        entry_no_return = {"stocks": [{'symbol': 'A', 'weight': 1.0}]}
        assert calculate_portfolio_entry_return(entry_no_return) == 0.0

        entry_no_weight = {"stocks": [{'symbol': 'A', 'annual_return': 10.0}]}
        assert calculate_portfolio_entry_return(entry_no_weight) == 0.0

    def test_single_stock_portfolio(self):
        entry = {"stocks": [{'symbol': 'A', 'weight': 1.0, 'annual_return': 15.0}]}
        # annual_return is in %, so 15.0 / 100.0 = 0.15. weight is 1.0. 0.15 * 1.0 = 0.15
        assert calculate_portfolio_entry_return(entry) == pytest.approx(0.15)

    def test_multiple_stocks_portfolio(self):
        entry = {"stocks": [
            {'symbol': 'A', 'weight': 0.5, 'annual_return': 10.0},  # 0.5 * 0.10 = 0.05
            {'symbol': 'B', 'weight': 0.3, 'annual_return': -5.0},  # 0.3 * -0.05 = -0.015
            {'symbol': 'C', 'weight': 0.2, 'annual_return': 20.0}  # 0.2 * 0.20 = 0.04
        ]}
        # Total = 0.05 - 0.015 + 0.04 = 0.075
        assert calculate_portfolio_entry_return(entry) == pytest.approx(0.075)

    def test_cash_only_portfolio(self):
        entry_cash = {"stocks": [{'symbol': 'CASH', 'weight': 1.0, 'annual_return': 1.0}]}  # 1% return on cash
        assert calculate_portfolio_entry_return(entry_cash) == pytest.approx(0.01)

        entry_cash_no_return = {"stocks": [{'symbol': 'CASH', 'weight': 1.0}]}
        assert calculate_portfolio_entry_return(entry_cash_no_return) == 0.0

    def test_mixed_stocks_and_cash_like_entry(self):
        # This scenario shouldn't typically happen if CASH means 100% cash,
        # but the function should handle it based on its logic.
        entry = {"stocks": [
            {'symbol': 'A', 'weight': 0.8, 'annual_return': 10.0},
            {'symbol': 'CASH', 'weight': 0.2, 'annual_return': 1.0}  # This is unusual
        ]}
        # (0.8 * 0.10) + (0.2 * 0.01) = 0.08 + 0.002 = 0.082
        assert calculate_portfolio_entry_return(entry) == pytest.approx(0.082)

    def test_non_numeric_values_handled(self):
        entry = {"stocks": [
            {'symbol': 'A', 'weight': 0.5, 'annual_return': "10.0"},  # string return
            {'symbol': 'B', 'weight': "0.3", 'annual_return': -5.0},  # string weight
            {'symbol': 'C', 'weight': 0.2, 'annual_return': None}  # None return
        ]}
        # (0.5 * 0.10) + (0.3 * -0.05) + (0.2 * 0.0) = 0.05 - 0.015 + 0 = 0.035
        assert calculate_portfolio_entry_return(entry) == pytest.approx(0.035)

        entry_bad_values = {"stocks": [
            {'symbol': 'A', 'weight': 'bad', 'annual_return': "data"}
        ]}
        assert calculate_portfolio_entry_return(entry_bad_values) == 0.0
