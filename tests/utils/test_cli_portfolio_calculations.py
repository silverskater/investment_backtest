"""Unit tests for portfolio calculation utility functions.

This module specifically tests the `calculate_portfolio_entry_return` function
used within the CLI and potentially other parts of the backtesting system.
"""
import pandas as pd
import pytest

from backtest.utils.portfolio_calculations import calculate_portfolio_entry_return


@pytest.mark.unit
class TestPortfolioCalculations:
    """Test suite for portfolio calculation utilities."""

    def test_empty_or_none_entry(self):
        """Tests return calculation with None or empty portfolio entries."""
        assert calculate_portfolio_entry_return(None) == 0.0
        assert calculate_portfolio_entry_return({}) == 0.0
        assert calculate_portfolio_entry_return({"stocks": []}) == 0.0

    def test_empty_stocks_dataframe(self):
        """Tests return calculation when the 'stocks' list results in an empty DataFrame."""
        entry = {"stocks": pd.DataFrame(
            columns=['symbol', 'weight', 'annual_return']
        ).to_dict('records')}
        assert calculate_portfolio_entry_return(entry) == 0.0

    def test_missing_required_columns(self):
        """Tests return calculation when essential columns ('annual_return', 'weight') are missing."""
        entry_no_return = {"stocks": [{'symbol': 'A', 'weight': 1.0}]}
        assert calculate_portfolio_entry_return(entry_no_return) == 0.0

        entry_no_weight = {"stocks": [{'symbol': 'A', 'annual_return': 10.0}]}
        assert calculate_portfolio_entry_return(entry_no_weight) == 0.0

    def test_single_stock_portfolio(self):
        """Tests return calculation for a portfolio with a single stock."""
        entry = {"stocks": [{'symbol': 'A', 'weight': 1.0, 'annual_return': 15.0}]}
        # Expected: 15.0% / 100.0 * 1.0 = 0.15.
        assert calculate_portfolio_entry_return(entry) == pytest.approx(0.15)

    def test_multiple_stocks_portfolio(self):
        """Tests return calculation for a portfolio with multiple stocks."""
        entry = {"stocks": [
            {'symbol': 'A', 'weight': 0.5, 'annual_return': 10.0},  # 0.5 * 0.10 = 0.05
            {'symbol': 'B', 'weight': 0.3, 'annual_return': -5.0},  # 0.3 * -0.05 = -0.015
            {'symbol': 'C', 'weight': 0.2, 'annual_return': 20.0}   # 0.2 * 0.20 = 0.04
        ]}
        # Total expected: 0.05 - 0.015 + 0.04 = 0.075.
        assert calculate_portfolio_entry_return(entry) == pytest.approx(0.075)

    def test_cash_only_portfolio(self):
        """Tests return calculation for a portfolio consisting only of CASH."""
        entry_cash_with_return = {
            "stocks": [{'symbol': 'CASH', 'weight': 1.0, 'annual_return': 1.0}]  # 1% return on cash.
        }
        assert calculate_portfolio_entry_return(entry_cash_with_return) == pytest.approx(0.01)

        entry_cash_no_return_field = {"stocks": [{'symbol': 'CASH', 'weight': 1.0}]}
        assert calculate_portfolio_entry_return(entry_cash_no_return_field) == 0.0

    def test_mixed_stocks_and_cash_like_entry(self):
        """Tests return calculation with a mix of regular stocks and a CASH-like entry.

        Note: This scenario (CASH not being 100% of portfolio) is unusual but
        should be handled correctly by the calculation logic.
        """
        entry = {"stocks": [
            {'symbol': 'A', 'weight': 0.8, 'annual_return': 10.0},
            {'symbol': 'CASH', 'weight': 0.2, 'annual_return': 1.0}
        ]}
        # Expected: (0.8 * 0.10) + (0.2 * 0.01) = 0.08 + 0.002 = 0.082.
        assert calculate_portfolio_entry_return(entry) == pytest.approx(0.082)

    def test_non_numeric_values_handled(self):
        """Tests that non-numeric values in 'annual_return' or 'weight' are handled.

        The function should coerce these to numeric, with unconvertible values
        becoming NaN and then typically 0.0 for calculation.
        """
        entry_strings = {"stocks": [
            {'symbol': 'A', 'weight': 0.5, 'annual_return': "10.0"},    # String return.
            {'symbol': 'B', 'weight': "0.3", 'annual_return': -5.0},    # String weight.
            {'symbol': 'C', 'weight': 0.2, 'annual_return': None}      # None return.
        ]}
        # Expected: (0.5 * 0.10) + (0.3 * -0.05) + (0.2 * 0.0) = 0.05 - 0.015 + 0 = 0.035.
        assert calculate_portfolio_entry_return(entry_strings) == pytest.approx(0.035)

        entry_bad_values = {"stocks": [
            {'symbol': 'A', 'weight': 'bad_weight', 'annual_return': "bad_data"}
        ]}
        # Coerced to NaN, then 0.0, so result should be 0.0.
        assert calculate_portfolio_entry_return(entry_bad_values) == 0.0