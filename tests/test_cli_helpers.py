import pytest
import pandas as pd
import json

from typing import List, Dict, Any
from unittest.mock import patch  # For mocking click.echo if needed

from backtest.cli import (
    rebalance,
    calculate_metrics,
    load_market_data,
    NOTIONAL_PORTFOLIO_VALUE_FOR_TRADES,
    DEFAULT_TRANSACTION_COST
)


# --- Fixtures and Helper Data ---

@pytest.fixture
def sample_target_stocks_df() -> pd.DataFrame:
    """Provides a sample target_stocks DataFrame."""
    return pd.DataFrame([
        {'symbol': 'AAPL', 'weight': 0.5, 'share_price': 150.0, 'annual_return': 10.0},
        {'symbol': 'MSFT', 'weight': 0.3, 'share_price': 300.0, 'annual_return': 5.0},
        {'symbol': 'GOOG', 'weight': 0.2, 'share_price': 2500.0, 'annual_return': 15.0},
    ])


@pytest.fixture
def sample_portfolio_history_entry(sample_target_stocks_df: pd.DataFrame) -> Dict[str, Any]:
    """Provides a sample portfolio history entry."""
    return {
        'date': '2021',
        'stocks': sample_target_stocks_df.to_dict('records'),
        'action': 'initial_investment',
        'value_bought': NOTIONAL_PORTFOLIO_VALUE_FOR_TRADES,
        'value_sold': 0.0,
        'transaction_cost': NOTIONAL_PORTFOLIO_VALUE_FOR_TRADES * DEFAULT_TRANSACTION_COST
    }


# --- Tests for rebalance() ---
@pytest.mark.unit
class TestRebalance:
    def test_initial_investment(self, sample_target_stocks_df: pd.DataFrame):
        history: List[Dict[str, Any]] = []
        updated_history = rebalance(
            history,
            sample_target_stocks_df.copy(),  # Pass a copy as rebalance might modify it
            '2020',
            transaction_cost_rate=0.001
        )
        assert len(updated_history) == 1
        entry = updated_history[0]
        assert entry['action'] == 'initial_investment'
        assert entry['date'] == '2020'
        assert len(entry['stocks']) == 3
        assert entry['value_bought'] == pytest.approx(NOTIONAL_PORTFOLIO_VALUE_FOR_TRADES * 1.0)  # sum of weights
        assert entry['value_sold'] == 0.0
        assert entry['transaction_cost'] == pytest.approx(NOTIONAL_PORTFOLIO_VALUE_FOR_TRADES * 1.0 * 0.001)

    def test_standard_rebalance_buys_sells(self, sample_portfolio_history_entry: Dict[str, Any]):
        current_history = [sample_portfolio_history_entry]
        new_target_stocks = pd.DataFrame([
            {'symbol': 'AAPL', 'weight': 0.4, 'share_price': 160.0},  # Reduced
            {'symbol': 'MSFT', 'weight': 0.4, 'share_price': 310.0},  # Increased
            {'symbol': 'TSLA', 'weight': 0.2, 'share_price': 800.0},  # New
            # GOOG is sold
        ])
        updated_history = rebalance(
            current_history,
            new_target_stocks.copy(),
            '2021',
            transaction_cost_rate=0.001
        )
        assert len(updated_history) == 2
        entry = updated_history[1]
        assert entry['action'] == 'rebalance'
        assert entry['date'] == '2021'
        # AAPL: 0.5 -> 0.4 (sell 0.1)
        # MSFT: 0.3 -> 0.4 (buy 0.1)
        # GOOG: 0.2 -> 0.0 (sell 0.2)
        # TSLA: 0.0 -> 0.2 (buy 0.2)
        # Total sold weight: 0.1 (AAPL) + 0.2 (GOOG) = 0.3
        # Total bought weight: 0.1 (MSFT) + 0.2 (TSLA) = 0.3
        expected_value_sold = 0.3 * NOTIONAL_PORTFOLIO_VALUE_FOR_TRADES
        expected_value_bought = 0.3 * NOTIONAL_PORTFOLIO_VALUE_FOR_TRADES
        assert entry['value_sold'] == pytest.approx(expected_value_sold)
        assert entry['value_bought'] == pytest.approx(expected_value_bought)
        expected_tx_cost = (expected_value_bought + expected_value_sold) * 0.001
        assert entry['transaction_cost'] == pytest.approx(expected_tx_cost)
        assert len(entry['stocks']) == 3  # AAPL, MSFT, TSLA

    def test_rebalance_to_cash(self, sample_portfolio_history_entry: Dict[str, Any]):
        current_history = [sample_portfolio_history_entry]
        empty_target_stocks = pd.DataFrame(columns=['symbol', 'weight', 'share_price'])
        updated_history = rebalance(
            current_history,
            empty_target_stocks.copy(),
            '2021',
            transaction_cost_rate=0.001
        )
        assert len(updated_history) == 2
        entry = updated_history[1]
        assert entry['action'] == 'rebalance'
        assert entry['stocks'] == []  # Rebalanced to cash
        # All previous holdings (total weight 1.0) are sold
        assert entry['value_sold'] == pytest.approx(NOTIONAL_PORTFOLIO_VALUE_FOR_TRADES * 1.0)
        assert entry['value_bought'] == 0.0
        assert entry['transaction_cost'] == pytest.approx(NOTIONAL_PORTFOLIO_VALUE_FOR_TRADES * 1.0 * 0.001)

    def test_rebalance_from_cash(self, sample_target_stocks_df: pd.DataFrame):
        cash_history_entry = {
            'date': '2020',
            'stocks': [{'symbol': 'CASH', 'weight': 1.0, 'share_price': 1.0}],  # Simplified cash representation
            'action': 'rebalance', 'value_bought': 0, 'value_sold': 0, 'transaction_cost': 0
        }
        current_history = [cash_history_entry]
        updated_history = rebalance(
            current_history,
            sample_target_stocks_df.copy(),
            '2021',
            transaction_cost_rate=0.001
        )
        assert len(updated_history) == 2
        entry = updated_history[1]
        assert entry['action'] == 'rebalance'
        # All target stocks (total weight 1.0) are bought
        assert entry['value_bought'] == pytest.approx(NOTIONAL_PORTFOLIO_VALUE_FOR_TRADES * 1.0)
        assert entry['value_sold'] == 0.0  # Sold from CASH
        assert entry['transaction_cost'] == pytest.approx(NOTIONAL_PORTFOLIO_VALUE_FOR_TRADES * 1.0 * 0.001)


    def test_dynamic_rebalance_no_change(self, sample_portfolio_history_entry: Dict[str, Any]):
        current_history = [sample_portfolio_history_entry]
        # Target stocks are identical to current holdings
        target_stocks_no_change = pd.DataFrame(sample_portfolio_history_entry['stocks'])

        updated_history = rebalance(
            current_history,
            target_stocks_no_change.copy(),
            '2021',
            transaction_cost_rate=0.001, # Using specific value from original test
            dynamic_rebalance_active=True,
            deviation_threshold=0.05
        )
        assert len(updated_history) == 2
        entry = updated_history[1]
        assert entry['action'] == 'hold'
        assert entry['value_bought'] == 0.0
        assert entry['value_sold'] == 0.0
        assert entry['transaction_cost'] == 0.0
        assert entry['stocks'] == sample_portfolio_history_entry['stocks']  # Carried forward

    def test_dynamic_rebalance_triggered_by_deviation(self, sample_portfolio_history_entry: Dict[str, Any]):
        current_history = [sample_portfolio_history_entry]  # AAPL:0.5, MSFT:0.3, GOOG:0.2
        target_stocks_deviated = pd.DataFrame([
            {'symbol': 'AAPL', 'weight': 0.3, 'share_price': 150.0},  # Deviated by 0.2 (>0.05)
            {'symbol': 'MSFT', 'weight': 0.5, 'share_price': 300.0},  # Deviated by 0.2
            {'symbol': 'GOOG', 'weight': 0.2, 'share_price': 2500.0},  # No change
        ])
        updated_history = rebalance(
            current_history,
            target_stocks_deviated.copy(),
            '2021',
            transaction_cost_rate=0.001,
            dynamic_rebalance_active=True,
            deviation_threshold=0.05
        )
        assert len(updated_history) == 2
        entry = updated_history[1]
        assert entry['action'] == 'rebalance'
        assert entry['value_bought'] > 0  # MSFT bought
        assert entry['value_sold'] > 0  # AAPL sold

    def test_dynamic_rebalance_triggered_by_new_stock(self, sample_portfolio_history_entry: Dict[str, Any]):
        current_history = [sample_portfolio_history_entry]
        target_stocks_new = pd.DataFrame([
            {'symbol': 'AAPL', 'weight': 0.5, 'share_price': 150.0},
            {'symbol': 'MSFT', 'weight': 0.3, 'share_price': 300.0},
            {'symbol': 'TSLA', 'weight': 0.2, 'share_price': 800.0},  # New stock, GOOG removed
        ])
        updated_history = rebalance(
            current_history,
            target_stocks_new.copy(),
            '2021',
            transaction_cost_rate=DEFAULT_TRANSACTION_COST,
            dynamic_rebalance_active=True,
            deviation_threshold=0.05
        )
        assert len(updated_history) == 2
        entry = updated_history[1]
        assert entry['action'] == 'rebalance'

    def test_dynamic_rebalance_triggered_by_sold_stock(self, sample_portfolio_history_entry: Dict[str, Any]):
        current_history = [sample_portfolio_history_entry]  # AAPL, MSFT, GOOG
        target_stocks_sold = pd.DataFrame([
            {'symbol': 'AAPL', 'weight': 0.7, 'share_price': 150.0},
            {'symbol': 'MSFT', 'weight': 0.3, 'share_price': 300.0},
            # GOOG is sold (not in target)
        ])
        updated_history = rebalance(
            current_history,
            target_stocks_sold.copy(),
            '2021',
            transaction_cost_rate=DEFAULT_TRANSACTION_COST,
            dynamic_rebalance_active=True,
            deviation_threshold=0.05
        )
        assert len(updated_history) == 2
        entry = updated_history[1]
        assert entry['action'] == 'rebalance'

    @patch('click.echo')
    def test_rebalance_target_weights_sum_zero(self, mock_click_echo, sample_portfolio_history_entry: Dict[str, Any]):
        current_history = [sample_portfolio_history_entry]
        target_stocks_zero_weight = pd.DataFrame([
            {'symbol': 'AAPL', 'weight': 0.0, 'share_price': 150.0},
            {'symbol': 'MSFT', 'weight': 0.0, 'share_price': 300.0},
        ])
        updated_history = rebalance(
            current_history,
            target_stocks_zero_weight.copy(),
            '2021',
            transaction_cost_rate=DEFAULT_TRANSACTION_COST
        )
        mock_click_echo.assert_any_call(
            "Warning: Target stock weights for 2021 sum to zero or are invalid (0.0). "
            "Assuming equal weighting for target stocks.",
            err=True
        )
        entry = updated_history[1]
        assert len(entry['stocks']) == 2
        # Check if weights were set to equal
        total_weight_after_fallback = sum(s['weight'] for s in entry['stocks'])
        assert total_weight_after_fallback == pytest.approx(1.0)
        assert entry['stocks'][0]['weight'] == pytest.approx(0.5)
        assert entry['stocks'][1]['weight'] == pytest.approx(0.5)

    @patch('click.echo')
    def test_rebalance_target_missing_share_price(self, mock_click_echo,
                                                  sample_portfolio_history_entry: Dict[str, Any]):
        current_history = [sample_portfolio_history_entry]
        target_stocks_no_price = pd.DataFrame([
            {'symbol': 'AAPL', 'weight': 1.0},  # Missing share_price
        ])
        updated_history = rebalance(
            current_history,
            target_stocks_no_price.copy(),
            '2021',
            transaction_cost_rate=DEFAULT_TRANSACTION_COST # FIXED: Added argument
        )
        mock_click_echo.assert_any_call(
            "Warning: 'share_price' missing in target_stocks for 2021. Using placeholder 1.0.",
            err=True
        )
        entry = updated_history[1]
        assert entry['stocks'][0]['share_price'] == 1.0


# --- Tests for calculate_metrics() ---
@pytest.mark.unit
class TestCalculateMetrics:
    def test_metrics_empty_history(self):
        metrics = calculate_metrics([])
        expected_zeros = {
            'sharpe_ratio': 0.0, 'max_drawdown': 0.0, 'sp500_comparison': 0.0,
            'turnover_ratio': 0.0, 'average_annual_return': 0.0,
            'total_return': 0.0, 'portfolio_size': 0, 'transaction_costs_total': 0.0
        }
        assert metrics == expected_zeros

    def test_metrics_single_period_positive_return(self):
        portfolio_history = [
            {'date': '2020', 'stocks': [], 'action': 'initial_investment', 'value_bought': 100, 'value_sold': 0,
             'transaction_cost': 0.1},
            {'date': '2021', 'stocks': [{'symbol': 'A', 'weight': 1.0, 'annual_return': 10.0}],  # 10% return
             'action': 'rebalance', 'value_bought': 100, 'value_sold': 0, 'transaction_cost': 0.1}
        ]
        metrics = calculate_metrics(portfolio_history)
        assert metrics['total_return'] == pytest.approx(10.0)  # (1 + 0.1) - 1 = 0.1
        assert metrics['average_annual_return'] == pytest.approx(10.0)  # CAGR for 1 period
        # Sharpe: (0.10 - 0.02) / std_dev. If std_dev is 0 (only one return), it's inf or specific handling.
        # Current implementation: if std_dev is 0 and mean > risk_free, sharpe is inf.
        assert metrics['sharpe_ratio'] == float('inf')
        assert metrics['max_drawdown'] == 0.0  # No drawdown with one positive return
        assert metrics['portfolio_size'] == 1

    def test_metrics_single_period_negative_return(self):
        portfolio_history = [
            {'date': '2020', 'stocks': [], 'action': 'initial_investment', 'value_bought': 100, 'value_sold': 0,
             'transaction_cost': 0.1},
            {'date': '2021', 'stocks': [{'symbol': 'A', 'weight': 1.0, 'annual_return': -5.0}],  # -5% return
             'action': 'rebalance', 'value_bought': 100, 'value_sold': 0, 'transaction_cost': 0.1}
        ]
        metrics = calculate_metrics(portfolio_history)
        assert metrics['total_return'] == pytest.approx(-5.0)
        assert metrics['average_annual_return'] == pytest.approx(-5.0)
        assert metrics['max_drawdown'] == pytest.approx(5.0)

    def test_metrics_multiple_periods_mixed_returns(self):
        portfolio_history = [
            {'date': '2019', 'stocks': [], 'action': 'initial_investment', 'value_bought': 100, 'value_sold': 0,
             'transaction_cost': 0.1},
            {'date': '2020', 'stocks': [{'symbol': 'A', 'weight': 1.0, 'annual_return': 10.0}],  # +10%
             'action': 'rebalance', 'value_bought': 10, 'value_sold': 0, 'transaction_cost': 0.01},
            {'date': '2021', 'stocks': [{'symbol': 'A', 'weight': 1.0, 'annual_return': -5.0}],  # -5%
             'action': 'rebalance', 'value_bought': 0, 'value_sold': 5, 'transaction_cost': 0.005},
            {'date': '2022', 'stocks': [{'symbol': 'A', 'weight': 1.0, 'annual_return': 20.0}],  # +20%
             'action': 'rebalance', 'value_bought': 20, 'value_sold': 0, 'transaction_cost': 0.02}
        ]
        # Returns: 0.10, -0.05, 0.20
        # Total growth: (1.10) * (0.95) * (1.20) = 1.254
        # Total return: 1.254 - 1 = 0.254 (25.4%)
        # CAGR: (1.254)**(1/3) - 1 = 0.0783... (7.83%)
        metrics = calculate_metrics(portfolio_history)
        assert metrics['total_return'] == pytest.approx(25.4, abs=0.01)
        assert metrics['average_annual_return'] == pytest.approx(7.83, abs=0.01)
        # Max drawdown: After +10%, then -5%. Value goes 1 -> 1.1 -> 1.045. Peak 1.1. Drawdown (1.045-1.1)/1.1 = -0.05/1.1 = -0.04545
        assert metrics['max_drawdown'] == pytest.approx(5.0,
                                                        abs=0.01)  # (1.1 -> 1.045) is a 5% drop from peak of 1.1 if we consider the values.
        # The code calculates based on (1+r).cumprod()
        # (1.1), (1.1*0.95=1.045). Peak = 1.1. Drawdown = (1.045-1.1)/1.1 = -0.05
        # Wait, the drawdown is on the compounded growth.
        # Growth factors: 1.1, 1.045, 1.254
        # Peaks: 1.1, 1.1, 1.254
        # Drawdowns: (1.1-1.1)/1.1=0, (1.045-1.1)/1.1 = -0.05, (1.254-1.254)/1.254=0. Min is -0.05. So 5%.
        assert metrics['sharpe_ratio'] > 0  # Exact value depends on std dev

    def test_metrics_zero_volatility_sharpe(self):
        # Case 1: Returns > risk-free rate
        portfolio_history_gt_rf = [
            {'date': '2019', 'stocks': [], 'action': 'initial_investment', 'value_bought': 0, 'value_sold': 0,
             'transaction_cost': 0},
            {'date': '2020', 'stocks': [{'symbol': 'A', 'weight': 1.0, 'annual_return': 5.0}], 'action': 'rebalance',
             'value_bought': 0, 'value_sold': 0, 'transaction_cost': 0},
            {'date': '2021', 'stocks': [{'symbol': 'A', 'weight': 1.0, 'annual_return': 5.0}], 'action': 'rebalance',
             'value_bought': 0, 'value_sold': 0, 'transaction_cost': 0}
        ]  # Periodic returns: 0.05, 0.05. ANNUAL_RISK_FREE_RATE = 0.02
        metrics_gt_rf = calculate_metrics(portfolio_history_gt_rf)
        assert metrics_gt_rf['sharpe_ratio'] == float('inf')

        # Case 2: Returns == risk-free rate
        portfolio_history_eq_rf = [
            {'date': '2019', 'stocks': [], 'action': 'initial_investment', 'value_bought': 0, 'value_sold': 0,
             'transaction_cost': 0},
            {'date': '2020', 'stocks': [{'symbol': 'A', 'weight': 1.0, 'annual_return': 2.0}], 'action': 'rebalance',
             'value_bought': 0, 'value_sold': 0, 'transaction_cost': 0},
            {'date': '2021', 'stocks': [{'symbol': 'A', 'weight': 1.0, 'annual_return': 2.0}], 'action': 'rebalance',
             'value_bought': 0, 'value_sold': 0, 'transaction_cost': 0}
        ]  # Periodic returns: 0.02, 0.02
        metrics_eq_rf = calculate_metrics(portfolio_history_eq_rf)
        assert metrics_eq_rf['sharpe_ratio'] == 0.0  # (0.02 - 0.02) / 0 = 0

        # Case 3: Returns < risk-free rate
        portfolio_history_lt_rf = [
            {'date': '2019', 'stocks': [], 'action': 'initial_investment', 'value_bought': 0, 'value_sold': 0,
             'transaction_cost': 0},
            {'date': '2020', 'stocks': [{'symbol': 'A', 'weight': 1.0, 'annual_return': 1.0}], 'action': 'rebalance',
             'value_bought': 0, 'value_sold': 0, 'transaction_cost': 0},
            {'date': '2021', 'stocks': [{'symbol': 'A', 'weight': 1.0, 'annual_return': 1.0}], 'action': 'rebalance',
             'value_bought': 0, 'value_sold': 0, 'transaction_cost': 0}
        ]  # Periodic returns: 0.01, 0.01
        metrics_lt_rf = calculate_metrics(portfolio_history_lt_rf)
        assert metrics_lt_rf['sharpe_ratio'] == 0.0  # (0.01 - 0.02) / 0, but excess return is negative.

    def test_ptr_calculation(self):
        # Scenario 1: Only initial investment, no rebalances
        history1 = [
            {'date': '2020', 'stocks': [], 'action': 'initial_investment',
             'value_bought': NOTIONAL_PORTFOLIO_VALUE_FOR_TRADES, 'value_sold': 0, 'transaction_cost': 10}
        ]
        metrics1 = calculate_metrics(history1)
        assert metrics1['turnover_ratio'] == 0.0  # Initial investment not counted in this PTR def

        # Scenario 2: One rebalance
        history2 = [
            {'date': '2020', 'stocks': [], 'action': 'initial_investment', 'value_bought': NOTIONAL_PORTFOLIO_VALUE_FOR_TRADES,
            'value_sold': 0, 'transaction_cost': 10},
        {'date': '2021', 'stocks': [], 'action': 'rebalance',
         'value_bought': 0.2 * NOTIONAL_PORTFOLIO_VALUE_FOR_TRADES,  # 200k
         'value_sold': 0.3 * NOTIONAL_PORTFOLIO_VALUE_FOR_TRADES,  # 300k
         'transaction_cost': 5}
        ]  # Turnover for event = min(200k, 300k) = 200k. PTR = (200k / 1M) * 100 = 20%
        metrics2 = calculate_metrics(history2)
        assert metrics2['turnover_ratio'] == pytest.approx(20.0)

        # Scenario 3: Multiple rebalances
        history3 = [
            {'date': '2020', 'stocks': [], 'action': 'initial_investment', 'value_bought': NOTIONAL_PORTFOLIO_VALUE_FOR_TRADES,
            'value_sold': 0, 'transaction_cost': 10},
        {'date': '2021', 'stocks': [], 'action': 'rebalance',
         'value_bought': 0.2 * NOTIONAL_PORTFOLIO_VALUE_FOR_TRADES,
         'value_sold': 0.3 * NOTIONAL_PORTFOLIO_VALUE_FOR_TRADES, 'transaction_cost': 5},  # PTR_event1 = 0.2
        {'date': '2022', 'stocks': [], 'action': 'hold', 'value_bought': 0, 'value_sold': 0,
         'transaction_cost': 0},  # No turnover
        {'date': '2023', 'stocks': [], 'action': 'rebalance',
         'value_bought': 0.4 * NOTIONAL_PORTFOLIO_VALUE_FOR_TRADES,
         'value_sold': 0.1 * NOTIONAL_PORTFOLIO_VALUE_FOR_TRADES, 'transaction_cost': 3}  # PTR_event2 = 0.1
        ]  # Avg PTR = ((0.2 + 0.1) / 2) * 100 = 15%
        metrics3 = calculate_metrics(history3)
        assert metrics3['turnover_ratio'] == pytest.approx(15.0)

    def test_transaction_cost_metric(self):
        history = [
            {'date': '2020', 'stocks': [], 'action': 'initial_investment', 'value_bought': NOTIONAL_PORTFOLIO_VALUE_FOR_TRADES,
            'value_sold': 0, 'transaction_cost': 1000},
        {'date': '2021', 'stocks': [], 'action': 'rebalance', 'value_bought': 200_000, 'value_sold': 300_000,
         'transaction_cost': 500},  # Cost for this rebalance
        {'date': '2022', 'stocks': [], 'action': 'rebalance', 'value_bought': 100_000, 'value_sold': 50_000,
         'transaction_cost': 150}  # Cost for this rebalance
        ]
        # Num rebalance events = 2
        # Total monetary costs from rebalances = 500 + 150 = 650
        # Avg cost % = (650 / (2 * 1_000_000)) * 100 = (650 / 2_000_000) * 100 = 0.000325 * 100 = 0.0325%
        metrics = calculate_metrics(history)
        assert metrics['transaction_costs_total'] == pytest.approx(0.0325)

    def test_metrics_with_cash_periods(self):
        portfolio_history = [
            {'date': '2019', 'stocks': [], 'action': 'initial_investment', 'value_bought': 100, 'value_sold': 0,
             'transaction_cost': 0.1},
            {'date': '2020', 'stocks': [{'symbol': 'A', 'weight': 1.0, 'annual_return': 10.0}],  # +10%
             'action': 'rebalance', 'value_bought': 10, 'value_sold': 0, 'transaction_cost': 0.01},
            {'date': '2021', 'stocks': [{'symbol': 'CASH', 'weight': 1.0, 'annual_return': 0.0}],  # 0% (cash)
             'action': 'rebalance', 'value_bought': 0, 'value_sold': 0, 'transaction_cost': 0},
            {'date': '2022', 'stocks': [{'symbol': 'B', 'weight': 1.0, 'annual_return': 5.0}],  # +5%
             'action': 'rebalance', 'value_bought': 20, 'value_sold': 0, 'transaction_cost': 0.02}
        ]
        # Returns: 0.10, 0.0, 0.05
        # Total growth: (1.10) * (1.0) * (1.05) = 1.155
        # Total return: 1.155 - 1 = 0.155 (15.5%)
        # CAGR: (1.155)**(1/3) - 1 = 0.049205... (4.92%)
        metrics = calculate_metrics(portfolio_history)
        assert metrics['total_return'] == pytest.approx(15.5, abs=0.01)
        assert metrics['average_annual_return'] == pytest.approx(4.92, abs=0.01)
        assert metrics['portfolio_size'] == 1  # Final portfolio has stock B


# --- Tests for load_market_data() ---
@pytest.mark.unit
class TestLoadMarketData:
    def test_load_valid_csv(self, tmp_path):
        file_path = tmp_path / "data.csv"
        data = {'col1': [1, 2], 'col2': ['a', 'b']}
        pd.DataFrame(data).to_csv(file_path, index=False)
        df = load_market_data(str(file_path))
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (2, 2)
        assert list(df.columns) == ['col1', 'col2']

    def test_load_valid_json(self, tmp_path):
        file_path = tmp_path / "data.json"
        data = [{'col1': 1, 'col2': 'a'}, {'col1': 2, 'col2': 'b'}]
        with open(file_path, 'w') as f:
            json.dump(data, f)
        df = load_market_data(str(file_path))
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (2, 2)
        assert set(df.columns) == {'col1', 'col2'}  # Order might not be preserved from list of dicts

    def test_load_file_not_found(self):
        with pytest.raises(FileNotFoundError, match="Data file not found: nonexistent.csv"):
            load_market_data("nonexistent.csv")

    def test_load_unsupported_extension(self, tmp_path):
        file_path = tmp_path / "data.txt"
        file_path.write_text("some data")
        with pytest.raises(ValueError, match="Unsupported file format: '.txt'"):
            load_market_data(str(file_path))

    def test_load_corrupt_json(self, tmp_path):
        file_path = tmp_path / "data.json"
        file_path.write_text("{'col1': 1, 'col2': 'a'")  # Malformed JSON
        with pytest.raises(ValueError, match="Error decoding JSON"):
            load_market_data(str(file_path))

    @patch('click.echo')  # To capture the warning
    def test_load_empty_csv(self, mock_click_echo, tmp_path):
        # Test CSV with headers but no data rows
        file_path_headers_only = tmp_path / "empty_with_headers.csv"
        pd.DataFrame(columns=['h1', 'h2']).to_csv(file_path_headers_only, index=False)
        df_headers_only = load_market_data(str(file_path_headers_only))
        assert df_headers_only.empty
        assert list(df_headers_only.columns) == ['h1', 'h2']
        mock_click_echo.assert_not_called()  # No warning for this case

        # Test truly empty CSV (0 bytes)
        file_path_truly_empty = tmp_path / "truly_empty.csv"
        file_path_truly_empty.write_text("")  # Creates an empty file

        df_truly_empty = load_market_data(str(file_path_truly_empty))
        assert df_truly_empty.empty
        assert list(df_truly_empty.columns) == []  # Should have no columns
        mock_click_echo.assert_any_call(
            f"Warning: CSV file {str(file_path_truly_empty)} is empty. Returning empty DataFrame.",
            err=True
        )

    def test_load_empty_json_list(self, tmp_path):
        file_path = tmp_path / "empty.json"
        with open(file_path, 'w') as f:
            json.dump([], f)  # JSON file with an empty list
        df = load_market_data(str(file_path))
        assert df.empty