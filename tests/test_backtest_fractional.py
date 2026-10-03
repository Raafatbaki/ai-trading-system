import unittest

import pandas as pd

from src.backtest_fractional import (
    calculate_rebalance,
    run_fractional_backtest,
)


class FractionalBacktestTests(unittest.TestCase):

    # ========================================================
    # HELPER
    # ========================================================

    def make_data(
        self,
        opens,
        closes,
        targets,
    ):

        length = len(opens)

        timestamps = pd.date_range(
            "2025-01-01",
            periods=length,
            freq="D",
            tz="UTC",
        )

        return pd.DataFrame(
            {
                "timestamp": (timestamps),
                "open": opens,
                "high": [
                    max(
                        open_price,
                        close_price,
                    )
                    for (
                        open_price,
                        close_price,
                    ) in zip(
                        opens,
                        closes,
                    )
                ],
                "low": [
                    min(
                        open_price,
                        close_price,
                    )
                    for (
                        open_price,
                        close_price,
                    ) in zip(
                        opens,
                        closes,
                    )
                ],
                "close": closes,
                "target_weight": targets,
            }
        )

    # ========================================================
    # 1. NEXT-CANDLE OPEN
    # ========================================================

    def test_target_executes_next_open(
        self,
    ):

        data = self.make_data(
            opens=[
                100.0,
                110.0,
                120.0,
            ],
            closes=[
                100.0,
                115.0,
                125.0,
            ],
            targets=[
                1.0,
                None,
                None,
            ],
        )

        transactions, _, _ = run_fractional_backtest(
            data,
            fee_rate=0.0,
            force_liquidation=False,
        )

        self.assertEqual(
            len(transactions),
            1,
        )

        trade = transactions.iloc[0]

        self.assertEqual(
            trade["signal_timestamp"],
            data.iloc[0]["timestamp"],
        )

        self.assertEqual(
            trade["execution_timestamp"],
            data.iloc[1]["timestamp"],
        )

        self.assertAlmostEqual(
            trade["execution_price"],
            110.0,
            places=10,
        )

    # ========================================================
    # 2. EXACT 50% WEIGHT WITH FEES
    # ========================================================

    def test_exact_half_weight_with_fee(
        self,
    ):

        result = calculate_rebalance(
            cash=100.0,
            quantity=0.0,
            price=100.0,
            target_weight=0.5,
            fee_rate=0.001,
        )

        self.assertEqual(
            result["side"],
            "BUY",
        )

        self.assertAlmostEqual(
            result["weight_after"],
            0.5,
            places=10,
        )

        self.assertGreater(
            result["fee"],
            0,
        )

    # ========================================================
    # 3. EXACT 100% WEIGHT WITH FEES
    # ========================================================

    def test_full_weight_with_fee(
        self,
    ):

        result = calculate_rebalance(
            cash=100.0,
            quantity=0.0,
            price=100.0,
            target_weight=1.0,
            fee_rate=0.001,
        )

        self.assertAlmostEqual(
            result["weight_after"],
            1.0,
            places=10,
        )

        self.assertAlmostEqual(
            result["cash"],
            0.0,
            places=10,
        )

    # ========================================================
    # 4. PARTIAL SELL
    # ========================================================

    def test_partial_sell_rebalances_exactly(
        self,
    ):

        # Start:
        # 50 cash
        # 50 asset
        #
        # Target becomes 25%.

        result = calculate_rebalance(
            cash=50.0,
            quantity=0.5,
            price=100.0,
            target_weight=0.25,
            fee_rate=0.001,
        )

        self.assertEqual(
            result["side"],
            "SELL",
        )

        self.assertAlmostEqual(
            result["weight_after"],
            0.25,
            places=10,
        )

    # ========================================================
    # 5. ZERO WEIGHT = FULL EXIT
    # ========================================================

    def test_zero_weight_sells_entire_position(
        self,
    ):

        result = calculate_rebalance(
            cash=20.0,
            quantity=0.8,
            price=100.0,
            target_weight=0.0,
            fee_rate=0.001,
        )

        self.assertEqual(
            result["side"],
            "SELL",
        )

        self.assertAlmostEqual(
            result["quantity"],
            0.0,
            places=10,
        )

        self.assertAlmostEqual(
            result["weight_after"],
            0.0,
            places=10,
        )

    # ========================================================
    # 6. INVALID WEIGHT
    # ========================================================

    def test_invalid_target_weight_rejected(
        self,
    ):

        data = self.make_data(
            opens=[
                100.0,
                100.0,
            ],
            closes=[
                100.0,
                100.0,
            ],
            targets=[
                1.2,
                None,
            ],
        )

        with self.assertRaises(ValueError):

            run_fractional_backtest(data)

    # ========================================================
    # 7. LAST-CANDLE TARGET MUST NOT EXECUTE
    # ========================================================

    def test_last_candle_target_is_ignored(
        self,
    ):

        data = self.make_data(
            opens=[
                100.0,
                100.0,
                100.0,
            ],
            closes=[
                100.0,
                100.0,
                100.0,
            ],
            targets=[
                None,
                None,
                1.0,
            ],
        )

        transactions, _, stats = run_fractional_backtest(data)

        self.assertTrue(transactions.empty)

        self.assertAlmostEqual(
            stats["final_capital"],
            100.0,
            places=10,
        )

    # ========================================================
    # 8. MARK-TO-MARKET DRAWDOWN
    # ========================================================

    def test_mark_to_market_drawdown(
        self,
    ):

        data = self.make_data(
            opens=[
                100.0,
                100.0,
                100.0,
                100.0,
            ],
            closes=[
                100.0,
                100.0,
                50.0,
                100.0,
            ],
            targets=[
                1.0,
                None,
                None,
                None,
            ],
        )

        _, _, stats = run_fractional_backtest(
            data,
            fee_rate=0.0,
            force_liquidation=False,
        )

        self.assertAlmostEqual(
            stats["max_drawdown"],
            -50.0,
            places=10,
        )

    # ========================================================
    # 9. FORCED FINAL LIQUIDATION
    # ========================================================

    def test_forced_final_liquidation(
        self,
    ):

        data = self.make_data(
            opens=[
                100.0,
                100.0,
                100.0,
            ],
            closes=[
                100.0,
                100.0,
                120.0,
            ],
            targets=[
                1.0,
                None,
                None,
            ],
        )

        transactions, _, stats = run_fractional_backtest(
            data,
            fee_rate=0.001,
            force_liquidation=True,
        )

        self.assertEqual(
            stats["forced_exits"],
            1,
        )

        self.assertTrue(bool(transactions.iloc[-1]["forced"]))

        self.assertEqual(
            transactions.iloc[-1]["side"],
            "SELL",
        )

        self.assertAlmostEqual(
            transactions.iloc[-1]["execution_price"],
            120.0,
            places=10,
        )

        self.assertGreater(
            stats["final_capital"],
            100.0,
        )


if __name__ == "__main__":
    unittest.main()
