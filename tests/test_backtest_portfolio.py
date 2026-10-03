import unittest

import pandas as pd

from src.backtest_portfolio import (
    calculate_exact_rebalance,
    run_portfolio_backtest,
)


class PortfolioBacktestTests(unittest.TestCase):

    # ========================================================
    # HELPERS
    # ========================================================

    def make_prices(
        self,
        opens,
        closes,
    ):

        timestamps = pd.date_range(
            "2025-01-01",
            periods=len(opens),
            freq="D",
            tz="UTC",
        )

        return pd.DataFrame(
            {
                "timestamp": (timestamps),
                "open": (opens),
                "close": (closes),
            }
        )

    def make_weights(
        self,
        rows,
    ):

        return pd.DataFrame(rows)

    # ========================================================
    # 1. NEXT OPEN
    # ========================================================

    def test_portfolio_executes_next_open(
        self,
    ):

        prices = {
            "AAA": self.make_prices(
                [
                    100.0,
                    110.0,
                    120.0,
                ],
                [
                    100.0,
                    110.0,
                    120.0,
                ],
            )
        }

        weights = self.make_weights(
            [
                {
                    "timestamp": pd.Timestamp(
                        "2025-01-01",
                        tz="UTC",
                    ),
                    "AAA_weight": 1.0,
                }
            ]
        )

        transactions, _, _ = run_portfolio_backtest(
            prices,
            weights,
            fee_rate=0.0,
            force_liquidation=False,
        )

        self.assertEqual(
            len(transactions),
            1,
        )

        trade = transactions.iloc[0]

        self.assertEqual(
            trade["execution_timestamp"],
            pd.Timestamp(
                "2025-01-02",
                tz="UTC",
            ),
        )

        self.assertAlmostEqual(
            trade["notional"],
            100.0,
            places=10,
        )

    # ========================================================
    # 2. EXACT 50 / 50 WITH FEES
    # ========================================================

    def test_exact_two_asset_weights_with_fees(
        self,
    ):

        result = calculate_exact_rebalance(
            cash=100.0,
            quantities={
                "AAA": 0.0,
                "BBB": 0.0,
            },
            prices={
                "AAA": 100.0,
                "BBB": 200.0,
            },
            target_weights={
                "AAA": 0.5,
                "BBB": 0.5,
            },
            fee_rate=0.001,
        )

        equity = result["equity_after"]

        aaa_value = result["quantities"]["AAA"] * 100.0

        bbb_value = result["quantities"]["BBB"] * 200.0

        self.assertAlmostEqual(
            aaa_value / equity,
            0.5,
            places=10,
        )

        self.assertAlmostEqual(
            bbb_value / equity,
            0.5,
            places=10,
        )

        self.assertAlmostEqual(
            result["cash"],
            0.0,
            places=10,
        )

    # ========================================================
    # 3. CASH REMAINDER
    # ========================================================

    def test_cash_remainder_is_preserved(
        self,
    ):

        result = calculate_exact_rebalance(
            cash=100.0,
            quantities={
                "AAA": 0.0,
                "BBB": 0.0,
            },
            prices={
                "AAA": 100.0,
                "BBB": 100.0,
            },
            target_weights={
                "AAA": 0.3,
                "BBB": 0.2,
            },
            fee_rate=0.0,
        )

        equity = result["equity_after"]

        self.assertAlmostEqual(
            result["cash"] / equity,
            0.5,
            places=10,
        )

    # ========================================================
    # 4. LEVERAGE REJECTED
    # ========================================================

    def test_weights_above_100_percent_rejected(
        self,
    ):

        prices = {
            "AAA": self.make_prices(
                [
                    100.0,
                    100.0,
                ],
                [
                    100.0,
                    100.0,
                ],
            ),
            "BBB": self.make_prices(
                [
                    100.0,
                    100.0,
                ],
                [
                    100.0,
                    100.0,
                ],
            ),
        }

        weights = pd.DataFrame(
            [
                {
                    "timestamp": pd.Timestamp(
                        "2025-01-01",
                        tz="UTC",
                    ),
                    "AAA_weight": 0.8,
                    "BBB_weight": 0.5,
                }
            ]
        )

        with self.assertRaises(ValueError):

            run_portfolio_backtest(
                prices,
                weights,
            )

    # ========================================================
    # 5. CROSS-ASSET REBALANCE
    # ========================================================

    def test_rebalance_between_assets(
        self,
    ):

        prices = {
            "AAA": self.make_prices(
                [
                    100.0,
                    100.0,
                    100.0,
                    100.0,
                ],
                [
                    100.0,
                    100.0,
                    100.0,
                    100.0,
                ],
            ),
            "BBB": self.make_prices(
                [
                    100.0,
                    100.0,
                    100.0,
                    100.0,
                ],
                [
                    100.0,
                    100.0,
                    100.0,
                    100.0,
                ],
            ),
        }

        weights = pd.DataFrame(
            [
                {
                    "timestamp": pd.Timestamp(
                        "2025-01-01",
                        tz="UTC",
                    ),
                    "AAA_weight": 1.0,
                    "BBB_weight": 0.0,
                },
                {
                    "timestamp": pd.Timestamp(
                        "2025-01-02",
                        tz="UTC",
                    ),
                    "AAA_weight": 0.0,
                    "BBB_weight": 1.0,
                },
            ]
        )

        transactions, _, _ = run_portfolio_backtest(
            prices,
            weights,
            fee_rate=0.0,
            force_liquidation=False,
        )

        final_execution = transactions[
            transactions["execution_timestamp"]
            == pd.Timestamp(
                "2025-01-03",
                tz="UTC",
            )
        ]

        sides = {
            (
                row["symbol"],
                row["side"],
            )
            for _, row in final_execution.iterrows()
        }

        self.assertIn(
            (
                "AAA",
                "SELL",
            ),
            sides,
        )

        self.assertIn(
            (
                "BBB",
                "BUY",
            ),
            sides,
        )

    # ========================================================
    # 6. MARK TO MARKET DRAWDOWN
    # ========================================================

    def test_portfolio_mark_to_market_drawdown(
        self,
    ):

        prices = {
            "AAA": self.make_prices(
                [
                    100.0,
                    100.0,
                    100.0,
                ],
                [
                    100.0,
                    100.0,
                    50.0,
                ],
            )
        }

        weights = pd.DataFrame(
            [
                {
                    "timestamp": pd.Timestamp(
                        "2025-01-01",
                        tz="UTC",
                    ),
                    "AAA_weight": 1.0,
                }
            ]
        )

        _, _, stats = run_portfolio_backtest(
            prices,
            weights,
            fee_rate=0.0,
            force_liquidation=False,
        )

        self.assertAlmostEqual(
            stats["max_drawdown"],
            -50.0,
            places=10,
        )

    # ========================================================
    # 7. LAST DECISION HAS NO NEXT-DAY EXECUTION
    # ========================================================

    def test_last_day_decision_is_ignored(
        self,
    ):

        prices = {
            "AAA": self.make_prices(
                [
                    100.0,
                    100.0,
                ],
                [
                    100.0,
                    100.0,
                ],
            )
        }

        weights = pd.DataFrame(
            [
                {
                    "timestamp": pd.Timestamp(
                        "2025-01-02",
                        tz="UTC",
                    ),
                    "AAA_weight": 1.0,
                }
            ]
        )

        transactions, _, stats = run_portfolio_backtest(
            prices,
            weights,
        )

        self.assertTrue(transactions.empty)

        self.assertAlmostEqual(
            stats["final_capital"],
            100.0,
            places=10,
        )

    # ========================================================
    # 8. FORCED LIQUIDATION
    # ========================================================

    def test_final_positions_are_liquidated(
        self,
    ):

        prices = {
            "AAA": self.make_prices(
                [
                    100.0,
                    100.0,
                    100.0,
                ],
                [
                    100.0,
                    100.0,
                    120.0,
                ],
            )
        }

        weights = pd.DataFrame(
            [
                {
                    "timestamp": pd.Timestamp(
                        "2025-01-01",
                        tz="UTC",
                    ),
                    "AAA_weight": 1.0,
                }
            ]
        )

        transactions, _, stats = run_portfolio_backtest(
            prices,
            weights,
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

        self.assertGreater(
            stats["final_capital"],
            100.0,
        )


if __name__ == "__main__":
    unittest.main()
