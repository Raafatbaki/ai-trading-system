import sys
import unittest
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(
    0,
    str(SRC_DIR),
)


from backtest import (  # noqa: E402
    FEE_RATE,
    INITIAL_CAPITAL,
    SLIPPAGE_RATE,
    calculate_buy_and_hold,
    run_backtest,
)


def make_dataframe(
    candles: list[tuple[float, float, str]],
) -> pd.DataFrame:
    """
    candles:
        [
            (open, close, signal),
            ...
        ]
    """

    timestamps = pd.date_range(
        start="2026-01-01 00:00:00",
        periods=len(candles),
        freq="1h",
        tz="UTC",
    )

    rows = []

    for timestamp, candle in zip(
        timestamps,
        candles,
    ):
        open_price, close_price, signal = candle

        rows.append(
            {
                "timestamp": timestamp,
                "open": float(open_price),
                "high": float(
                    max(
                        open_price,
                        close_price,
                    )
                    + 1
                ),
                "low": float(
                    min(
                        open_price,
                        close_price,
                    )
                    - 1
                ),
                "close": float(close_price),
                "signal": signal,
            }
        )

    return pd.DataFrame(rows)


class BacktestAuditTests(unittest.TestCase):

    def test_current_cost_configuration(
        self,
    ):
        self.assertEqual(
            INITIAL_CAPITAL,
            100.0,
        )

        self.assertEqual(
            FEE_RATE,
            0.001,
        )

        self.assertEqual(
            SLIPPAGE_RATE,
            0.0,
        )

    def test_buy_and_exit_execute_on_next_open(
        self,
    ):
        dataframe = make_dataframe(
            [
                # Signal BUY here.
                # Entry must NOT be 100 or 101.
                (100, 101, "BUY"),
                # Actual entry = 110.
                (110, 120, "HOLD"),
                # EXIT signal here.
                # Exit must NOT use 124.
                (125, 124, "EXIT"),
                # Actual exit = 130.
                (130, 130, "NO SIGNAL"),
            ]
        )

        (
            trades,
            equity,
            stats,
        ) = run_backtest(dataframe)

        self.assertEqual(
            len(trades),
            1,
        )

        trade = trades.iloc[0]

        self.assertAlmostEqual(
            trade["entry_price"],
            110.0,
            places=8,
        )

        self.assertAlmostEqual(
            trade["exit_price"],
            130.0,
            places=8,
        )

        self.assertEqual(
            trade["entry_signal_time"],
            dataframe.iloc[0]["timestamp"],
        )

        self.assertEqual(
            trade["entry_time"],
            dataframe.iloc[1]["timestamp"],
        )

        self.assertEqual(
            trade["exit_signal_time"],
            dataframe.iloc[2]["timestamp"],
        )

        self.assertEqual(
            trade["exit_time"],
            dataframe.iloc[3]["timestamp"],
        )

        # Manual fee calculation.
        entry_fee = INITIAL_CAPITAL * FEE_RATE

        investable = INITIAL_CAPITAL - entry_fee

        quantity = investable / 110.0

        gross_exit_value = quantity * 130.0

        exit_fee = gross_exit_value * FEE_RATE

        expected_final_capital = gross_exit_value - exit_fee

        expected_total_fees = entry_fee + exit_fee

        self.assertAlmostEqual(
            stats["final_capital"],
            expected_final_capital,
            places=8,
        )

        self.assertAlmostEqual(
            stats["total_fees"],
            expected_total_fees,
            places=8,
        )

        self.assertAlmostEqual(
            stats["total_return"],
            (expected_final_capital / INITIAL_CAPITAL - 1) * 100,
            places=8,
        )

        # Position is open during candles
        # index 1 and 2 only.
        self.assertAlmostEqual(
            stats["exposure_pct"],
            50.0,
            places=8,
        )

        self.assertFalse(bool(trade["forced_exit"]))

    def test_mark_to_market_drawdown(
        self,
    ):
        dataframe = make_dataframe(
            [
                (100, 100, "BUY"),
                # Entry at 100 open.
                # Then price falls to 80 close.
                (100, 80, "HOLD"),
                # Recovery while still holding.
                (80, 120, "EXIT"),
                # Exit at 120 open.
                (120, 120, "NO SIGNAL"),
            ]
        )

        (
            trades,
            equity,
            stats,
        ) = run_backtest(dataframe)

        self.assertEqual(
            len(trades),
            1,
        )

        entry_fee = INITIAL_CAPITAL * FEE_RATE

        quantity = (INITIAL_CAPITAL - entry_fee) / 100.0

        expected_equity_at_80 = quantity * 80.0

        self.assertAlmostEqual(
            equity.iloc[1]["equity"],
            expected_equity_at_80,
            places=8,
        )

        expected_drawdown = ((expected_equity_at_80 / INITIAL_CAPITAL) - 1) * 100

        self.assertAlmostEqual(
            expected_drawdown,
            -20.08,
            places=8,
        )

        self.assertAlmostEqual(
            stats["max_drawdown"],
            expected_drawdown,
            places=8,
        )

    def test_forced_exit_uses_last_close(
        self,
    ):
        dataframe = make_dataframe(
            [
                (100, 100, "BUY"),
                # Entry at 100.
                (100, 105, "HOLD"),
                # No EXIT signal.
                # Position must be force-closed
                # at final CLOSE = 110.
                (108, 110, "HOLD"),
            ]
        )

        (
            trades,
            equity,
            stats,
        ) = run_backtest(dataframe)

        self.assertEqual(
            len(trades),
            1,
        )

        trade = trades.iloc[0]

        self.assertTrue(bool(trade["forced_exit"]))

        self.assertEqual(
            stats["forced_exits"],
            1,
        )

        self.assertAlmostEqual(
            trade["entry_price"],
            100.0,
            places=8,
        )

        self.assertAlmostEqual(
            trade["exit_price"],
            110.0,
            places=8,
        )

        self.assertEqual(
            trade["exit_time"],
            dataframe.iloc[-1]["timestamp"],
        )

        self.assertTrue(pd.isna(trade["exit_signal_time"]))

        entry_fee = 100.0 * FEE_RATE

        quantity = (100.0 - entry_fee) / 100.0

        gross_exit_value = quantity * 110.0

        exit_fee = gross_exit_value * FEE_RATE

        expected_final = gross_exit_value - exit_fee

        self.assertAlmostEqual(
            stats["final_capital"],
            expected_final,
            places=8,
        )

        # Final equity must include
        # the forced-exit fee.
        self.assertAlmostEqual(
            equity.iloc[-1]["equity"],
            expected_final,
            places=8,
        )

    def test_last_candle_buy_is_not_executed(
        self,
    ):
        dataframe = make_dataframe(
            [
                (
                    100,
                    100,
                    "NO SIGNAL",
                ),
                (
                    100,
                    100,
                    "BUY",
                ),
            ]
        )

        (
            trades,
            equity,
            stats,
        ) = run_backtest(dataframe)

        self.assertEqual(
            len(trades),
            0,
        )

        self.assertEqual(
            stats["trades"],
            0,
        )

        self.assertAlmostEqual(
            stats["final_capital"],
            INITIAL_CAPITAL,
            places=8,
        )

        self.assertAlmostEqual(
            stats["total_return"],
            0.0,
            places=8,
        )

        self.assertAlmostEqual(
            stats["exposure_pct"],
            0.0,
            places=8,
        )

    def test_exit_while_cash_and_duplicate_buy_are_ignored(
        self,
    ):
        dataframe = make_dataframe(
            [
                # EXIT while CASH:
                # must do nothing.
                (100, 100, "EXIT"),
                # BUY signal.
                (100, 100, "BUY"),
                # Entry happens here.
                # Another BUY must not create
                # another position.
                (100, 105, "BUY"),
                # EXIT signal.
                (105, 108, "EXIT"),
                # Actual exit here at 110.
                (110, 110, "NO SIGNAL"),
            ]
        )

        (
            trades,
            equity,
            stats,
        ) = run_backtest(dataframe)

        self.assertEqual(
            len(trades),
            1,
        )

        trade = trades.iloc[0]

        self.assertAlmostEqual(
            trade["entry_price"],
            100.0,
            places=8,
        )

        self.assertAlmostEqual(
            trade["exit_price"],
            110.0,
            places=8,
        )

        self.assertEqual(
            trade["entry_signal_time"],
            dataframe.iloc[1]["timestamp"],
        )

        self.assertEqual(
            trade["exit_signal_time"],
            dataframe.iloc[3]["timestamp"],
        )

    def test_buy_and_hold_fee_calculation(
        self,
    ):
        dataframe = make_dataframe(
            [
                (
                    100,
                    100,
                    "NO SIGNAL",
                ),
                (
                    105,
                    110,
                    "NO SIGNAL",
                ),
            ]
        )

        actual_return = calculate_buy_and_hold(dataframe)

        entry_fee = INITIAL_CAPITAL * FEE_RATE

        investable = INITIAL_CAPITAL - entry_fee

        quantity = investable / 100.0

        gross_final_value = quantity * 110.0

        exit_fee = gross_final_value * FEE_RATE

        expected_final_value = gross_final_value - exit_fee

        expected_return = ((expected_final_value / INITIAL_CAPITAL) - 1) * 100

        self.assertAlmostEqual(
            actual_return,
            expected_return,
            places=8,
        )

        self.assertAlmostEqual(
            expected_return,
            9.78011,
            places=5,
        )


if __name__ == "__main__":
    unittest.main()
