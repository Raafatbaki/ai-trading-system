import unittest

import numpy as np
import pandas as pd

from src.strategies.tsmom_reference_risk_scaled import (
    calculate_target_weight,
    mark_month_end,
    build_monthly_decisions,
)


class TSMOMRiskScaledTests(unittest.TestCase):

    def test_80_percent_vol_gives_50_percent_weight(
        self,
    ):

        weight = calculate_target_weight(
            momentum_pct=10.0,
            annual_vol=0.80,
        )

        self.assertAlmostEqual(
            weight,
            0.50,
            places=10,
        )

    def test_low_vol_is_capped_at_100_percent(
        self,
    ):

        weight = calculate_target_weight(
            momentum_pct=10.0,
            annual_vol=0.20,
        )

        self.assertAlmostEqual(
            weight,
            1.0,
            places=10,
        )

    def test_negative_momentum_is_cash(
        self,
    ):

        weight = calculate_target_weight(
            momentum_pct=-5.0,
            annual_vol=0.50,
        )

        self.assertAlmostEqual(
            weight,
            0.0,
            places=10,
        )

    def test_zero_momentum_is_cash(
        self,
    ):

        weight = calculate_target_weight(
            momentum_pct=0.0,
            annual_vol=0.50,
        )

        self.assertAlmostEqual(
            weight,
            0.0,
            places=10,
        )

    def test_missing_momentum_returns_nan(
        self,
    ):

        weight = calculate_target_weight(
            momentum_pct=np.nan,
            annual_vol=0.50,
        )

        self.assertTrue(np.isnan(weight))

    def test_only_real_calendar_month_end_is_marked(
        self,
    ):

        dataframe = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    [
                        "2026-08-30",
                        "2026-08-31",
                        "2026-09-01",
                        "2026-09-29",
                    ],
                    utc=True,
                )
            }
        )

        result = mark_month_end(dataframe)

        marked_dates = (
            result[result["is_month_end"]]["timestamp"].dt.strftime("%Y-%m-%d").tolist()
        )

        self.assertEqual(
            marked_dates,
            ["2026-08-31"],
        )

    def test_first_12_months_are_warmup(
        self,
    ):

        dates = pd.date_range(
            "2023-01-01",
            "2025-01-31",
            freq="D",
            tz="UTC",
        )

        close = 100.0 * (1.0005 ** np.arange(len(dates)))

        dataframe = pd.DataFrame(
            {
                "timestamp": dates,
                "open": close,
                "high": close,
                "low": close,
                "close": close,
            }
        )

        monthly = build_monthly_decisions(dataframe)

        first_twelve = monthly.iloc[:12]

        self.assertTrue(first_twelve["momentum_12m_pct"].isna().all())

        self.assertTrue(pd.notna(monthly.iloc[12]["momentum_12m_pct"]))


if __name__ == "__main__":
    unittest.main()
