from pathlib import Path

import numpy as np
import pandas as pd

# ============================================================
# PATHS
# ============================================================

DATA_DIR = Path("data/reference_tsmom")

OUTPUT_DIR = DATA_DIR / "robustness"

CALENDAR_OUTPUT = OUTPUT_DIR / "calendar_year_analysis.csv"

ROLLING_OUTPUT = OUTPUT_DIR / "rolling_12m_analysis.csv"

SUMMARY_OUTPUT = OUTPUT_DIR / "portfolio_robustness_summary.csv"


# ============================================================
# PORTFOLIOS
# ============================================================

PORTFOLIOS = {
    "Risk-Scaled Momentum": (
        DATA_DIR / "portfolio" / "results" / "tsmom_reference_portfolio_equity.csv"
    ),
    "Equal-Weight Momentum": (
        DATA_DIR / "benchmarks" / "equal_weight_momentum_results" / "equity.csv"
    ),
    "Equal-Weight Market": (
        DATA_DIR / "benchmarks" / "equal_weight_market_results" / "equity.csv"
    ),
}


# ============================================================
# LOAD EQUITY
# ============================================================


def load_equity(
    name: str,
    file_path: Path,
) -> pd.DataFrame:

    if not file_path.exists():

        raise FileNotFoundError(f"{name}: missing equity file: " f"{file_path}")

    dataframe = pd.read_csv(
        file_path,
        parse_dates=["timestamp"],
    )

    required = [
        "timestamp",
        "equity",
        "portfolio_exposure",
    ]

    missing = [column for column in required if column not in dataframe.columns]

    if missing:

        raise RuntimeError(f"{name}: missing columns " f"{missing}")

    dataframe = (
        dataframe.sort_values("timestamp")
        .drop_duplicates(subset=["timestamp"])
        .reset_index(drop=True)
    )

    dataframe["equity"] = pd.to_numeric(
        dataframe["equity"],
        errors="raise",
    )

    dataframe["portfolio_exposure"] = pd.to_numeric(
        dataframe["portfolio_exposure"],
        errors="raise",
    )

    return dataframe


# ============================================================
# CALENDAR-YEAR ANALYSIS
# ============================================================


def analyze_calendar_years(
    name: str,
    dataframe: pd.DataFrame,
) -> list[dict]:

    result = []

    dataframe = dataframe.copy()

    dataframe["year"] = dataframe["timestamp"].dt.year

    years = sorted(dataframe["year"].unique())

    first_timestamp = dataframe.iloc[0]["timestamp"]

    last_timestamp = dataframe.iloc[-1]["timestamp"]

    for year in years:

        year_data = dataframe[dataframe["year"] == year].copy()

        first_index = year_data.index[0]

        # ----------------------------------------------------
        # Starting equity:
        #
        # use previous year's final equity where available.
        # For the first partial year, use the first observation.
        # ----------------------------------------------------

        if first_index > 0:

            starting_equity = float(
                dataframe.loc[
                    first_index - 1,
                    "equity",
                ]
            )

        else:

            starting_equity = float(year_data.iloc[0]["equity"])

        ending_equity = float(year_data.iloc[-1]["equity"])

        return_pct = ((ending_equity / starting_equity) - 1.0) * 100.0

        # ----------------------------------------------------
        # Within-year drawdown.
        #
        # Include previous year-end equity as the starting peak.
        # ----------------------------------------------------

        equity_for_dd = year_data["equity"].copy()

        if first_index > 0:

            equity_for_dd = pd.concat(
                [
                    pd.Series([starting_equity]),
                    equity_for_dd.reset_index(drop=True),
                ],
                ignore_index=True,
            )

        running_peak = equity_for_dd.cummax()

        drawdown = ((equity_for_dd / running_peak) - 1.0) * 100.0

        max_dd = float(drawdown.min())

        average_exposure = float(year_data["portfolio_exposure"].mean() * 100.0)

        invested_days_pct = float(
            (year_data["portfolio_exposure"] > 1e-12).mean() * 100.0
        )

        first_day = year_data.iloc[0]["timestamp"]

        last_day = year_data.iloc[-1]["timestamp"]

        partial_year = (
            year == first_timestamp.year
            and (first_timestamp.month != 1 or first_timestamp.day != 1)
        ) or (
            year == last_timestamp.year
            and (last_timestamp.month != 12 or last_timestamp.day != 31)
        )

        result.append(
            {
                "portfolio": name,
                "year": int(year),
                "partial_year": (partial_year),
                "from": (first_day),
                "to": (last_day),
                "starting_equity": (starting_equity),
                "ending_equity": (ending_equity),
                "return_pct": (return_pct),
                "max_drawdown_pct": (max_dd),
                "average_exposure_pct": (average_exposure),
                "invested_days_pct": (invested_days_pct),
            }
        )

    return result


# ============================================================
# ROLLING 365-DAY ANALYSIS
# ============================================================


def analyze_rolling_12m(
    name: str,
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:

    result = dataframe[
        [
            "timestamp",
            "equity",
        ]
    ].copy()

    # Daily crypto data:
    # 365 observations ~= 12 months.

    result["rolling_365d_return_pct"] = (
        result["equity"].pct_change(
            periods=365,
            fill_method=None,
        )
        * 100.0
    )

    valid = result["rolling_365d_return_pct"].dropna()

    if valid.empty:

        raise RuntimeError(f"{name}: not enough data " "for 365-day analysis.")

    positive_windows_pct = float((valid > 0).mean() * 100.0)

    summary = {
        "portfolio": name,
        "rolling_windows": (len(valid)),
        "positive_12m_windows_pct": (positive_windows_pct),
        "worst_12m_return_pct": float(valid.min()),
        "median_12m_return_pct": float(valid.median()),
        "average_12m_return_pct": float(valid.mean()),
        "best_12m_return_pct": float(valid.max()),
    }

    result["portfolio"] = name

    return (
        result,
        summary,
    )


# ============================================================
# MAIN
# ============================================================


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    calendar_rows = []
    rolling_tables = []
    robustness_rows = []

    loaded = {}

    for (
        name,
        file_path,
    ) in PORTFOLIOS.items():

        dataframe = load_equity(
            name,
            file_path,
        )

        loaded[name] = dataframe

        calendar_rows.extend(
            analyze_calendar_years(
                name,
                dataframe,
            )
        )

        (
            rolling_table,
            rolling_summary,
        ) = analyze_rolling_12m(
            name,
            dataframe,
        )

        rolling_tables.append(rolling_table)

        robustness_rows.append(rolling_summary)

    # ========================================================
    # SAVE
    # ========================================================

    calendar = pd.DataFrame(calendar_rows)

    rolling = pd.concat(
        rolling_tables,
        ignore_index=True,
    )

    robustness = pd.DataFrame(robustness_rows)

    calendar.to_csv(
        CALENDAR_OUTPUT,
        index=False,
    )

    rolling.to_csv(
        ROLLING_OUTPUT,
        index=False,
    )

    robustness.to_csv(
        SUMMARY_OUTPUT,
        index=False,
    )

    # ========================================================
    # PRINT CALENDAR RETURNS
    # ========================================================

    print()
    print("=" * 110)

    print("TSMOM PORTFOLIO — " "TIME ROBUSTNESS AUDIT")

    print("=" * 110)

    print()

    print("CALENDAR / PARTIAL-YEAR RETURNS")

    print("-" * 110)

    returns_pivot = calendar.pivot(
        index="year",
        columns="portfolio",
        values="return_pct",
    )

    print(returns_pivot.to_string(float_format=lambda x: (f"{x:8.2f}%")))

    # ========================================================
    # PRINT YEARLY DRAWDOWN
    # ========================================================

    print()
    print("CALENDAR / PARTIAL-YEAR MAX DRAWDOWN")

    print("-" * 110)

    dd_pivot = calendar.pivot(
        index="year",
        columns="portfolio",
        values="max_drawdown_pct",
    )

    print(dd_pivot.to_string(float_format=lambda x: (f"{x:8.2f}%")))

    # ========================================================
    # ROLLING 12M
    # ========================================================

    print()
    print("ROLLING 365-DAY ROBUSTNESS")

    print("-" * 110)

    display_columns = [
        "portfolio",
        "rolling_windows",
        "positive_12m_windows_pct",
        "worst_12m_return_pct",
        "median_12m_return_pct",
        "average_12m_return_pct",
        "best_12m_return_pct",
    ]

    print(
        robustness[display_columns].to_string(
            index=False,
            float_format=lambda x: (f"{x:.2f}"),
        )
    )

    # ========================================================
    # RISK-SCALED SPECIFIC AUDIT
    # ========================================================

    risk_calendar = calendar[calendar["portfolio"] == "Risk-Scaled Momentum"].copy()

    profitable_periods = int((risk_calendar["return_pct"] > 0).sum())

    losing_periods = int((risk_calendar["return_pct"] < 0).sum())

    print()
    print("RISK-SCALED PERIOD AUDIT")

    print("-" * 110)

    print(f"Profitable calendar/partial years: " f"{profitable_periods}")

    print(f"Losing calendar/partial years:     " f"{losing_periods}")

    print()

    print("NOTE:")

    print("The first and last calendar years " "may be partial periods.")

    print()

    print(f"Calendar file: " f"{CALENDAR_OUTPUT}")

    print(f"Rolling file:  " f"{ROLLING_OUTPUT}")

    print(f"Summary file:  " f"{SUMMARY_OUTPUT}")


if __name__ == "__main__":
    main()
