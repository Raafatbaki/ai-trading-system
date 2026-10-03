from pathlib import Path

import numpy as np
import pandas as pd

# ============================================================
# PATHS
# ============================================================

DATA_DIR = Path("data/reference_tsmom")

UNIVERSE_FILE = DATA_DIR / "universe_screen.csv"

OUTPUT_DIR = DATA_DIR / "risk_scaled_signals"

MONTHLY_DIR = DATA_DIR / "risk_scaled_monthly"

SUMMARY_FILE = OUTPUT_DIR / "tsmom_reference_risk_scaled_summary.csv"


# ============================================================
# FROZEN PARAMETERS
# ============================================================

LOOKBACK_MONTHS = 12

TARGET_VOL_ANNUAL = 0.40

EWMA_COM_DAYS = 60

ANNUALIZATION_DAYS = 365

MIN_VOL_OBSERVATIONS = 60


# ============================================================
# LOAD FROZEN UNIVERSE
# ============================================================


def load_frozen_universe() -> list[str]:

    if not UNIVERSE_FILE.exists():

        raise FileNotFoundError(f"Missing universe file: " f"{UNIVERSE_FILE}")

    dataframe = pd.read_csv(UNIVERSE_FILE)

    required = [
        "symbol",
        "eligible",
    ]

    missing = [column for column in required if column not in dataframe.columns]

    if missing:

        raise RuntimeError("Universe file missing columns: " f"{missing}")

    eligible_mask = dataframe["eligible"].astype(str).str.strip().str.lower().eq("true")

    symbols = (
        dataframe.loc[
            eligible_mask,
            "symbol",
        ]
        .astype(str)
        .tolist()
    )

    if not symbols:

        raise RuntimeError("No eligible symbols " "in frozen universe.")

    return symbols


# ============================================================
# LOAD DAILY DATA
# ============================================================


def load_data(
    symbol: str,
) -> pd.DataFrame:

    file_path = DATA_DIR / f"{symbol}_spot_1d.csv"

    if not file_path.exists():

        raise FileNotFoundError(f"Missing data file: " f"{file_path}")

    dataframe = pd.read_csv(
        file_path,
        parse_dates=["timestamp"],
    )

    dataframe = (
        dataframe.sort_values("timestamp")
        .drop_duplicates(subset=["timestamp"])
        .reset_index(drop=True)
    )

    return dataframe


# ============================================================
# EWMA VOLATILITY
# ============================================================


def add_ewma_volatility(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:

    result = dataframe.copy()

    result["daily_return"] = result["close"].pct_change(fill_method=None)

    squared_returns = result["daily_return"] ** 2

    ewma_variance_daily = squared_returns.ewm(
        com=EWMA_COM_DAYS,
        adjust=False,
        min_periods=(MIN_VOL_OBSERVATIONS),
    ).mean()

    result["ewma_vol_annual"] = np.sqrt(ewma_variance_daily * ANNUALIZATION_DAYS)

    result["ewma_vol_annual_pct"] = result["ewma_vol_annual"] * 100

    return result


# ============================================================
# POSITION SIZING
# ============================================================


def calculate_target_weight(
    momentum_pct: float,
    annual_vol: float,
) -> float:

    # No valid momentum information.
    if pd.isna(momentum_pct):
        return np.nan

    # Long/Cash:
    #
    # zero or negative momentum = CASH.
    if momentum_pct <= 0:
        return 0.0

    # Positive momentum requires a
    # valid volatility estimate.
    if pd.isna(annual_vol) or annual_vol <= 0:
        return np.nan

    raw_weight = TARGET_VOL_ANNUAL / annual_vol

    # No leverage and no shorting.
    target_weight = min(
        1.0,
        max(
            0.0,
            raw_weight,
        ),
    )

    return float(target_weight)


# ============================================================
# MONTH-END MARKING
# ============================================================


def mark_month_end(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:

    result = dataframe.copy()

    result["is_month_end"] = result["timestamp"].dt.is_month_end

    return result


# ============================================================
# MONTHLY DECISIONS
# ============================================================


def build_monthly_decisions(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:

    volatility_data = add_ewma_volatility(dataframe)

    marked = mark_month_end(volatility_data)

    monthly = marked[marked["is_month_end"]].copy().reset_index(drop=True)

    monthly["close_12m_ago"] = monthly["close"].shift(LOOKBACK_MONTHS)

    monthly["momentum_12m_pct"] = (
        (monthly["close"] / monthly["close_12m_ago"]) - 1
    ) * 100

    # --------------------------------------------------------
    # Direction
    # --------------------------------------------------------

    monthly["target_position"] = np.nan

    valid_momentum = monthly["momentum_12m_pct"].notna()

    monthly.loc[
        valid_momentum,
        "target_position",
    ] = (
        monthly.loc[
            valid_momentum,
            "momentum_12m_pct",
        ]
        > 0
    ).astype(int)

    # --------------------------------------------------------
    # Raw volatility-scaled position
    # before applying the no-leverage cap.
    # --------------------------------------------------------

    monthly["raw_target_weight"] = np.nan

    positive = (
        valid_momentum
        & (monthly["momentum_12m_pct"] > 0)
        & (monthly["ewma_vol_annual"] > 0)
    )

    negative_or_zero = valid_momentum & (monthly["momentum_12m_pct"] <= 0)

    monthly.loc[
        negative_or_zero,
        "raw_target_weight",
    ] = 0.0

    monthly.loc[
        positive,
        "raw_target_weight",
    ] = (
        TARGET_VOL_ANNUAL
        / monthly.loc[
            positive,
            "ewma_vol_annual",
        ]
    )

    # --------------------------------------------------------
    # Final target:
    #
    # 0.0 <= weight <= 1.0
    # --------------------------------------------------------

    monthly["target_weight"] = np.nan

    monthly.loc[
        negative_or_zero,
        "target_weight",
    ] = 0.0

    monthly.loc[
        positive,
        "target_weight",
    ] = monthly.loc[
        positive,
        "raw_target_weight",
    ].clip(
        lower=0.0,
        upper=1.0,
    )

    return monthly


# ============================================================
# APPLY MONTHLY TARGETS TO DAILY DATA
# ============================================================


def apply_monthly_targets(
    dataframe: pd.DataFrame,
    monthly: pd.DataFrame,
) -> pd.DataFrame:

    result = add_ewma_volatility(dataframe)

    result["momentum_12m_pct"] = np.nan

    result["target_position"] = np.nan

    result["raw_target_weight"] = np.nan

    result["target_weight"] = np.nan

    monthly_lookup = monthly.set_index("timestamp")

    columns_to_copy = [
        "momentum_12m_pct",
        "target_position",
        "raw_target_weight",
        "target_weight",
    ]

    for index, row in result.iterrows():

        timestamp = row["timestamp"]

        if timestamp not in monthly_lookup.index:
            continue

        monthly_row = monthly_lookup.loc[timestamp]

        for column in columns_to_copy:

            result.at[
                index,
                column,
            ] = monthly_row[column]

    return result


# ============================================================
# PROCESS ONE SYMBOL
# ============================================================


def process_symbol(
    symbol: str,
) -> dict:

    dataframe = load_data(symbol)

    monthly = build_monthly_decisions(dataframe)

    daily_output = apply_monthly_targets(
        dataframe,
        monthly,
    )

    valid = monthly[monthly["target_weight"].notna()].copy()

    long_decisions = valid[valid["target_position"] == 1]

    cash_decisions = valid[valid["target_position"] == 0]

    capped_decisions = long_decisions[long_decisions["target_weight"] >= (1.0 - 1e-12)]

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    MONTHLY_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    daily_file = OUTPUT_DIR / (
        f"{symbol}_" "spot_1d_" "tsmom_reference_" "risk_scaled_signals.csv"
    )

    monthly_file = MONTHLY_DIR / (
        f"{symbol}_" "tsmom_reference_" "risk_scaled_monthly.csv"
    )

    daily_output.to_csv(
        daily_file,
        index=False,
    )

    monthly.to_csv(
        monthly_file,
        index=False,
    )

    if long_decisions.empty:

        average_long_weight = np.nan
        median_long_weight = np.nan
        minimum_long_weight = np.nan
        maximum_long_weight = np.nan
        average_long_vol = np.nan

    else:

        average_long_weight = float(long_decisions["target_weight"].mean())

        median_long_weight = float(long_decisions["target_weight"].median())

        minimum_long_weight = float(long_decisions["target_weight"].min())

        maximum_long_weight = float(long_decisions["target_weight"].max())

        average_long_vol = float(long_decisions["ewma_vol_annual"].mean())

    print()
    print("=" * 110)

    print(symbol)

    print("=" * 110)

    print(f"Valid monthly decisions: " f"{len(valid)}")

    print(f"LONG decisions:          " f"{len(long_decisions)}")

    print(f"CASH decisions:          " f"{len(cash_decisions)}")

    print(f"100% capped decisions:   " f"{len(capped_decisions)}")

    if not long_decisions.empty:

        print()

        print(f"Average LONG volatility: " f"{average_long_vol * 100:.2f}%")

        print(f"Average LONG weight:     " f"{average_long_weight * 100:.2f}%")

        print(f"Median LONG weight:      " f"{median_long_weight * 100:.2f}%")

        print(f"Minimum LONG weight:     " f"{minimum_long_weight * 100:.2f}%")

        print(f"Maximum LONG weight:     " f"{maximum_long_weight * 100:.2f}%")

    print()

    print(f"Daily signals: " f"{daily_file}")

    print(f"Monthly audit: " f"{monthly_file}")

    return {
        "symbol": symbol,
        "valid_decisions": (len(valid)),
        "long_decisions": (len(long_decisions)),
        "cash_decisions": (len(cash_decisions)),
        "capped_100_decisions": (len(capped_decisions)),
        "avg_long_vol_pct": (
            average_long_vol * 100 if pd.notna(average_long_vol) else np.nan
        ),
        "avg_long_weight_pct": (
            average_long_weight * 100 if pd.notna(average_long_weight) else np.nan
        ),
        "median_long_weight_pct": (
            median_long_weight * 100 if pd.notna(median_long_weight) else np.nan
        ),
        "min_long_weight_pct": (
            minimum_long_weight * 100 if pd.notna(minimum_long_weight) else np.nan
        ),
        "max_long_weight_pct": (
            maximum_long_weight * 100 if pd.notna(maximum_long_weight) else np.nan
        ),
    }


# ============================================================
# MAIN
# ============================================================


def main():

    symbols = load_frozen_universe()

    print()
    print("=" * 110)

    print("TSMOM REFERENCE — " "RISK-SCALED SIGNAL GENERATION")

    print("=" * 110)

    print()

    print("Frozen rules:")

    print(f"  Momentum lookback:     " f"{LOOKBACK_MONTHS} months")

    print("  Decision frequency:    " "month-end")

    print("  Positive momentum:     " "LONG")

    print("  Zero/negative:         " "CASH")

    print(f"  Target annual vol:     " f"{TARGET_VOL_ANNUAL * 100:.0f}%")

    print(f"  EWMA COM:              " f"{EWMA_COM_DAYS} days")

    print(f"  Annualization:         " f"{ANNUALIZATION_DAYS} days")

    print("  Maximum weight:        " "100%")

    print("  Short selling:         " "NONE")

    print("  Leverage:              " "NONE")

    print()

    print(f"Frozen universe: " f"{len(symbols)} assets")

    summaries = []

    for symbol in symbols:

        summary = process_symbol(symbol)

        summaries.append(summary)

    summary_dataframe = pd.DataFrame(summaries)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_dataframe.to_csv(
        SUMMARY_FILE,
        index=False,
    )

    print()
    print()
    print("=" * 110)

    print("RISK-SCALED SUMMARY")

    print("=" * 110)

    print()

    print(summary_dataframe.to_string(index=False))

    print()

    print(f"Summary saved: " f"{SUMMARY_FILE}")


if __name__ == "__main__":
    main()
