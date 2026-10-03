from pathlib import Path

import pandas as pd

# ============================================================
# PATHS
# ============================================================

DATA_DIR = Path("data/reference_tsmom")

OUTPUT_DIR = DATA_DIR / "signals"

UNIVERSE_FILE = DATA_DIR / "universe_screen.csv"

SUMMARY_FILE = OUTPUT_DIR / "tsmom_reference_signal_summary.csv"


# ============================================================
# FROZEN STRATEGY PARAMETERS
# ============================================================

LOOKBACK_MONTHS = 12


# ============================================================
# LOAD FROZEN UNIVERSE
# ============================================================


def load_frozen_universe() -> list[str]:

    if not UNIVERSE_FILE.exists():

        raise FileNotFoundError(f"Missing frozen universe file: " f"{UNIVERSE_FILE}")

    dataframe = pd.read_csv(UNIVERSE_FILE)

    required_columns = [
        "symbol",
        "eligible",
    ]

    missing = [column for column in required_columns if column not in dataframe.columns]

    if missing:

        raise RuntimeError("Universe file missing columns: " f"{missing}")

    # Robust handling whether pandas reads
    # eligible as bool or string.
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

        raise RuntimeError("Frozen universe contains " "no eligible symbols.")

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
# REAL CALENDAR MONTH-END
# ============================================================


def mark_month_end(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:

    result = dataframe.copy()

    # Only completed calendar months are valid.
    #
    # Example:
    # 2026-09-29 is NOT treated as September month-end.

    result["is_month_end"] = result["timestamp"].dt.is_month_end

    return result


# ============================================================
# MONTHLY 12-MONTH MOMENTUM
# ============================================================


def build_monthly_signal_table(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:

    marked = mark_month_end(dataframe)

    monthly = marked[marked["is_month_end"]].copy().reset_index(drop=True)

    # --------------------------------------------------------
    # Reference TSMOM:
    #
    # compare current month-end close with
    # month-end close exactly 12 observations earlier.
    # --------------------------------------------------------

    monthly["close_12m_ago"] = monthly["close"].shift(LOOKBACK_MONTHS)

    monthly["momentum_12m_pct"] = (
        (monthly["close"] / monthly["close_12m_ago"]) - 1
    ) * 100

    # --------------------------------------------------------
    # Spot Long/Cash adaptation
    #
    # <NA> = insufficient 12-month history
    # 1    = LONG
    # 0    = CASH
    # --------------------------------------------------------

    monthly["target_position"] = pd.Series(
        pd.NA,
        index=monthly.index,
        dtype="Int64",
    )

    valid = monthly["momentum_12m_pct"].notna()

    monthly.loc[
        valid,
        "target_position",
    ] = (
        monthly.loc[
            valid,
            "momentum_12m_pct",
        ]
        > 0
    ).astype(int)

    return monthly[
        [
            "timestamp",
            "close",
            "close_12m_ago",
            "momentum_12m_pct",
            "target_position",
        ]
    ]


# ============================================================
# APPLY MONTHLY DECISIONS TO DAILY DATA
# ============================================================


def apply_signals_to_daily_data(
    dataframe: pd.DataFrame,
    monthly: pd.DataFrame,
) -> pd.DataFrame:

    result = dataframe.copy()

    result["momentum_12m_pct"] = pd.NA

    result["target_position"] = pd.NA

    # Existing backtester contract:
    #
    # BUY
    # EXIT
    # HOLD

    result["signal"] = "HOLD"

    monthly_lookup = monthly.set_index("timestamp")

    # Strategy starts in CASH.
    previous_target = 0

    for index, row in result.iterrows():

        timestamp = row["timestamp"]

        if timestamp not in monthly_lookup.index:
            continue

        monthly_row = monthly_lookup.loc[timestamp]

        momentum = monthly_row["momentum_12m_pct"]

        target = monthly_row["target_position"]

        result.at[
            index,
            "momentum_12m_pct",
        ] = momentum

        result.at[
            index,
            "target_position",
        ] = target

        if pd.isna(target):
            continue

        target = int(target)

        # CASH -> LONG
        if previous_target == 0 and target == 1:

            result.at[
                index,
                "signal",
            ] = "BUY"

        # LONG -> CASH
        elif previous_target == 1 and target == 0:

            result.at[
                index,
                "signal",
            ] = "EXIT"

        previous_target = target

    return result


# ============================================================
# PRINT SIGNAL TRANSITIONS
# ============================================================


def print_signal_transitions(
    signals: pd.DataFrame,
):

    transitions = signals[
        signals["signal"].isin(
            [
                "BUY",
                "EXIT",
            ]
        )
    ][
        [
            "timestamp",
            "close",
            "momentum_12m_pct",
            "target_position",
            "signal",
        ]
    ].copy()

    if transitions.empty:

        print("No BUY / EXIT transitions.")

        return

    print(transitions.to_string(index=False))


# ============================================================
# RUN ONE SYMBOL
# ============================================================


def process_symbol(
    symbol: str,
) -> dict:

    dataframe = load_data(symbol)

    monthly = build_monthly_signal_table(dataframe)

    signals = apply_signals_to_daily_data(
        dataframe,
        monthly,
    )

    valid_monthly = monthly[monthly["momentum_12m_pct"].notna()]

    buys = int((signals["signal"] == "BUY").sum())

    exits = int((signals["signal"] == "EXIT").sum())

    output_file = OUTPUT_DIR / (
        f"{symbol}_" "spot_1d_" "tsmom_reference_" "signals.csv"
    )

    signals.to_csv(
        output_file,
        index=False,
    )

    print()
    print("=" * 110)

    print(symbol)

    print("=" * 110)

    print(f"Daily candles: " f"{len(dataframe)}")

    print(f"Completed month-ends: " f"{len(monthly)}")

    print(f"Valid 12-month decisions: " f"{len(valid_monthly)}")

    print()
    print("SIGNAL TRANSITIONS")

    print("-" * 110)

    print_signal_transitions(signals)

    print()

    print(f"BUY signals:  " f"{buys}")

    print(f"EXIT signals: " f"{exits}")

    print(f"Saved: " f"{output_file}")

    return {
        "symbol": symbol,
        "daily_candles": (len(dataframe)),
        "month_ends": (len(monthly)),
        "valid_12m_decisions": (len(valid_monthly)),
        "buy_signals": (buys),
        "exit_signals": (exits),
        "round_trips_expected": (
            min(
                buys,
                exits,
            )
        ),
        "earliest_data": (dataframe.iloc[0]["timestamp"]),
        "latest_data": (dataframe.iloc[-1]["timestamp"]),
        "signal_file": str(output_file),
    }


# ============================================================
# MAIN
# ============================================================


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    symbols = load_frozen_universe()

    print()
    print("=" * 110)

    print("TSMOM REFERENCE — " "FROZEN UNIVERSE SIGNAL GENERATION")

    print("=" * 110)

    print()

    print("Frozen strategy:")

    print("  Lookback:            12 months")

    print("  Decision frequency:  month-end")

    print("  Positive momentum:   LONG")

    print("  Zero/negative:       CASH")

    print("  Trend MA:            NONE")

    print("  Short selling:       NONE")

    print("  Leverage:            NONE")

    print("  Signal format:       BUY / EXIT / HOLD")

    print()

    print(f"Frozen universe size: " f"{len(symbols)}")

    for symbol in symbols:

        print(f"  {symbol}")

    summaries = []

    for symbol in symbols:

        summary = process_symbol(symbol)

        summaries.append(summary)

    summary_dataframe = pd.DataFrame(summaries)

    summary_dataframe.to_csv(
        SUMMARY_FILE,
        index=False,
    )

    print()
    print()
    print("=" * 110)

    print("FROZEN UNIVERSE SIGNAL SUMMARY")

    print("=" * 110)

    print()

    display_columns = [
        "symbol",
        "valid_12m_decisions",
        "buy_signals",
        "exit_signals",
        "round_trips_expected",
    ]

    print(summary_dataframe[display_columns].to_string(index=False))

    print()

    print("TOTALS")

    print("-" * 110)

    print(
        f"Valid monthly decisions: "
        f"{int(summary_dataframe['valid_12m_decisions'].sum())}"
    )

    print(f"BUY signals: " f"{int(summary_dataframe['buy_signals'].sum())}")

    print(f"EXIT signals: " f"{int(summary_dataframe['exit_signals'].sum())}")

    print(
        f"Expected completed round trips: "
        f"{int(summary_dataframe['round_trips_expected'].sum())}"
    )

    print()

    print(f"Summary saved: " f"{SUMMARY_FILE}")


if __name__ == "__main__":
    main()
