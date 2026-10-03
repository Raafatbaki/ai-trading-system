from pathlib import Path

import pandas as pd

# ============================================================
# PATHS
# ============================================================

DATA_DIR = Path("data/reference_tsmom")

UNIVERSE_FILE = DATA_DIR / "universe_screen.csv"

MONTHLY_DIR = DATA_DIR / "risk_scaled_monthly"

OUTPUT_DIR = DATA_DIR / "benchmarks"

OUTPUT_FILE = OUTPUT_DIR / "equal_weight_market_weights.csv"


# ============================================================
# LOAD FROZEN UNIVERSE
# ============================================================


def load_frozen_universe() -> list[str]:

    if not UNIVERSE_FILE.exists():

        raise FileNotFoundError(f"Missing universe file: " f"{UNIVERSE_FILE}")

    dataframe = pd.read_csv(UNIVERSE_FILE)

    required_columns = [
        "symbol",
        "eligible",
    ]

    missing = [column for column in required_columns if column not in dataframe.columns]

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

        raise RuntimeError("Frozen universe is empty.")

    return symbols


# ============================================================
# LOAD MONTHLY DATA
# ============================================================


def load_monthly(
    symbol: str,
) -> pd.DataFrame:

    file_path = MONTHLY_DIR / (
        f"{symbol}_" "tsmom_reference_" "risk_scaled_monthly.csv"
    )

    if not file_path.exists():

        raise FileNotFoundError(f"Missing monthly file: " f"{file_path}")

    dataframe = pd.read_csv(
        file_path,
        parse_dates=["timestamp"],
    )

    dataframe = dataframe.sort_values("timestamp").reset_index(drop=True)

    return dataframe


# ============================================================
# BUILD ELIGIBILITY TABLE
# ============================================================


def build_eligibility_table(
    symbols: list[str],
) -> pd.DataFrame:

    rows = []

    for symbol in symbols:

        monthly = load_monthly(symbol)

        for _, row in monthly.iterrows():

            # ------------------------------------------------
            # Asset becomes benchmark-eligible only when
            # 12-month history exists.
            #
            # Momentum SIGN is deliberately ignored.
            # ------------------------------------------------

            valid_history = pd.notna(row["momentum_12m_pct"])

            if not valid_history:
                continue

            rows.append(
                {
                    "timestamp": (row["timestamp"]),
                    "symbol": (symbol),
                    "eligible": 1.0,
                }
            )

    dataframe = pd.DataFrame(rows)

    if dataframe.empty:

        raise RuntimeError("No valid market benchmark " "observations.")

    return dataframe


# ============================================================
# BUILD EQUAL-WEIGHT MARKET PORTFOLIO
# ============================================================


def build_market_weights(
    symbols: list[str],
) -> pd.DataFrame:

    eligibility = build_eligibility_table(symbols)

    matrix = eligibility.pivot(
        index="timestamp",
        columns="symbol",
        values="eligible",
    ).sort_index()

    matrix = matrix.reindex(columns=symbols).fillna(0.0)

    eligible_count = matrix.sum(axis=1)

    weights = matrix.copy()

    for timestamp in weights.index:

        count = float(eligible_count.loc[timestamp])

        if count > 0:

            weights.loc[timestamp] = matrix.loc[timestamp] / count

        else:

            weights.loc[timestamp] = 0.0

    exposure = weights.sum(axis=1)

    result = pd.DataFrame(
        {
            "timestamp": (weights.index),
            "eligible_assets": (eligible_count.values),
            "portfolio_exposure": (exposure.values),
            "cash_weight": ((1.0 - exposure).values),
            "max_asset_weight": (weights.max(axis=1).values),
        }
    )

    for symbol in symbols:

        result[f"{symbol}_weight"] = weights[symbol].values

    return result


# ============================================================
# VALIDATION
# ============================================================


def validate_weights(
    dataframe: pd.DataFrame,
    symbols: list[str],
):

    tolerance = 1e-10

    weight_columns = [f"{symbol}_weight" for symbol in symbols]

    weights = dataframe[weight_columns]

    if (weights < -tolerance).any().any():

        raise RuntimeError("Negative benchmark weight detected.")

    if (weights > 1.0 + tolerance).any().any():

        raise RuntimeError("Asset weight above 100%.")

    total = weights.sum(axis=1)

    if (total > 1.0 + tolerance).any():

        raise RuntimeError("Benchmark leverage detected.")

    # Once at least one asset is eligible,
    # this benchmark should be fully invested.

    active = dataframe["eligible_assets"] > 0

    if not (total.loc[active].sub(1.0).abs() <= tolerance).all():

        raise RuntimeError(
            "Market benchmark is not " "100% invested while assets " "are eligible."
        )


# ============================================================
# FIRST ELIGIBLE DATE AUDIT
# ============================================================


def first_eligible_dates(
    dataframe: pd.DataFrame,
    symbols: list[str],
):

    print()
    print("FIRST ELIGIBLE MONTH")

    print("-" * 100)

    for symbol in symbols:

        column = f"{symbol}_weight"

        active = dataframe[dataframe[column] > 0]

        if active.empty:

            print(f"{symbol:<10} NONE")

        else:

            timestamp = active.iloc[0]["timestamp"]

            print(f"{symbol:<10} " f"{timestamp}")


# ============================================================
# MAIN
# ============================================================


def main():

    symbols = load_frozen_universe()

    dataframe = build_market_weights(symbols)

    validate_weights(
        dataframe,
        symbols,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print("=" * 100)

    print("EQUAL-WEIGHT MARKET BENCHMARK")

    print("=" * 100)

    print()

    print("Rules:")

    print("  Universe: frozen 11 assets")

    print("  Eligibility: 12 months " "of available history")

    print("  Momentum sign: IGNORED")

    print("  Eligible assets: " "equal weight")

    print("  Portfolio exposure: " "100% when >=1 asset eligible")

    print("  Rebalance: monthly")

    print("  Leverage: NONE")

    print("  Short: NONE")

    print()

    print(f"Monthly decisions: " f"{len(dataframe)}")

    print(f"First decision: " f"{dataframe.iloc[0]['timestamp']}")

    print(f"Last decision:  " f"{dataframe.iloc[-1]['timestamp']}")

    print()

    print(f"Average eligible assets: " f"{dataframe['eligible_assets'].mean():.2f}")

    print(f"Minimum eligible assets: " f"{int(dataframe['eligible_assets'].min())}")

    print(f"Maximum eligible assets: " f"{int(dataframe['eligible_assets'].max())}")

    print()

    print(f"Average exposure: " f"{dataframe['portfolio_exposure'].mean() * 100:.2f}%")

    print(f"Minimum exposure: " f"{dataframe['portfolio_exposure'].min() * 100:.2f}%")

    print(f"Maximum exposure: " f"{dataframe['portfolio_exposure'].max() * 100:.2f}%")

    print()

    print(
        f"Average largest position: "
        f"{dataframe['max_asset_weight'].mean() * 100:.2f}%"
    )

    print(
        f"Maximum single position: " f"{dataframe['max_asset_weight'].max() * 100:.2f}%"
    )

    first_eligible_dates(
        dataframe,
        symbols,
    )

    print()
    print("LAST 12 MONTHS")

    print("-" * 100)

    print(
        dataframe[
            [
                "timestamp",
                "eligible_assets",
                "portfolio_exposure",
                "cash_weight",
                "max_asset_weight",
            ]
        ]
        .tail(12)
        .to_string(index=False)
    )

    print()

    print(f"Saved: " f"{OUTPUT_FILE}")


if __name__ == "__main__":
    main()
