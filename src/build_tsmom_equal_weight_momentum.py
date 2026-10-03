from pathlib import Path

import pandas as pd

# ============================================================
# PATHS
# ============================================================

DATA_DIR = Path("data/reference_tsmom")

UNIVERSE_FILE = DATA_DIR / "universe_screen.csv"

MONTHLY_DIR = DATA_DIR / "risk_scaled_monthly"

OUTPUT_DIR = DATA_DIR / "benchmarks"

OUTPUT_FILE = OUTPUT_DIR / "tsmom_equal_weight_momentum_weights.csv"


# ============================================================
# LOAD FROZEN UNIVERSE
# ============================================================


def load_frozen_universe() -> list[str]:

    dataframe = pd.read_csv(UNIVERSE_FILE)

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
# LOAD MONTHLY MOMENTUM DECISIONS
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

    return dataframe


# ============================================================
# BUILD EQUAL-WEIGHT MOMENTUM PORTFOLIO
# ============================================================


def build_weights(
    symbols: list[str],
) -> pd.DataFrame:

    rows = []

    for symbol in symbols:

        monthly = load_monthly(symbol)

        valid = monthly[monthly["momentum_12m_pct"].notna()]

        for _, row in valid.iterrows():

            rows.append(
                {
                    "timestamp": (row["timestamp"]),
                    "symbol": (symbol),
                    "long_signal": (1 if (row["momentum_12m_pct"] > 0) else 0),
                }
            )

    long_table = pd.DataFrame(rows)

    if long_table.empty:

        raise RuntimeError("No valid momentum decisions.")

    signals = long_table.pivot(
        index="timestamp",
        columns="symbol",
        values="long_signal",
    ).sort_index()

    signals = signals.reindex(columns=symbols).fillna(0.0)

    active_assets = signals.sum(axis=1)

    weights = signals.copy()

    for timestamp in weights.index:

        count = active_assets.loc[timestamp]

        if count > 0:

            weights.loc[timestamp] = signals.loc[timestamp] / count

        else:

            weights.loc[timestamp] = 0.0

    result = pd.DataFrame(
        {
            "timestamp": (weights.index),
            "active_assets": (active_assets.values),
            "portfolio_exposure": ((weights.sum(axis=1)).values),
            "cash_weight": ((1.0 - weights.sum(axis=1)).values),
        }
    )

    for symbol in symbols:

        result[f"{symbol}_weight"] = weights[symbol].values

    return result


# ============================================================
# VALIDATION
# ============================================================


def validate(
    dataframe: pd.DataFrame,
    symbols: list[str],
):

    columns = [f"{symbol}_weight" for symbol in symbols]

    total = dataframe[columns].sum(axis=1)

    if (total > 1.0 + 1e-10).any():

        raise RuntimeError("Leverage detected.")

    if (dataframe[columns] < -1e-10).any().any():

        raise RuntimeError("Negative weight detected.")


# ============================================================
# MAIN
# ============================================================


def main():

    symbols = load_frozen_universe()

    dataframe = build_weights(symbols)

    validate(
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

    print("TSMOM EQUAL-WEIGHT " "MOMENTUM BENCHMARK")

    print("=" * 100)

    print()

    print("Rules:")

    print("  Momentum: 12 months")

    print("  Positive momentum: LONG")

    print("  Negative/zero: CASH")

    print("  Active assets: equal weight")

    print("  Leverage: NONE")

    print("  Short: NONE")

    print()

    print(f"Monthly decisions: " f"{len(dataframe)}")

    print(f"Average active assets: " f"{dataframe['active_assets'].mean():.2f}")

    print(f"Maximum active assets: " f"{int(dataframe['active_assets'].max())}")

    print(
        f"Months completely in cash: " f"{int((dataframe['active_assets'] == 0).sum())}"
    )

    print()

    print("LAST 12 MONTHS")

    print("-" * 100)

    print(
        dataframe[
            [
                "timestamp",
                "active_assets",
                "portfolio_exposure",
                "cash_weight",
            ]
        ]
        .tail(12)
        .to_string(index=False)
    )

    print()

    print(f"Saved: " f"{OUTPUT_FILE}")


if __name__ == "__main__":
    main()
