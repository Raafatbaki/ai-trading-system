from pathlib import Path

import pandas as pd

# ============================================================
# PATHS
# ============================================================

DATA_DIR = Path("data/reference_tsmom")

UNIVERSE_FILE = DATA_DIR / "universe_screen.csv"

MONTHLY_DIR = DATA_DIR / "risk_scaled_monthly"

OUTPUT_DIR = DATA_DIR / "portfolio"

OUTPUT_FILE = OUTPUT_DIR / "tsmom_reference_portfolio_weights.csv"


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

        raise RuntimeError("No eligible symbols.")

    return symbols


# ============================================================
# LOAD ONE MONTHLY FILE
# ============================================================


def load_monthly(
    symbol: str,
) -> pd.DataFrame:

    file_path = MONTHLY_DIR / (
        f"{symbol}_" "tsmom_reference_" "risk_scaled_monthly.csv"
    )

    if not file_path.exists():

        raise FileNotFoundError(f"Missing file: " f"{file_path}")

    dataframe = pd.read_csv(
        file_path,
        parse_dates=["timestamp"],
    )

    return dataframe


# ============================================================
# BUILD CROSS-ASSET TABLE
# ============================================================


def build_portfolio_table(
    symbols: list[str],
) -> pd.DataFrame:

    rows = []

    for symbol in symbols:

        monthly = load_monthly(symbol)

        valid = monthly[monthly["target_weight"].notna()].copy()

        for _, row in valid.iterrows():

            rows.append(
                {
                    "timestamp": (row["timestamp"]),
                    "symbol": symbol,
                    "momentum_12m_pct": (row["momentum_12m_pct"]),
                    "ewma_vol_annual": (row["ewma_vol_annual"]),
                    "raw_asset_weight": (row["target_weight"]),
                }
            )

    long_table = pd.DataFrame(rows)

    if long_table.empty:

        raise RuntimeError("No valid portfolio decisions.")

    # ========================================================
    # PIVOT:
    #
    # one row per month
    # one column per asset
    # ========================================================

    pivot = long_table.pivot(
        index="timestamp",
        columns="symbol",
        values="raw_asset_weight",
    ).sort_index()

    # Assets without a valid signal for a month
    # contribute zero portfolio weight.
    pivot = pivot.fillna(0.0)

    # Keep frozen universe column ordering.
    pivot = pivot.reindex(
        columns=symbols,
        fill_value=0.0,
    )

    raw_total = pivot.sum(axis=1)

    # ========================================================
    # NO-LEVERAGE NORMALIZATION
    #
    # If raw total <= 1:
    #     preserve raw weights.
    #
    # If raw total > 1:
    #     divide all weights by raw total.
    # ========================================================

    normalization_factor = raw_total.clip(lower=1.0)

    normalized = pivot.div(
        normalization_factor,
        axis=0,
    )

    portfolio_exposure = normalized.sum(axis=1)

    cash_weight = 1.0 - portfolio_exposure

    active_assets = (normalized > 0).sum(axis=1)

    max_asset_weight = normalized.max(axis=1)

    # ========================================================
    # OUTPUT TABLE
    # ========================================================

    result = pd.DataFrame(
        {
            "timestamp": (normalized.index),
            "raw_total_weight": (raw_total.values),
            "normalization_factor": (normalization_factor.values),
            "portfolio_exposure": (portfolio_exposure.values),
            "cash_weight": (cash_weight.values),
            "active_assets": (active_assets.values),
            "max_asset_weight": (max_asset_weight.values),
        }
    )

    for symbol in symbols:

        result[f"{symbol}_raw"] = pivot[symbol].values

        result[f"{symbol}_weight"] = normalized[symbol].values

    return result


# ============================================================
# VALIDATION
# ============================================================


def validate_portfolio(
    dataframe: pd.DataFrame,
    symbols: list[str],
):

    tolerance = 1e-10

    weight_columns = [f"{symbol}_weight" for symbol in symbols]

    if (dataframe[weight_columns] < -tolerance).any().any():

        raise RuntimeError("Negative portfolio weight detected.")

    if (dataframe[weight_columns] > 1.0 + tolerance).any().any():

        raise RuntimeError("Asset weight above 100%.")

    if (dataframe["portfolio_exposure"] > 1.0 + tolerance).any():

        raise RuntimeError("Portfolio leverage detected.")

    if (dataframe["cash_weight"] < -tolerance).any():

        raise RuntimeError("Negative cash weight detected.")


# ============================================================
# MAIN
# ============================================================


def main():

    symbols = load_frozen_universe()

    dataframe = build_portfolio_table(symbols)

    validate_portfolio(
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
    print("=" * 110)

    print("TSMOM REFERENCE — " "PORTFOLIO WEIGHT AUDIT")

    print("=" * 110)

    print()

    print("Frozen portfolio rule:")

    print("  raw asset weights = " "40% vol-scaled weights")

    print("  if total <= 100%: " "keep cash remainder")

    print("  if total > 100%: " "scale all assets proportionally")

    print("  leverage: NONE")

    print("  short: NONE")

    print()

    print(f"Monthly portfolio decisions: " f"{len(dataframe)}")

    print()

    print(f"Average exposure: " f"{dataframe['portfolio_exposure'].mean() * 100:.2f}%")

    print(
        f"Median exposure:  " f"{dataframe['portfolio_exposure'].median() * 100:.2f}%"
    )

    print(f"Minimum exposure: " f"{dataframe['portfolio_exposure'].min() * 100:.2f}%")

    print(f"Maximum exposure: " f"{dataframe['portfolio_exposure'].max() * 100:.2f}%")

    print()

    print(f"Average active assets: " f"{dataframe['active_assets'].mean():.2f}")

    print(f"Maximum active assets: " f"{int(dataframe['active_assets'].max())}")

    print()

    print(
        f"Average largest position: "
        f"{dataframe['max_asset_weight'].mean() * 100:.2f}%"
    )

    print(
        f"Maximum single position:  "
        f"{dataframe['max_asset_weight'].max() * 100:.2f}%"
    )

    print()

    normalized_months = int((dataframe["normalization_factor"] > 1.0).sum())

    cash_months = int((dataframe["cash_weight"] > 1e-12).sum())

    print(f"Months requiring normalization: " f"{normalized_months}")

    print(f"Months holding some cash:        " f"{cash_months}")

    print()

    print("LAST 12 MONTHS")

    print("-" * 110)

    display_columns = [
        "timestamp",
        "raw_total_weight",
        "portfolio_exposure",
        "cash_weight",
        "active_assets",
        "max_asset_weight",
    ]

    print(dataframe[display_columns].tail(12).to_string(index=False))

    print()

    print(f"Saved: " f"{OUTPUT_FILE}")


if __name__ == "__main__":
    main()
