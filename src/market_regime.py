import argparse
from pathlib import Path

import pandas as pd

SYMBOL = "BTCUSDT"
MARKET = "spot"

SUPPORTED_TIMEFRAMES = [
    "15m",
    "1h",
    "4h",
]

# Baseline thresholds.
# We do NOT optimize them yet.
ADX_TREND_MIN = 25.0
ADX_SIDEWAYS_MAX = 20.0


def parse_arguments():
    parser = argparse.ArgumentParser(description="Classify historical market regimes.")

    parser.add_argument(
        "--timeframe",
        choices=SUPPORTED_TIMEFRAMES,
        default="1h",
        help="Timeframe to classify.",
    )

    return parser.parse_args()


def classify_market_regime(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    dataframe = dataframe.copy()

    # ------------------------------------------
    # Bullish trend
    # ------------------------------------------

    bullish_structure = (
        (dataframe["ema_20"] > dataframe["ema_50"])
        & (dataframe["ema_50"] > dataframe["ema_200"])
        & (dataframe["plus_di_14"] > dataframe["minus_di_14"])
    )

    bullish_trend = bullish_structure & (dataframe["adx_14"] >= ADX_TREND_MIN)

    # ------------------------------------------
    # Bearish trend
    # ------------------------------------------

    bearish_structure = (
        (dataframe["ema_20"] < dataframe["ema_50"])
        & (dataframe["ema_50"] < dataframe["ema_200"])
        & (dataframe["minus_di_14"] > dataframe["plus_di_14"])
    )

    bearish_trend = bearish_structure & (dataframe["adx_14"] >= ADX_TREND_MIN)

    # ------------------------------------------
    # Sideways / weak trend
    # ------------------------------------------

    sideways = dataframe["adx_14"] <= ADX_SIDEWAYS_MAX

    # ------------------------------------------
    # Default regime
    # ------------------------------------------

    dataframe["market_regime"] = "TRANSITION"

    dataframe.loc[
        sideways,
        "market_regime",
    ] = "SIDEWAYS"

    dataframe.loc[
        bullish_trend,
        "market_regime",
    ] = "TRENDING_BULLISH"

    dataframe.loc[
        bearish_trend,
        "market_regime",
    ] = "TRENDING_BEARISH"

    return dataframe


def main():
    args = parse_arguments()

    timeframe = args.timeframe

    input_file = Path(f"data/{SYMBOL}_{MARKET}_" f"{timeframe}_indicators.csv")

    output_file = Path(f"data/{SYMBOL}_{MARKET}_" f"{timeframe}_regime.csv")

    if not input_file.exists():
        raise FileNotFoundError(f"Indicator file not found: {input_file}")

    dataframe = pd.read_csv(
        input_file,
        parse_dates=["timestamp"],
    )

    dataframe = classify_market_regime(dataframe)

    dataframe.to_csv(
        output_file,
        index=False,
    )

    counts = dataframe["market_regime"].value_counts()

    percentages = dataframe["market_regime"].value_counts(normalize=True) * 100

    print()
    print("=" * 55)
    print("MARKET REGIME CLASSIFICATION")
    print("=" * 55)

    print()
    print(f"Symbol:    {SYMBOL}")
    print(f"Market:    {MARKET.upper()}")
    print(f"Timeframe: {timeframe}")
    print(f"Candles:   {len(dataframe)}")

    print()
    print("-" * 55)
    print("REGIMES")
    print("-" * 55)

    regimes = [
        "TRENDING_BULLISH",
        "TRENDING_BEARISH",
        "SIDEWAYS",
        "TRANSITION",
    ]

    for regime in regimes:
        count = counts.get(regime, 0)
        percentage = percentages.get(
            regime,
            0.0,
        )

        print(f"{regime:<20}" f"{count:>8} candles " f"({percentage:>6.2f}%)")

    print()
    print(f"Saved to: {output_file}")


if __name__ == "__main__":
    main()
