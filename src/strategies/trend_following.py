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


# --------------------------------------------------
# Strategy configuration
# --------------------------------------------------
#
# Important:
# Keep these values identical across all timeframes
# for the first comparison.
#
# We do NOT optimize them yet.
# --------------------------------------------------

ADX_MIN = 25.0

RSI_MIN = 50.0
RSI_MAX = 68.0

MIN_VOLUME_RATIO = 1.10


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=("Generate long-only trend-following signals.")
    )

    parser.add_argument(
        "--timeframe",
        choices=SUPPORTED_TIMEFRAMES,
        default="15m",
        help="Timeframe to process.",
    )

    return parser.parse_args()


def add_trend_following_signals(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Long-only trend-following strategy.

    Current trading mode:
    - Spot only
    - Long only
    - No short
    - No leverage
    """

    dataframe = dataframe.copy()

    # --------------------------------------------------
    # 1. Trend direction
    # --------------------------------------------------

    dataframe["trend_bullish"] = (
        (dataframe["ema_20"] > dataframe["ema_50"])
        & (dataframe["ema_50"] > dataframe["ema_200"])
        & (dataframe["close"] > dataframe["ema_20"])
    )

    # --------------------------------------------------
    # 2. Trend strength
    # --------------------------------------------------

    dataframe["trend_strong"] = dataframe["adx_14"] >= ADX_MIN

    dataframe["direction_bullish"] = dataframe["plus_di_14"] > dataframe["minus_di_14"]

    # --------------------------------------------------
    # 3. Momentum
    # --------------------------------------------------

    dataframe["momentum_valid"] = (dataframe["rsi_14"] >= RSI_MIN) & (
        dataframe["rsi_14"] <= RSI_MAX
    )

    # --------------------------------------------------
    # 4. Volume confirmation
    # --------------------------------------------------

    dataframe["volume_valid"] = dataframe["volume_ratio"] >= MIN_VOLUME_RATIO

    # --------------------------------------------------
    # 5. Entry condition
    # --------------------------------------------------

    dataframe["long_condition"] = (
        dataframe["trend_bullish"]
        & dataframe["trend_strong"]
        & dataframe["direction_bullish"]
        & dataframe["momentum_valid"]
        & dataframe["volume_valid"]
    )

    # --------------------------------------------------
    # 6. Exit condition
    # --------------------------------------------------

    dataframe["exit_condition"] = (dataframe["ema_20"] < dataframe["ema_50"]) | (
        dataframe["close"] < dataframe["ema_50"]
    )

    # --------------------------------------------------
    # 7. Position state machine
    # --------------------------------------------------
    #
    # Only one position may be open.
    #
    # BUY:
    # only if no position is open.
    #
    # HOLD:
    # position is already open.
    #
    # EXIT:
    # close the current position.
    #
    # NO_SIGNAL:
    # nothing to do.
    # --------------------------------------------------

    signals = []

    position_open = False

    for row in dataframe.itertuples():
        if not position_open:
            if row.long_condition:
                signals.append("BUY")
                position_open = True
            else:
                signals.append("NO_SIGNAL")

        else:
            if row.exit_condition:
                signals.append("EXIT")
                position_open = False
            else:
                signals.append("HOLD")

    dataframe["signal"] = signals

    dataframe["buy_signal"] = dataframe["signal"] == "BUY"

    dataframe["exit_signal"] = dataframe["signal"] == "EXIT"

    return dataframe


def main():
    args = parse_arguments()

    timeframe = args.timeframe

    input_file = Path(f"data/{SYMBOL}_{MARKET}_" f"{timeframe}_indicators.csv")

    output_file = Path(f"data/{SYMBOL}_{MARKET}_" f"{timeframe}_trend_signals.csv")

    if not input_file.exists():
        raise FileNotFoundError(f"Indicator file not found: " f"{input_file}")

    dataframe = pd.read_csv(
        input_file,
        parse_dates=["timestamp"],
    )

    dataframe = add_trend_following_signals(dataframe)

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe.to_csv(
        output_file,
        index=False,
    )

    buy_signals = dataframe[dataframe["signal"] == "BUY"]

    exit_signals = dataframe[dataframe["signal"] == "EXIT"]

    hold_signals = dataframe[dataframe["signal"] == "HOLD"]

    no_signals = dataframe[dataframe["signal"] == "NO_SIGNAL"]

    print()
    print("=" * 55)
    print("TREND FOLLOWING SIGNALS")
    print("=" * 55)
    print()

    print(f"Symbol:      {SYMBOL}")
    print(f"Market:      {MARKET.upper()}")
    print(f"Timeframe:   {timeframe}")
    print("Mode:        LONG ONLY")
    print("Short:       DISABLED")
    print("Leverage:    DISABLED")

    print()
    print("-" * 55)
    print("SIGNALS")
    print("-" * 55)

    print(f"Candles:      " f"{len(dataframe)}")

    print(f"BUY signals:  " f"{len(buy_signals)}")

    print(f"EXIT signals: " f"{len(exit_signals)}")

    print(f"HOLD candles: " f"{len(hold_signals)}")

    print(f"NO SIGNAL:    " f"{len(no_signals)}")

    print()

    if not buy_signals.empty:
        columns = [
            "timestamp",
            "close",
            "ema_20",
            "ema_50",
            "ema_200",
            "rsi_14",
            "adx_14",
            "volume_ratio",
        ]

        print("Last BUY signals:")
        print(buy_signals[columns].tail(5))

    print()
    print(f"Saved to: {output_file}")


if __name__ == "__main__":
    main()
