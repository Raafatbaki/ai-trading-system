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
# Baseline breakout configuration
# --------------------------------------------------

ADX_MIN = 20.0
MIN_VOLUME_RATIO = 1.50


def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate long-only breakout signals.")

    parser.add_argument(
        "--timeframe",
        choices=SUPPORTED_TIMEFRAMES,
        default="1h",
        help="Timeframe to process.",
    )

    return parser.parse_args()


def add_breakout_signals(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    dataframe = dataframe.copy()

    # --------------------------------------------------
    # Breakout
    # --------------------------------------------------
    #
    # Donchian High uses PREVIOUS candles only,
    # so the current candle is not used to create
    # its own breakout level.
    # --------------------------------------------------

    dataframe["price_breakout"] = dataframe["close"] > dataframe["donchian_high_20"]

    # --------------------------------------------------
    # Larger trend filter
    # --------------------------------------------------

    dataframe["trend_valid"] = dataframe["close"] > dataframe["ema_200"]

    # --------------------------------------------------
    # Direction confirmation
    # --------------------------------------------------

    dataframe["direction_valid"] = dataframe["plus_di_14"] > dataframe["minus_di_14"]

    # --------------------------------------------------
    # Trend strength
    # --------------------------------------------------

    dataframe["strength_valid"] = dataframe["adx_14"] >= ADX_MIN

    # --------------------------------------------------
    # Volume confirmation
    # --------------------------------------------------

    dataframe["volume_valid"] = dataframe["volume_ratio"] >= MIN_VOLUME_RATIO

    # --------------------------------------------------
    # Entry
    # --------------------------------------------------

    dataframe["long_condition"] = (
        dataframe["price_breakout"]
        & dataframe["trend_valid"]
        & dataframe["direction_valid"]
        & dataframe["strength_valid"]
        & dataframe["volume_valid"]
    )

    # --------------------------------------------------
    # Exit
    # --------------------------------------------------
    #
    # For baseline:
    # leave when momentum/trend weakens.
    # --------------------------------------------------

    dataframe["exit_condition"] = (dataframe["close"] < dataframe["ema_20"]) | (
        dataframe["minus_di_14"] > dataframe["plus_di_14"]
    )

    # --------------------------------------------------
    # Position state machine
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

    output_file = Path(f"data/{SYMBOL}_{MARKET}_" f"{timeframe}_breakout_signals.csv")

    if not input_file.exists():
        raise FileNotFoundError(f"Indicator file not found: " f"{input_file}")

    dataframe = pd.read_csv(
        input_file,
        parse_dates=["timestamp"],
    )

    dataframe = add_breakout_signals(dataframe)

    dataframe.to_csv(
        output_file,
        index=False,
    )

    buy_signals = dataframe[dataframe["signal"] == "BUY"]

    exit_signals = dataframe[dataframe["signal"] == "EXIT"]

    hold_signals = dataframe[dataframe["signal"] == "HOLD"]

    print()
    print("=" * 55)
    print("BREAKOUT STRATEGY")
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

    print(f"Candles:      {len(dataframe)}")

    print(f"BUY signals:  {len(buy_signals)}")

    print(f"EXIT signals: {len(exit_signals)}")

    print(f"HOLD candles: {len(hold_signals)}")

    print()

    if not buy_signals.empty:

        columns = [
            "timestamp",
            "close",
            "donchian_high_20",
            "ema_200",
            "adx_14",
            "volume_ratio",
        ]

        print("Last BUY signals:")

        print(buy_signals[columns].tail(10))

    print()
    print(f"Saved to: {output_file}")


if __name__ == "__main__":
    main()
