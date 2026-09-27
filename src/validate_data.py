import argparse
from pathlib import Path

import pandas as pd

MARKET = "spot"

SUPPORTED_SYMBOLS = [
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
]

TIMEFRAMES = {
    "15m": pd.Timedelta(minutes=15),
    "1h": pd.Timedelta(hours=1),
    "4h": pd.Timedelta(hours=4),
}


def parse_arguments():
    parser = argparse.ArgumentParser(description="Validate historical market data.")

    parser.add_argument(
        "--symbol",
        choices=SUPPORTED_SYMBOLS,
        default="BTCUSDT",
    )

    parser.add_argument(
        "--timeframe",
        choices=TIMEFRAMES.keys(),
        default="1h",
    )

    return parser.parse_args()


def validate_data(
    data_file: Path,
    expected_interval: pd.Timedelta,
) -> bool:

    dataframe = pd.read_csv(
        data_file,
        parse_dates=["timestamp"],
    )

    duplicate_timestamps = dataframe["timestamp"].duplicated().sum()

    missing_values = dataframe.isna().sum().sum()

    chronological_order = dataframe["timestamp"].is_monotonic_increasing

    time_differences = dataframe["timestamp"].diff()

    gaps = dataframe[time_differences.notna() & (time_differences != expected_interval)]

    invalid_prices = dataframe[
        (dataframe["open"] <= 0)
        | (dataframe["high"] <= 0)
        | (dataframe["low"] <= 0)
        | (dataframe["close"] <= 0)
    ]

    invalid_ohlc = dataframe[
        (dataframe["high"] < dataframe["low"])
        | (dataframe["high"] < dataframe["open"])
        | (dataframe["high"] < dataframe["close"])
        | (dataframe["low"] > dataframe["open"])
        | (dataframe["low"] > dataframe["close"])
    ]

    invalid_volume = dataframe[dataframe["volume"] < 0]

    print()
    print("=" * 60)
    print("MARKET DATA VALIDATION")
    print("=" * 60)

    print()
    print(f"File:                  {data_file}")
    print(f"Candles:               {len(dataframe)}")
    print(f"From:                  {dataframe['timestamp'].min()}")
    print(f"To:                    {dataframe['timestamp'].max()}")

    print()
    print(f"Duplicate timestamps:  {duplicate_timestamps}")
    print(f"Missing values:        {missing_values}")
    print(f"Chronological order:   {chronological_order}")
    print(f"Time gaps:             {len(gaps)}")
    print(f"Invalid prices:        {len(invalid_prices)}")
    print(f"Invalid OHLC candles:  {len(invalid_ohlc)}")
    print(f"Invalid volume:        {len(invalid_volume)}")

    passed = (
        duplicate_timestamps == 0
        and missing_values == 0
        and chronological_order
        and len(gaps) == 0
        and len(invalid_prices) == 0
        and len(invalid_ohlc) == 0
        and len(invalid_volume) == 0
    )

    print()

    if passed:
        print("✅ Data validation PASSED")
    else:
        print("❌ Data validation FAILED")

        if not gaps.empty:
            print()
            print("First gaps:")

            for index in gaps.index[:10]:
                print(
                    dataframe.loc[index - 1, "timestamp"],
                    "→",
                    dataframe.loc[index, "timestamp"],
                )

    return passed


def main():
    args = parse_arguments()

    symbol = args.symbol
    timeframe = args.timeframe

    data_file = Path(f"data/{symbol}_{MARKET}_{timeframe}.csv")

    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {data_file}")

    validate_data(
        data_file=data_file,
        expected_interval=TIMEFRAMES[timeframe],
    )


if __name__ == "__main__":
    main()
