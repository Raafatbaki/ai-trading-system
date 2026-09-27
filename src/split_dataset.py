import argparse
from pathlib import Path

import pandas as pd

MARKET = "spot"

SUPPORTED_SYMBOLS = [
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
]

SUPPORTED_TIMEFRAMES = [
    "15m",
    "1h",
    "4h",
]

VALIDATION_DAYS = 183
HOLDOUT_DAYS = 183


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Split historical data into " "development, validation and holdout."
        )
    )

    parser.add_argument(
        "--symbol",
        choices=SUPPORTED_SYMBOLS,
        default="BTCUSDT",
        help="Trading pair.",
    )

    parser.add_argument(
        "--timeframe",
        choices=SUPPORTED_TIMEFRAMES,
        default="1h",
        help="Timeframe to split.",
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    symbol = args.symbol
    timeframe = args.timeframe

    input_file = Path(f"data/" f"{symbol}_{MARKET}_" f"{timeframe}_indicators.csv")

    output_directory = Path("data/splits")

    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: " f"{input_file}")

    dataframe = pd.read_csv(
        input_file,
        parse_dates=["timestamp"],
    )

    dataframe = (
        dataframe.sort_values("timestamp")
        .drop_duplicates(subset="timestamp")
        .reset_index(drop=True)
    )

    start_date = dataframe["timestamp"].min()

    end_date = dataframe["timestamp"].max()

    holdout_start = end_date - pd.Timedelta(days=HOLDOUT_DAYS)

    validation_start = holdout_start - pd.Timedelta(days=VALIDATION_DAYS)

    development = dataframe[dataframe["timestamp"] < validation_start].copy()

    validation = dataframe[
        (dataframe["timestamp"] >= validation_start)
        & (dataframe["timestamp"] < holdout_start)
    ].copy()

    historical_holdout = dataframe[dataframe["timestamp"] >= holdout_start].copy()

    if development.empty:
        raise RuntimeError("Development dataset is empty.")

    if validation.empty:
        raise RuntimeError("Validation dataset is empty.")

    if historical_holdout.empty:
        raise RuntimeError("Historical holdout dataset is empty.")

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    development_file = output_directory / (
        f"{symbol}_{MARKET}_" f"{timeframe}_development.csv"
    )

    validation_file = output_directory / (
        f"{symbol}_{MARKET}_" f"{timeframe}_validation.csv"
    )

    holdout_file = output_directory / (
        f"{symbol}_{MARKET}_" f"{timeframe}_historical_holdout.csv"
    )

    development.to_csv(
        development_file,
        index=False,
    )

    validation.to_csv(
        validation_file,
        index=False,
    )

    historical_holdout.to_csv(
        holdout_file,
        index=False,
    )

    print()
    print("=" * 65)
    print("TIME SERIES DATA SPLIT")
    print("=" * 65)

    print()
    print(f"Symbol:     {symbol}")
    print(f"Market:     {MARKET.upper()}")
    print(f"Timeframe:  {timeframe}")

    print()
    print(f"Full period: " f"{start_date} " f"→ {end_date}")

    print()
    print("-" * 65)
    print("DEVELOPMENT")
    print("-" * 65)

    print(f"Candles: {len(development)}")

    print(f"From:    " f"{development['timestamp'].min()}")

    print(f"To:      " f"{development['timestamp'].max()}")

    print()
    print("-" * 65)
    print("VALIDATION")
    print("-" * 65)

    print(f"Candles: {len(validation)}")

    print(f"From:    " f"{validation['timestamp'].min()}")

    print(f"To:      " f"{validation['timestamp'].max()}")

    print()
    print("-" * 65)
    print("HISTORICAL HOLDOUT")
    print("-" * 65)

    print(f"Candles: " f"{len(historical_holdout)}")

    print(f"From:    " f"{historical_holdout['timestamp'].min()}")

    print(f"To:      " f"{historical_holdout['timestamp'].max()}")

    print()
    print("-" * 65)

    print(f"Saved development: " f"{development_file}")

    print(f"Saved validation:  " f"{validation_file}")

    print(f"Saved holdout:     " f"{holdout_file}")


if __name__ == "__main__":
    main()
