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

SUPPORTED_DATASETS = [
    "development",
    "validation",
    "historical_holdout",
]


CANDLES_PER_DAY = {
    "15m": 96,
    "1h": 24,
    "4h": 6,
}


ENTRY_DAYS = 20
EXIT_DAYS = 10


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Donchian 20/10 long-only breakout strategy."
    )

    parser.add_argument(
        "--symbol",
        choices=SUPPORTED_SYMBOLS,
        default="BTCUSDT",
    )

    parser.add_argument(
        "--timeframe",
        choices=SUPPORTED_TIMEFRAMES,
        default="1h",
    )

    parser.add_argument(
        "--dataset",
        choices=SUPPORTED_DATASETS,
        default="development",
    )

    return parser.parse_args()


def get_periods(
    timeframe: str,
) -> tuple[int, int]:
    candles_per_day = CANDLES_PER_DAY[timeframe]

    entry_window = ENTRY_DAYS * candles_per_day

    exit_window = EXIT_DAYS * candles_per_day

    return (
        entry_window,
        exit_window,
    )


def get_dataset_file(
    symbol: str,
    timeframe: str,
    dataset: str,
) -> Path:
    return Path(f"data/splits/" f"{symbol}_{MARKET}_{timeframe}_{dataset}.csv")


def load_csv(
    file_path: Path,
) -> pd.DataFrame:
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")

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


def load_dataset_with_history(
    symbol: str,
    timeframe: str,
    dataset: str,
) -> pd.DataFrame:
    """
    Load previous datasets only to calculate
    Donchian channels at the beginning of the
    target dataset.

    Trading state still starts in CASH at the
    beginning of the target dataset.
    """

    frames = []

    if dataset == "development":
        development = load_csv(
            get_dataset_file(
                symbol,
                timeframe,
                "development",
            )
        )

        development["_target"] = True

        frames.append(development)

    elif dataset == "validation":
        development = load_csv(
            get_dataset_file(
                symbol,
                timeframe,
                "development",
            )
        )

        validation = load_csv(
            get_dataset_file(
                symbol,
                timeframe,
                "validation",
            )
        )

        development["_target"] = False
        validation["_target"] = True

        frames.extend(
            [
                development,
                validation,
            ]
        )

    elif dataset == "historical_holdout":
        development = load_csv(
            get_dataset_file(
                symbol,
                timeframe,
                "development",
            )
        )

        validation = load_csv(
            get_dataset_file(
                symbol,
                timeframe,
                "validation",
            )
        )

        holdout = load_csv(
            get_dataset_file(
                symbol,
                timeframe,
                "historical_holdout",
            )
        )

        development["_target"] = False
        validation["_target"] = False
        holdout["_target"] = True

        frames.extend(
            [
                development,
                validation,
                holdout,
            ]
        )

    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    dataframe = pd.concat(
        frames,
        ignore_index=True,
    )

    dataframe = (
        dataframe.sort_values("timestamp")
        .drop_duplicates(
            subset=["timestamp"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    return dataframe


def calculate_channels(
    dataframe: pd.DataFrame,
    entry_window: int,
    exit_window: int,
) -> pd.DataFrame:
    dataframe = dataframe.copy()

    # IMPORTANT:
    # shift(1) means today's/current candle
    # is NOT included in the channel.
    #
    # This prevents look-ahead bias.

    dataframe["donchian_high_20d"] = (
        dataframe["high"]
        .shift(1)
        .rolling(
            window=entry_window,
            min_periods=entry_window,
        )
        .max()
    )

    dataframe["donchian_low_10d"] = (
        dataframe["low"]
        .shift(1)
        .rolling(
            window=exit_window,
            min_periods=exit_window,
        )
        .min()
    )

    dataframe["entry_breakout"] = dataframe["close"] > dataframe["donchian_high_20d"]

    dataframe["exit_breakout"] = dataframe["close"] < dataframe["donchian_low_10d"]

    return dataframe


def generate_signals(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    dataframe = dataframe.copy()

    dataframe["signal"] = "NO SIGNAL"

    position_open = False

    for index in dataframe.index:
        entry_ready = pd.notna(
            dataframe.at[
                index,
                "donchian_high_20d",
            ]
        )

        exit_ready = pd.notna(
            dataframe.at[
                index,
                "donchian_low_10d",
            ]
        )

        if not position_open:
            if entry_ready and bool(
                dataframe.at[
                    index,
                    "entry_breakout",
                ]
            ):
                dataframe.at[
                    index,
                    "signal",
                ] = "BUY"

                position_open = True

            else:
                dataframe.at[
                    index,
                    "signal",
                ] = "NO SIGNAL"

        else:
            if exit_ready and bool(
                dataframe.at[
                    index,
                    "exit_breakout",
                ]
            ):
                dataframe.at[
                    index,
                    "signal",
                ] = "EXIT"

                position_open = False

            else:
                dataframe.at[
                    index,
                    "signal",
                ] = "HOLD"

    return dataframe


def main():
    args = parse_arguments()

    symbol = args.symbol
    timeframe = args.timeframe
    dataset = args.dataset

    (
        entry_window,
        exit_window,
    ) = get_periods(timeframe)

    dataframe = load_dataset_with_history(
        symbol=symbol,
        timeframe=timeframe,
        dataset=dataset,
    )

    dataframe = calculate_channels(
        dataframe=dataframe,
        entry_window=entry_window,
        exit_window=exit_window,
    )

    # Only the selected dataset is evaluated.
    # Previous data was used only as history
    # for channel calculation.
    target = dataframe[dataframe["_target"]].copy()

    target = target.drop(columns=["_target"]).reset_index(drop=True)

    # Start every dataset evaluation from CASH.
    target = generate_signals(target)

    buy_count = int((target["signal"] == "BUY").sum())

    exit_count = int((target["signal"] == "EXIT").sum())

    hold_count = int((target["signal"] == "HOLD").sum())

    no_signal_count = int((target["signal"] == "NO SIGNAL").sum())

    output_file = Path(
        f"data/splits/"
        f"{symbol}_{MARKET}_{timeframe}_"
        f"{dataset}_donchian_20_10_signals.csv"
    )

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    target.to_csv(
        output_file,
        index=False,
    )

    print()
    print("=" * 72)
    print("DONCHIAN 20/10 BREAKOUT")
    print("=" * 72)

    print()
    print(f"Symbol:          {symbol}")
    print(f"Market:          {MARKET.upper()}")
    print(f"Timeframe:       {timeframe}")
    print(f"Dataset:         {dataset.upper()}")
    print("Mode:            LONG / CASH")
    print("Short:           DISABLED")
    print("Leverage:        DISABLED")

    print()
    print("-" * 72)
    print("CONFIGURATION")
    print("-" * 72)

    print(f"Entry channel:   {ENTRY_DAYS} days")
    print(f"Entry candles:   {entry_window}")
    print(f"Exit channel:    {EXIT_DAYS} days")
    print(f"Exit candles:    {exit_window}")
    print("Decision:        EVERY CLOSED CANDLE")
    print("Execution:       NEXT CANDLE OPEN")

    print()
    print("-" * 72)
    print("SIGNALS")
    print("-" * 72)

    print(f"Candles:       {len(target)}")
    print(f"BUY signals:   {buy_count}")
    print(f"EXIT signals:  {exit_count}")
    print(f"HOLD candles:  {hold_count}")
    print(f"NO SIGNAL:     {no_signal_count}")

    buys = target[target["signal"] == "BUY"]

    if not buys.empty:
        print()
        print("Last BUY signals:")

        columns = [
            "timestamp",
            "close",
            "donchian_high_20d",
            "donchian_low_10d",
        ]

        print(buys[columns].tail(10).to_string())

    print()
    print(f"Saved to: {output_file}")


if __name__ == "__main__":
    main()
