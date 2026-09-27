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


BOLLINGER_WINDOW = 20
BOLLINGER_STD = 2.0


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Bollinger Band mean-reversion strategy."
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

        # فقط آخر جزء صغير من Development
        # نحتاجه لحساب أول Bollinger Bands.
        development = development.tail(BOLLINGER_WINDOW + 5)

        frames.extend(
            [
                development,
                validation,
            ]
        )

    elif dataset == "historical_holdout":
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

        validation["_target"] = False
        holdout["_target"] = True

        validation = validation.tail(BOLLINGER_WINDOW + 5)

        frames.extend(
            [
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


def calculate_bollinger_bands(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    dataframe = dataframe.copy()

    dataframe["bb_middle"] = (
        dataframe["close"]
        .rolling(
            window=BOLLINGER_WINDOW,
            min_periods=BOLLINGER_WINDOW,
        )
        .mean()
    )

    rolling_std = (
        dataframe["close"]
        .rolling(
            window=BOLLINGER_WINDOW,
            min_periods=BOLLINGER_WINDOW,
        )
        .std()
    )

    dataframe["bb_upper"] = dataframe["bb_middle"] + BOLLINGER_STD * rolling_std

    dataframe["bb_lower"] = dataframe["bb_middle"] - BOLLINGER_STD * rolling_std

    return dataframe


def generate_signals(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    dataframe = dataframe.copy()

    dataframe["signal"] = "NO SIGNAL"

    position_open = False

    for index in dataframe.index:
        close = float(
            dataframe.at[
                index,
                "close",
            ]
        )

        middle = dataframe.at[
            index,
            "bb_middle",
        ]

        lower = dataframe.at[
            index,
            "bb_lower",
        ]

        if pd.isna(middle) or pd.isna(lower):
            dataframe.at[
                index,
                "signal",
            ] = "NO SIGNAL"

            continue

        # ---------------------------------------
        # CASH -> BUY
        # ---------------------------------------

        if not position_open:
            if close < float(lower):
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

        # ---------------------------------------
        # LONG -> HOLD / EXIT
        # ---------------------------------------

        else:
            if close >= float(middle):
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

    dataframe = load_dataset_with_history(
        symbol=symbol,
        timeframe=timeframe,
        dataset=dataset,
    )

    dataframe = calculate_bollinger_bands(dataframe)

    # Previous dataset data is used only
    # for indicator warm-up.
    target = dataframe[dataframe["_target"]].copy()

    target = target.drop(columns=["_target"]).reset_index(drop=True)

    # Every dataset evaluation starts in CASH.
    target = generate_signals(target)

    buy_count = int((target["signal"] == "BUY").sum())

    exit_count = int((target["signal"] == "EXIT").sum())

    hold_count = int((target["signal"] == "HOLD").sum())

    no_signal_count = int((target["signal"] == "NO SIGNAL").sum())

    output_file = Path(
        f"data/splits/"
        f"{symbol}_{MARKET}_{timeframe}_"
        f"{dataset}_bollinger_mr_signals.csv"
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
    print("BOLLINGER MEAN REVERSION")
    print("=" * 72)

    print()
    print(f"Symbol:          {symbol}")
    print(f"Market:          {MARKET.upper()}")
    print(f"Timeframe:       {timeframe}")
    print(f"Dataset:         {dataset.upper()}")

    print("Mode:            LONG / CASH")
    print("Short:           DISABLED")
    print("Leverage:        DISABLED")
    print("Stop loss:       DISABLED")

    print()
    print("-" * 72)
    print("CONFIGURATION")
    print("-" * 72)

    print(f"Bollinger window: {BOLLINGER_WINDOW} candles")
    print(f"Std deviations:   {BOLLINGER_STD}")
    print("BUY:              close < lower band")
    print("EXIT:             close >= middle band")
    print("Decision:         EVERY CLOSED CANDLE")
    print("Execution:        NEXT CANDLE OPEN")

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

        print(
            buys[
                [
                    "timestamp",
                    "close",
                    "bb_lower",
                    "bb_middle",
                    "bb_upper",
                ]
            ]
            .tail(10)
            .to_string()
        )

    print()
    print(f"Saved to: {output_file}")


if __name__ == "__main__":
    main()
