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


FAST_EMA = 20
TREND_EMA = 50
LONG_EMA = 200


def parse_arguments():
    parser = argparse.ArgumentParser(description="Trend pullback strategy.")

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

    return (
        dataframe.sort_values("timestamp")
        .drop_duplicates(subset=["timestamp"])
        .reset_index(drop=True)
    )


def load_dataset_with_history(
    symbol: str,
    timeframe: str,
    dataset: str,
) -> pd.DataFrame:

    history_needed = LONG_EMA + 20

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

        development = development.tail(history_needed)

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

        validation = validation.tail(history_needed)

        frames.extend(
            [
                validation,
                holdout,
            ]
        )

    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    return (
        pd.concat(
            frames,
            ignore_index=True,
        )
        .sort_values("timestamp")
        .drop_duplicates(
            subset=["timestamp"],
            keep="last",
        )
        .reset_index(drop=True)
    )


def calculate_features(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:

    dataframe = dataframe.copy()

    dataframe["ema20"] = (
        dataframe["close"]
        .ewm(
            span=FAST_EMA,
            adjust=False,
        )
        .mean()
    )

    dataframe["ema50"] = (
        dataframe["close"]
        .ewm(
            span=TREND_EMA,
            adjust=False,
        )
        .mean()
    )

    dataframe["ema200"] = (
        dataframe["close"]
        .ewm(
            span=LONG_EMA,
            adjust=False,
        )
        .mean()
    )

    dataframe["previous_close"] = dataframe["close"].shift(1)

    dataframe["previous_ema20"] = dataframe["ema20"].shift(1)

    dataframe["trend_up"] = dataframe["ema50"] > dataframe["ema200"]

    dataframe["pullback_recovery"] = (
        dataframe["previous_close"] <= dataframe["previous_ema20"]
    ) & (dataframe["close"] > dataframe["ema20"])

    return dataframe


def generate_signals(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:

    dataframe = dataframe.copy()

    dataframe["signal"] = "NO SIGNAL"

    position_open = False

    for index in dataframe.index:

        ema20 = dataframe.at[
            index,
            "ema20",
        ]

        ema50 = dataframe.at[
            index,
            "ema50",
        ]

        ema200 = dataframe.at[
            index,
            "ema200",
        ]

        close = dataframe.at[
            index,
            "close",
        ]

        previous_close = dataframe.at[
            index,
            "previous_close",
        ]

        previous_ema20 = dataframe.at[
            index,
            "previous_ema20",
        ]

        if (
            pd.isna(ema20)
            or pd.isna(ema50)
            or pd.isna(ema200)
            or pd.isna(previous_close)
            or pd.isna(previous_ema20)
        ):
            continue

        trend_up = ema50 > ema200

        recovery = previous_close <= previous_ema20 and close > ema20

        if not position_open:

            if trend_up and recovery:

                dataframe.at[
                    index,
                    "signal",
                ] = "BUY"

                position_open = True

        else:

            exit_condition = close < ema50 or ema50 <= ema200

            if exit_condition:

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

    dataframe = calculate_features(dataframe)

    target = dataframe[dataframe["_target"]].copy()

    target = target.drop(columns=["_target"]).reset_index(drop=True)

    target = generate_signals(target)

    buy_count = int((target["signal"] == "BUY").sum())

    exit_count = int((target["signal"] == "EXIT").sum())

    hold_count = int((target["signal"] == "HOLD").sum())

    no_signal_count = int((target["signal"] == "NO SIGNAL").sum())

    output_file = Path(
        f"data/splits/"
        f"{symbol}_{MARKET}_{timeframe}_"
        f"{dataset}_trend_pullback_signals.csv"
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
    print("TREND PULLBACK")
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

    print(f"Pullback EMA:     {FAST_EMA}")
    print(f"Trend EMA:        {TREND_EMA}")
    print(f"Long EMA:         {LONG_EMA}")

    print("Trend:            EMA50 > EMA200")

    print("BUY:              cross back above EMA20")

    print("EXIT:             close < EMA50")

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
                    "ema20",
                    "ema50",
                    "ema200",
                ]
            ]
            .tail(10)
            .to_string()
        )

    print()
    print(f"Saved to: {output_file}")


if __name__ == "__main__":
    main()
