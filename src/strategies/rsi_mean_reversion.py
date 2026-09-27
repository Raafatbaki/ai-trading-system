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


RSI_PERIOD = 2
RSI_ENTRY = 10.0
RSI_EXIT = 50.0
TREND_DAYS = 200


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="RSI mean reversion with long-term trend filter."
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

    return (
        dataframe.sort_values("timestamp")
        .drop_duplicates(subset=["timestamp"])
        .reset_index(drop=True)
    )


def get_trend_window(
    timeframe: str,
) -> int:
    return TREND_DAYS * CANDLES_PER_DAY[timeframe]


def load_dataset_with_history(
    symbol: str,
    timeframe: str,
    dataset: str,
) -> pd.DataFrame:

    trend_window = get_trend_window(timeframe)

    history_needed = trend_window + 10

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


def calculate_rsi(
    series: pd.Series,
    period: int,
) -> pd.Series:

    delta = series.diff()

    gain = delta.clip(lower=0)

    loss = -delta.clip(upper=0)

    average_gain = gain.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    average_loss = loss.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()

    rs = average_gain / average_loss

    rsi = 100 - (100 / (1 + rs))

    rsi = rsi.where(
        average_loss != 0,
        100.0,
    )

    return rsi


def calculate_features(
    dataframe: pd.DataFrame,
    timeframe: str,
) -> pd.DataFrame:

    dataframe = dataframe.copy()

    trend_window = get_trend_window(timeframe)

    dataframe["rsi_2"] = calculate_rsi(
        dataframe["close"],
        RSI_PERIOD,
    )

    dataframe["ma_200d"] = (
        dataframe["close"]
        .rolling(
            window=trend_window,
            min_periods=trend_window,
        )
        .mean()
    )

    dataframe["trend_ok"] = dataframe["close"] > dataframe["ma_200d"]

    return dataframe


def generate_signals(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:

    dataframe = dataframe.copy()

    dataframe["signal"] = "NO SIGNAL"

    position_open = False

    for index in dataframe.index:

        rsi = dataframe.at[
            index,
            "rsi_2",
        ]

        ma = dataframe.at[
            index,
            "ma_200d",
        ]

        close = dataframe.at[
            index,
            "close",
        ]

        if pd.isna(rsi) or pd.isna(ma):
            continue

        trend_ok = close > ma

        if not position_open:

            if trend_ok and rsi <= RSI_ENTRY:
                dataframe.at[
                    index,
                    "signal",
                ] = "BUY"

                position_open = True

        else:

            if rsi >= RSI_EXIT:

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

    dataframe = calculate_features(
        dataframe=dataframe,
        timeframe=timeframe,
    )

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
        f"{dataset}_rsi_mr_signals.csv"
    )

    target.to_csv(
        output_file,
        index=False,
    )

    trend_window = get_trend_window(timeframe)

    print()
    print("=" * 72)
    print("RSI MEAN REVERSION + TREND FILTER")
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

    print(f"RSI period:       {RSI_PERIOD}")
    print(f"BUY RSI:          <= {RSI_ENTRY}")
    print(f"EXIT RSI:         >= {RSI_EXIT}")
    print(f"Trend filter:     {TREND_DAYS} days")
    print(f"Trend candles:    {trend_window}")
    print("Trend condition:  close > MA200D")
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
                    "rsi_2",
                    "ma_200d",
                ]
            ]
            .tail(10)
            .to_string()
        )

    print()
    print(f"Saved to: {output_file}")


if __name__ == "__main__":
    main()
