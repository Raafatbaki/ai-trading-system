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


BREAKOUT_DAYS = 20
ATR_DAYS = 14

FAST_TREND_DAYS = 50
SLOW_TREND_DAYS = 200

BREAKOUT_ATR_BUFFER = 0.25
ATR_TRAIL_MULTIPLIER = 3.0


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="ATR volatility-adjusted trend breakout."
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


def get_periods(
    timeframe: str,
) -> dict:

    candles_per_day = CANDLES_PER_DAY[timeframe]

    return {
        "breakout": (BREAKOUT_DAYS * candles_per_day),
        "atr": (ATR_DAYS * candles_per_day),
        "fast_trend": (FAST_TREND_DAYS * candles_per_day),
        "slow_trend": (SLOW_TREND_DAYS * candles_per_day),
    }


def calculate_true_range(
    dataframe: pd.DataFrame,
) -> pd.Series:

    previous_close = dataframe["close"].shift(1)

    high_low = dataframe["high"] - dataframe["low"]

    high_previous = (dataframe["high"] - previous_close).abs()

    low_previous = (dataframe["low"] - previous_close).abs()

    true_range = pd.concat(
        [
            high_low,
            high_previous,
            low_previous,
        ],
        axis=1,
    ).max(axis=1)

    return true_range


def calculate_features(
    dataframe: pd.DataFrame,
    timeframe: str,
) -> pd.DataFrame:

    dataframe = dataframe.copy()

    periods = get_periods(timeframe)

    true_range = calculate_true_range(dataframe)

    dataframe["atr"] = true_range.ewm(
        alpha=(1 / periods["atr"]),
        adjust=False,
        min_periods=periods["atr"],
    ).mean()

    dataframe["ema_fast"] = (
        dataframe["close"]
        .ewm(
            span=periods["fast_trend"],
            adjust=False,
            min_periods=periods["fast_trend"],
        )
        .mean()
    )

    dataframe["ema_slow"] = (
        dataframe["close"]
        .ewm(
            span=periods["slow_trend"],
            adjust=False,
            min_periods=periods["slow_trend"],
        )
        .mean()
    )

    dataframe["breakout_high"] = (
        dataframe["high"]
        .shift(1)
        .rolling(
            window=periods["breakout"],
            min_periods=periods["breakout"],
        )
        .max()
    )

    dataframe["breakout_threshold"] = dataframe["breakout_high"] + (
        BREAKOUT_ATR_BUFFER * dataframe["atr"]
    )

    dataframe["trend_up"] = dataframe["ema_fast"] > dataframe["ema_slow"]

    return dataframe


def generate_signals(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:

    dataframe = dataframe.copy()

    dataframe["signal"] = "NO SIGNAL"

    dataframe["atr_trailing_stop"] = float("nan")

    position_open = False

    highest_high_since_entry = None

    for index in dataframe.index:

        close = float(
            dataframe.at[
                index,
                "close",
            ]
        )

        high = float(
            dataframe.at[
                index,
                "high",
            ]
        )

        atr = dataframe.at[
            index,
            "atr",
        ]

        ema_fast = dataframe.at[
            index,
            "ema_fast",
        ]

        ema_slow = dataframe.at[
            index,
            "ema_slow",
        ]

        breakout_threshold = dataframe.at[
            index,
            "breakout_threshold",
        ]

        if (
            pd.isna(atr)
            or pd.isna(ema_fast)
            or pd.isna(ema_slow)
            or pd.isna(breakout_threshold)
        ):
            continue

        trend_up = float(ema_fast) > float(ema_slow)

        if not position_open:

            entry_condition = trend_up and close > float(breakout_threshold)

            if entry_condition:

                dataframe.at[
                    index,
                    "signal",
                ] = "BUY"

                position_open = True

                # Actual execution happens at
                # next candle open.
                #
                # Therefore we do not use the
                # signal candle high as a
                # post-entry high.
                highest_high_since_entry = None

        else:

            if highest_high_since_entry is None:
                highest_high_since_entry = high

            else:
                highest_high_since_entry = max(
                    highest_high_since_entry,
                    high,
                )

            trailing_stop = highest_high_since_entry - (
                ATR_TRAIL_MULTIPLIER * float(atr)
            )

            dataframe.at[
                index,
                "atr_trailing_stop",
            ] = trailing_stop

            exit_condition = close < trailing_stop or not trend_up

            if exit_condition:

                dataframe.at[
                    index,
                    "signal",
                ] = "EXIT"

                position_open = False

                highest_high_since_entry = None

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

    # Every independent dataset starts CASH.
    target = generate_signals(target)

    buy_count = int((target["signal"] == "BUY").sum())

    exit_count = int((target["signal"] == "EXIT").sum())

    hold_count = int((target["signal"] == "HOLD").sum())

    no_signal_count = int((target["signal"] == "NO SIGNAL").sum())

    output_file = Path(
        f"data/splits/"
        f"{symbol}_{MARKET}_{timeframe}_"
        f"{dataset}_atr_breakout_signals.csv"
    )

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    target.to_csv(
        output_file,
        index=False,
    )

    periods = get_periods(timeframe)

    print()
    print("=" * 76)
    print("ATR VOLATILITY-ADJUSTED TREND BREAKOUT")
    print("=" * 76)

    print()
    print(f"Symbol:          {symbol}")
    print(f"Market:          {MARKET.upper()}")
    print(f"Timeframe:       {timeframe}")
    print(f"Dataset:         {dataset.upper()}")
    print("Mode:            LONG / CASH")
    print("Short:           DISABLED")
    print("Leverage:        DISABLED")

    print()
    print("-" * 76)
    print("CONFIGURATION")
    print("-" * 76)

    print(f"Breakout:         {BREAKOUT_DAYS} days")
    print(f"Breakout candles: {periods['breakout']}")

    print(f"ATR:              {ATR_DAYS} days")
    print(f"ATR candles:      {periods['atr']}")

    print(f"Fast trend:       EMA {FAST_TREND_DAYS} days")
    print(f"Slow trend:       EMA {SLOW_TREND_DAYS} days")

    print(f"Breakout buffer:  {BREAKOUT_ATR_BUFFER} ATR")

    print(f"ATR trail:        {ATR_TRAIL_MULTIPLIER} ATR")

    print("Execution:        NEXT CANDLE OPEN")

    print()
    print("-" * 76)
    print("SIGNALS")
    print("-" * 76)

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
                    "breakout_high",
                    "breakout_threshold",
                    "atr",
                    "ema_fast",
                    "ema_slow",
                ]
            ]
            .tail(10)
            .to_string()
        )

    print()
    print(f"Saved to: {output_file}")


if __name__ == "__main__":
    main()
