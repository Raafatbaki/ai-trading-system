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

SUPPORTED_REBALANCE = [
    "weekly",
    "daily",
]

MOMENTUM_LOOKBACK_DAYS = 28
LONG_TREND_DAYS = 200

DATA_DIRECTORY = Path("data/splits")


TIMEFRAME_CONFIG = {
    "15m": {
        "candles_per_day": 96,
        "label": "15 minutes",
    },
    "1h": {
        "candles_per_day": 24,
        "label": "1 hour",
    },
    "4h": {
        "candles_per_day": 6,
        "label": "4 hours",
    },
}


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Long-only Time-Series Momentum V2 " "with long-term trend filter."
        )
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

    parser.add_argument(
        "--rebalance",
        choices=SUPPORTED_REBALANCE,
        default="weekly",
    )

    return parser.parse_args()


def get_strategy_periods(
    timeframe: str,
) -> tuple[int, int]:
    candles_per_day = TIMEFRAME_CONFIG[timeframe]["candles_per_day"]

    momentum_lookback = MOMENTUM_LOOKBACK_DAYS * candles_per_day

    long_trend_window = LONG_TREND_DAYS * candles_per_day

    return (
        momentum_lookback,
        long_trend_window,
    )


def dataset_file(
    symbol: str,
    timeframe: str,
    dataset: str,
) -> Path:
    return DATA_DIRECTORY / (f"{symbol}_{MARKET}_" f"{timeframe}_{dataset}.csv")


def load_dataset_with_history(
    symbol: str,
    timeframe: str,
    dataset: str,
    required_history: int,
) -> pd.DataFrame:
    target_file = dataset_file(
        symbol,
        timeframe,
        dataset,
    )

    if not target_file.exists():
        raise FileNotFoundError(f"Dataset not found: {target_file}")

    target = pd.read_csv(
        target_file,
        parse_dates=["timestamp"],
    )

    target["is_target"] = True

    target = (
        target.sort_values("timestamp")
        .drop_duplicates(subset="timestamp")
        .reset_index(drop=True)
    )

    if dataset == "development":
        return target

    history_parts = []

    development_file = dataset_file(
        symbol,
        timeframe,
        "development",
    )

    if development_file.exists():
        development = pd.read_csv(
            development_file,
            parse_dates=["timestamp"],
        )

        history_parts.append(development)

    if dataset == "historical_holdout":
        validation_file = dataset_file(
            symbol,
            timeframe,
            "validation",
        )

        if validation_file.exists():
            validation = pd.read_csv(
                validation_file,
                parse_dates=["timestamp"],
            )

            history_parts.append(validation)

    if not history_parts:
        raise RuntimeError("Historical warm-up data not found.")

    history = pd.concat(
        history_parts,
        ignore_index=True,
    )

    history = (
        history.sort_values("timestamp")
        .drop_duplicates(subset="timestamp")
        .reset_index(drop=True)
    )

    history = history.tail(required_history + 2).copy()

    history["is_target"] = False

    combined = pd.concat(
        [
            history,
            target,
        ],
        ignore_index=True,
    )

    return (
        combined.sort_values("timestamp")
        .drop_duplicates(
            subset="timestamp",
            keep="last",
        )
        .reset_index(drop=True)
    )


def calculate_features(
    dataframe: pd.DataFrame,
    momentum_lookback: int,
    long_trend_window: int,
    rebalance: str,
) -> pd.DataFrame:
    dataframe = dataframe.copy()

    # فقط معلومات الشمعة المغلقة السابقة.
    previous_close = dataframe["close"].shift(1)

    historical_close = previous_close.shift(momentum_lookback)

    dataframe["momentum_4w"] = previous_close / historical_close - 1

    dataframe["ma_200d"] = previous_close.rolling(
        window=long_trend_window,
        min_periods=long_trend_window,
    ).mean()

    dataframe["previous_close"] = previous_close

    dataframe["above_200d_ma"] = dataframe["previous_close"] > dataframe["ma_200d"]

    if rebalance == "weekly":
        dataframe["is_rebalance"] = (
            (dataframe["timestamp"].dt.weekday == 0)
            & (dataframe["timestamp"].dt.hour == 0)
            & (dataframe["timestamp"].dt.minute == 0)
        )

    elif rebalance == "daily":
        dataframe["is_rebalance"] = (dataframe["timestamp"].dt.hour == 0) & (
            dataframe["timestamp"].dt.minute == 0
        )

    else:
        raise ValueError(f"Unsupported rebalance: {rebalance}")

    return dataframe


def generate_signals(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    dataframe = dataframe.copy()

    signals = []
    position_open = False

    for row in dataframe.itertuples():
        momentum = row.momentum_4w
        moving_average = row.ma_200d

        if not row.is_rebalance:
            signals.append("HOLD" if position_open else "NO_SIGNAL")
            continue

        if pd.isna(momentum) or pd.isna(moving_average):
            signals.append("HOLD" if position_open else "NO_SIGNAL")
            continue

        long_condition = momentum > 0 and row.above_200d_ma

        if long_condition:
            if position_open:
                signals.append("HOLD")
            else:
                signals.append("BUY")
                position_open = True

        else:
            if position_open:
                signals.append("EXIT")
                position_open = False
            else:
                signals.append("NO_SIGNAL")

    dataframe["signal"] = signals

    dataframe["buy_signal"] = dataframe["signal"] == "BUY"

    dataframe["exit_signal"] = dataframe["signal"] == "EXIT"

    return dataframe


def main():
    args = parse_arguments()

    symbol = args.symbol
    timeframe = args.timeframe
    dataset = args.dataset
    rebalance = args.rebalance

    (
        momentum_lookback,
        long_trend_window,
    ) = get_strategy_periods(timeframe)

    required_history = max(
        momentum_lookback,
        long_trend_window,
    )

    dataframe = load_dataset_with_history(
        symbol=symbol,
        timeframe=timeframe,
        dataset=dataset,
        required_history=required_history,
    )

    dataframe = calculate_features(
        dataframe=dataframe,
        momentum_lookback=momentum_lookback,
        long_trend_window=long_trend_window,
        rebalance=rebalance,
    )

    target = dataframe[dataframe["is_target"]].copy()

    target = target.sort_values("timestamp").reset_index(drop=True)

    target = generate_signals(target)

    output_file = DATA_DIRECTORY / (
        f"{symbol}_{MARKET}_{timeframe}_"
        f"{dataset}_tsmom_v2_"
        f"{rebalance}_signals.csv"
    )

    target.to_csv(
        output_file,
        index=False,
    )

    buy_signals = target[target["signal"] == "BUY"]

    exit_signals = target[target["signal"] == "EXIT"]

    hold_signals = target[target["signal"] == "HOLD"]

    no_signals = target[target["signal"] == "NO_SIGNAL"]

    rebalance_points = target[target["is_rebalance"]]

    print()
    print("=" * 70)
    print("TIME-SERIES MOMENTUM V2")
    print("=" * 70)

    print()
    print(f"Symbol:          {symbol}")
    print(f"Market:          {MARKET.upper()}")
    print(f"Timeframe:       {timeframe}")
    print(f"Frame size:      " f"{TIMEFRAME_CONFIG[timeframe]['label']}")
    print(f"Dataset:         {dataset.upper()}")
    print(f"Rebalance:       {rebalance.upper()}")
    print("Mode:            LONG / CASH")
    print("Short:           DISABLED")
    print("Leverage:        DISABLED")

    print()
    print("-" * 70)
    print("CONFIGURATION")
    print("-" * 70)

    print(f"Momentum:        " f"{MOMENTUM_LOOKBACK_DAYS} days")

    print(f"Momentum candles: " f"{momentum_lookback}")

    print(f"Trend filter:    " f"{LONG_TREND_DAYS} days")

    print(f"Trend candles:   " f"{long_trend_window}")

    print(f"Decision points: " f"{len(rebalance_points)}")

    print()
    print("-" * 70)
    print("SIGNALS")
    print("-" * 70)

    print(f"Candles:       {len(target)}")
    print(f"BUY signals:   {len(buy_signals)}")
    print(f"EXIT signals:  {len(exit_signals)}")
    print(f"HOLD candles:  {len(hold_signals)}")
    print(f"NO SIGNAL:     {len(no_signals)}")

    if not buy_signals.empty:
        print()
        print("Last BUY signals:")

        columns = [
            "timestamp",
            "close",
            "momentum_4w",
            "previous_close",
            "ma_200d",
        ]

        print(buy_signals[columns].tail(10))

    print()
    print(f"Saved to: {output_file}")


if __name__ == "__main__":
    main()
