import argparse
from pathlib import Path

import numpy as np
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


RSI_PERIOD = 14
ADX_PERIOD = 14
ATR_PERIOD = 14

VOLUME_PERIOD = 20
DONCHIAN_PERIOD = 20


def parse_arguments():
    parser = argparse.ArgumentParser(description="Calculate technical indicators.")

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
        help="Timeframe to process.",
    )

    return parser.parse_args()


def wilder_smoothing(
    series: pd.Series,
    period: int,
) -> pd.Series:
    return series.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()


def add_ema(
    dataframe: pd.DataFrame,
) -> None:
    dataframe["ema_20"] = (
        dataframe["close"]
        .ewm(
            span=20,
            adjust=False,
        )
        .mean()
    )

    dataframe["ema_50"] = (
        dataframe["close"]
        .ewm(
            span=50,
            adjust=False,
        )
        .mean()
    )

    dataframe["ema_200"] = (
        dataframe["close"]
        .ewm(
            span=200,
            adjust=False,
        )
        .mean()
    )


def add_rsi(
    dataframe: pd.DataFrame,
) -> None:
    delta = dataframe["close"].diff()

    gain = delta.clip(lower=0)

    loss = -delta.clip(upper=0)

    average_gain = wilder_smoothing(
        gain,
        RSI_PERIOD,
    )

    average_loss = wilder_smoothing(
        loss,
        RSI_PERIOD,
    )

    relative_strength = average_gain / average_loss.replace(
        0,
        np.nan,
    )

    rsi = 100 - (100 / (1 + relative_strength))

    # If there were no losses during the
    # calculation window, RSI should be 100.
    rsi = rsi.where(
        average_loss != 0,
        100.0,
    )

    # If there were no gains and no losses,
    # RSI is neutral.
    rsi = rsi.where(
        ~((average_gain == 0) & (average_loss == 0)),
        50.0,
    )

    dataframe["rsi_14"] = rsi


def calculate_true_range(
    dataframe: pd.DataFrame,
) -> pd.Series:
    previous_close = dataframe["close"].shift(1)

    high_low = dataframe["high"] - dataframe["low"]

    high_previous_close = (dataframe["high"] - previous_close).abs()

    low_previous_close = (dataframe["low"] - previous_close).abs()

    return pd.concat(
        [
            high_low,
            high_previous_close,
            low_previous_close,
        ],
        axis=1,
    ).max(axis=1)


def add_atr(
    dataframe: pd.DataFrame,
) -> None:
    true_range = calculate_true_range(dataframe)

    dataframe["atr_14"] = wilder_smoothing(
        true_range,
        ATR_PERIOD,
    )


def add_adx(
    dataframe: pd.DataFrame,
) -> None:
    high_change = dataframe["high"].diff()

    low_change = -dataframe["low"].diff()

    plus_dm = pd.Series(
        np.where(
            (high_change > low_change) & (high_change > 0),
            high_change,
            0.0,
        ),
        index=dataframe.index,
    )

    minus_dm = pd.Series(
        np.where(
            (low_change > high_change) & (low_change > 0),
            low_change,
            0.0,
        ),
        index=dataframe.index,
    )

    true_range = calculate_true_range(dataframe)

    atr = wilder_smoothing(
        true_range,
        ADX_PERIOD,
    )

    atr_safe = atr.replace(
        0,
        np.nan,
    )

    plus_di = 100 * (
        wilder_smoothing(
            plus_dm,
            ADX_PERIOD,
        )
        / atr_safe
    )

    minus_di = 100 * (
        wilder_smoothing(
            minus_dm,
            ADX_PERIOD,
        )
        / atr_safe
    )

    denominator = (plus_di + minus_di).replace(
        0,
        np.nan,
    )

    dx = 100 * ((plus_di - minus_di).abs() / denominator)

    dataframe["plus_di_14"] = plus_di

    dataframe["minus_di_14"] = minus_di

    dataframe["adx_14"] = wilder_smoothing(
        dx,
        ADX_PERIOD,
    )


def add_volume_indicators(
    dataframe: pd.DataFrame,
) -> None:
    # Use only previous candles for the volume
    # average, avoiding current-candle leakage.
    previous_volume = dataframe["volume"].shift(1)

    dataframe["volume_sma_20"] = previous_volume.rolling(VOLUME_PERIOD).mean()

    volume_sma_safe = dataframe["volume_sma_20"].replace(
        0,
        np.nan,
    )

    dataframe["volume_ratio"] = dataframe["volume"] / volume_sma_safe


def add_donchian_channels(
    dataframe: pd.DataFrame,
) -> None:
    # Important:
    # Current candle must not define its own
    # breakout level.
    previous_high = dataframe["high"].shift(1)

    previous_low = dataframe["low"].shift(1)

    dataframe["donchian_high_20"] = previous_high.rolling(DONCHIAN_PERIOD).max()

    dataframe["donchian_low_20"] = previous_low.rolling(DONCHIAN_PERIOD).min()


def calculate_indicators(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    dataframe = dataframe.copy()

    add_ema(dataframe)

    add_rsi(dataframe)

    add_atr(dataframe)

    add_adx(dataframe)

    add_volume_indicators(dataframe)

    add_donchian_channels(dataframe)

    return dataframe


def validate_required_columns(
    dataframe: pd.DataFrame,
) -> None:
    required_columns = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "turnover",
    ]

    missing_columns = [
        column for column in required_columns if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError("Missing required columns: " f"{missing_columns}")


def main():
    args = parse_arguments()

    symbol = args.symbol
    timeframe = args.timeframe

    input_file = Path(f"data/" f"{symbol}_{MARKET}_" f"{timeframe}.csv")

    output_file = Path(f"data/" f"{symbol}_{MARKET}_" f"{timeframe}_indicators.csv")

    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: " f"{input_file}")

    dataframe = pd.read_csv(
        input_file,
        parse_dates=["timestamp"],
    )

    validate_required_columns(dataframe)

    dataframe = dataframe.sort_values("timestamp").reset_index(drop=True)

    dataframe = calculate_indicators(dataframe)

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe.to_csv(
        output_file,
        index=False,
    )

    print()
    print("=" * 60)
    print("INDICATOR CALCULATION")
    print("=" * 60)

    print()
    print(f"Symbol:    {symbol}")

    print(f"Market:    " f"{MARKET.upper()}")

    print(f"Timeframe: {timeframe}")

    print(f"Candles:   " f"{len(dataframe)}")

    print()
    print(f"Saved to: {output_file}")

    print()

    columns_to_show = [
        "timestamp",
        "close",
        "ema_20",
        "ema_50",
        "ema_200",
        "rsi_14",
        "adx_14",
        "atr_14",
        "volume_ratio",
        "donchian_high_20",
        "donchian_low_20",
    ]

    print(dataframe[columns_to_show].tail(5))


if __name__ == "__main__":
    main()
