from pathlib import Path

import numpy as np
import pandas as pd

INPUT_FILE = Path("data/BTCUSDT_spot_15m.csv")
OUTPUT_FILE = Path("data/BTCUSDT_spot_15m_indicators.csv")

RSI_PERIOD = 14
ADX_PERIOD = 14
ATR_PERIOD = 14

VOLUME_PERIOD = 20
DONCHIAN_PERIOD = 20


def wilder_smoothing(series: pd.Series, period: int) -> pd.Series:
    """
    Wilder's smoothing (RMA), used by RSI, ATR and ADX.
    """
    return series.ewm(
        alpha=1 / period,
        adjust=False,
        min_periods=period,
    ).mean()


def add_ema(dataframe: pd.DataFrame) -> None:
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


def add_rsi(dataframe: pd.DataFrame) -> None:
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

    relative_strength = average_gain / average_loss

    dataframe["rsi_14"] = 100 - (100 / (1 + relative_strength))


def calculate_true_range(
    dataframe: pd.DataFrame,
) -> pd.Series:
    previous_close = dataframe["close"].shift(1)

    high_low = dataframe["high"] - dataframe["low"]

    high_previous_close = (dataframe["high"] - previous_close).abs()

    low_previous_close = (dataframe["low"] - previous_close).abs()

    true_range = pd.concat(
        [
            high_low,
            high_previous_close,
            low_previous_close,
        ],
        axis=1,
    ).max(axis=1)

    return true_range


def add_atr(dataframe: pd.DataFrame) -> None:
    true_range = calculate_true_range(dataframe)

    dataframe["atr_14"] = wilder_smoothing(
        true_range,
        ATR_PERIOD,
    )


def add_adx(dataframe: pd.DataFrame) -> None:
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

    plus_di = 100 * (
        wilder_smoothing(
            plus_dm,
            ADX_PERIOD,
        )
        / atr
    )

    minus_di = 100 * (
        wilder_smoothing(
            minus_dm,
            ADX_PERIOD,
        )
        / atr
    )

    denominator = plus_di + minus_di

    dx = 100 * ((plus_di - minus_di).abs() / denominator.replace(0, np.nan))

    dataframe["plus_di_14"] = plus_di
    dataframe["minus_di_14"] = minus_di

    dataframe["adx_14"] = wilder_smoothing(
        dx,
        ADX_PERIOD,
    )


def add_volume_indicators(
    dataframe: pd.DataFrame,
) -> None:
    # Previous 20 completed candles only.
    # Current candle is deliberately excluded.
    previous_volume = dataframe["volume"].shift(1)

    dataframe["volume_sma_20"] = previous_volume.rolling(VOLUME_PERIOD).mean()

    dataframe["volume_ratio"] = dataframe["volume"] / dataframe["volume_sma_20"]


def add_donchian_channels(
    dataframe: pd.DataFrame,
) -> None:
    # Important:
    # shift(1) prevents the current candle from influencing
    # its own breakout threshold.
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


def main() -> None:
    dataframe = pd.read_csv(
        INPUT_FILE,
        parse_dates=["timestamp"],
    )

    dataframe = calculate_indicators(dataframe)

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    indicator_columns = [
        "timestamp",
        "close",
        "ema_20",
        "ema_50",
        "ema_200",
        "rsi_14",
        "adx_14",
        "atr_14",
        "volume_sma_20",
        "volume_ratio",
        "donchian_high_20",
        "donchian_low_20",
    ]

    print("Indicator calculation completed.")
    print(f"Candles: {len(dataframe)}")
    print(f"Saved to: {OUTPUT_FILE}")
    print()
    print(dataframe[indicator_columns].tail())


if __name__ == "__main__":
    main()
