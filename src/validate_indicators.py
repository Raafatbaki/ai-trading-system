from pathlib import Path

import numpy as np
import pandas as pd

DATA_FILE = Path("data/BTCUSDT_spot_15m_indicators.csv")
WARMUP_CANDLES = 250


def main() -> None:
    dataframe = pd.read_csv(
        DATA_FILE,
        parse_dates=["timestamp"],
    )

    print(f"Candles: {len(dataframe)}")
    print()

    indicator_columns = [
        "ema_20",
        "ema_50",
        "ema_200",
        "rsi_14",
        "atr_14",
        "plus_di_14",
        "minus_di_14",
        "adx_14",
        "volume_sma_20",
        "volume_ratio",
        "donchian_high_20",
        "donchian_low_20",
    ]

    # The first candles naturally contain NaN values
    # because indicators need historical data.
    validated = dataframe.iloc[WARMUP_CANDLES:].copy()

    missing_values = validated[indicator_columns].isna().sum().sum()

    infinite_values = np.isinf(validated[indicator_columns].to_numpy()).sum()

    invalid_rsi = validated[(validated["rsi_14"] < 0) | (validated["rsi_14"] > 100)]

    invalid_adx = validated[(validated["adx_14"] < 0) | (validated["adx_14"] > 100)]

    invalid_atr = validated[validated["atr_14"] <= 0]

    invalid_volume_ratio = validated[validated["volume_ratio"] < 0]

    invalid_donchian = validated[
        validated["donchian_high_20"] < validated["donchian_low_20"]
    ]

    invalid_ema = validated[
        (validated["ema_20"] <= 0)
        | (validated["ema_50"] <= 0)
        | (validated["ema_200"] <= 0)
    ]

    print(f"Missing indicator values: {missing_values}")
    print(f"Infinite values:          {infinite_values}")
    print(f"Invalid RSI:              {len(invalid_rsi)}")
    print(f"Invalid ADX:              {len(invalid_adx)}")
    print(f"Invalid ATR:              {len(invalid_atr)}")
    print(f"Invalid volume ratio:     " f"{len(invalid_volume_ratio)}")
    print(f"Invalid Donchian:         " f"{len(invalid_donchian)}")
    print(f"Invalid EMA:              {len(invalid_ema)}")

    print()

    passed = (
        missing_values == 0
        and infinite_values == 0
        and len(invalid_rsi) == 0
        and len(invalid_adx) == 0
        and len(invalid_atr) == 0
        and len(invalid_volume_ratio) == 0
        and len(invalid_donchian) == 0
        and len(invalid_ema) == 0
    )

    if passed:
        print("✅ Indicator validation PASSED")
    else:
        print("❌ Indicator validation FAILED")


if __name__ == "__main__":
    main()
