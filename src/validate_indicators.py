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

WARMUP_CANDLES = 250


def parse_arguments():
    parser = argparse.ArgumentParser(description="Validate technical indicators.")

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
        help="Timeframe to validate.",
    )

    return parser.parse_args()


def validate_indicators(
    data_file: Path,
) -> bool:
    dataframe = pd.read_csv(
        data_file,
        parse_dates=["timestamp"],
    )

    print()
    print("=" * 60)
    print("INDICATOR VALIDATION")
    print("=" * 60)

    print()
    print(f"File:     {data_file}")
    print(f"Candles:  {len(dataframe)}")

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

    missing_columns = [
        column for column in indicator_columns if column not in dataframe.columns
    ]

    if missing_columns:
        print()
        print(f"❌ Missing columns: " f"{missing_columns}")
        return False

    if len(dataframe) <= WARMUP_CANDLES:
        print()
        print("❌ Not enough candles " "for indicator validation.")
        return False

    # Ignore warm-up area.
    validated = dataframe.iloc[WARMUP_CANDLES:].copy()

    indicator_data = validated[indicator_columns]

    missing_values = indicator_data.isna().sum().sum()

    infinite_values = int(np.isinf(indicator_data.to_numpy()).sum())

    invalid_rsi = validated[(validated["rsi_14"] < 0) | (validated["rsi_14"] > 100)]

    invalid_adx = validated[(validated["adx_14"] < 0) | (validated["adx_14"] > 100)]

    invalid_plus_di = validated[
        (validated["plus_di_14"] < 0) | (validated["plus_di_14"] > 100)
    ]

    invalid_minus_di = validated[
        (validated["minus_di_14"] < 0) | (validated["minus_di_14"] > 100)
    ]

    invalid_atr = validated[validated["atr_14"] <= 0]

    invalid_volume_sma = validated[validated["volume_sma_20"] < 0]

    invalid_volume_ratio = validated[validated["volume_ratio"] < 0]

    invalid_donchian = validated[
        validated["donchian_high_20"] < validated["donchian_low_20"]
    ]

    invalid_ema = validated[
        (validated["ema_20"] <= 0)
        | (validated["ema_50"] <= 0)
        | (validated["ema_200"] <= 0)
    ]

    print()
    print(f"Missing indicator values: " f"{missing_values}")

    print(f"Infinite values:          " f"{infinite_values}")

    print(f"Invalid RSI:              " f"{len(invalid_rsi)}")

    print(f"Invalid ADX:              " f"{len(invalid_adx)}")

    print(f"Invalid +DI:              " f"{len(invalid_plus_di)}")

    print(f"Invalid -DI:              " f"{len(invalid_minus_di)}")

    print(f"Invalid ATR:              " f"{len(invalid_atr)}")

    print(f"Invalid Volume SMA:       " f"{len(invalid_volume_sma)}")

    print(f"Invalid Volume Ratio:     " f"{len(invalid_volume_ratio)}")

    print(f"Invalid Donchian:         " f"{len(invalid_donchian)}")

    print(f"Invalid EMA:              " f"{len(invalid_ema)}")

    passed = (
        missing_values == 0
        and infinite_values == 0
        and len(invalid_rsi) == 0
        and len(invalid_adx) == 0
        and len(invalid_plus_di) == 0
        and len(invalid_minus_di) == 0
        and len(invalid_atr) == 0
        and len(invalid_volume_sma) == 0
        and len(invalid_volume_ratio) == 0
        and len(invalid_donchian) == 0
        and len(invalid_ema) == 0
    )

    print()

    if passed:
        print("✅ Indicator validation PASSED")
    else:
        print("❌ Indicator validation FAILED")

    return passed


def main():
    args = parse_arguments()

    symbol = args.symbol
    timeframe = args.timeframe

    data_file = Path(f"data/" f"{symbol}_{MARKET}_" f"{timeframe}_indicators.csv")

    if not data_file.exists():
        raise FileNotFoundError(f"Indicator file not found: " f"{data_file}")

    print()
    print(f"Symbol:    {symbol}")
    print(f"Market:    {MARKET.upper()}")
    print(f"Timeframe: {timeframe}")

    validate_indicators(
        data_file=data_file,
    )


if __name__ == "__main__":
    main()
