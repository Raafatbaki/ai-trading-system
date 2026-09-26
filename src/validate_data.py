from pathlib import Path

import pandas as pd

DATA_FILE = Path("data/BTCUSDT_spot_15m.csv")
EXPECTED_INTERVAL = pd.Timedelta(minutes=15)


def main():
    df = pd.read_csv(DATA_FILE, parse_dates=["timestamp"])

    print(f"Candles: {len(df)}")
    print(f"From:    {df['timestamp'].min()}")
    print(f"To:      {df['timestamp'].max()}")
    print()

    # 1. Duplicate timestamps
    duplicates = df["timestamp"].duplicated().sum()
    print(f"Duplicate timestamps: {duplicates}")

    # 2. Missing values
    missing_values = df.isna().sum().sum()
    print(f"Missing values:       {missing_values}")

    # 3. Check chronological order
    sorted_correctly = df["timestamp"].is_monotonic_increasing
    print(f"Chronological order:  {sorted_correctly}")

    # 4. Check gaps
    time_diff = df["timestamp"].diff()
    gaps = df[time_diff.notna() & (time_diff != EXPECTED_INTERVAL)]

    print(f"Time gaps:            {len(gaps)}")

    if not gaps.empty:
        print("\nFirst gaps:")
        for index in gaps.index[:10]:
            print(
                df.loc[index - 1, "timestamp"],
                "→",
                df.loc[index, "timestamp"],
            )

    # 5. Invalid prices
    invalid_prices = df[
        (df["open"] <= 0) | (df["high"] <= 0) | (df["low"] <= 0) | (df["close"] <= 0)
    ]

    print(f"Invalid prices:       {len(invalid_prices)}")

    # 6. OHLC consistency
    invalid_ohlc = df[
        (df["high"] < df["low"])
        | (df["high"] < df["open"])
        | (df["high"] < df["close"])
        | (df["low"] > df["open"])
        | (df["low"] > df["close"])
    ]

    print(f"Invalid OHLC candles: {len(invalid_ohlc)}")

    # 7. Negative volume
    invalid_volume = df[df["volume"] < 0]
    print(f"Invalid volume:       {len(invalid_volume)}")

    print()

    if (
        duplicates == 0
        and missing_values == 0
        and sorted_correctly
        and len(gaps) == 0
        and len(invalid_prices) == 0
        and len(invalid_ohlc) == 0
        and len(invalid_volume) == 0
    ):
        print("✅ Data validation PASSED")
    else:
        print("❌ Data validation FAILED")


if __name__ == "__main__":
    main()
