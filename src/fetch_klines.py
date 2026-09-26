from pathlib import Path

import pandas as pd
import requests


BASE_URL = "https://api.bybit.com"
SYMBOL = "BTCUSDT"
CATEGORY = "linear"
INTERVAL = "15"
LIMIT = 1000

OUTPUT_FILE = Path("data/BTCUSDT_15m.csv")


def fetch_klines():
    url = f"{BASE_URL}/v5/market/kline"

    params = {
        "category": CATEGORY,
        "symbol": SYMBOL,
        "interval": INTERVAL,
        "limit": LIMIT,
    }

    response = requests.get(url, params=params, timeout=20)
    response.raise_for_status()

    payload = response.json()

    if payload["retCode"] != 0:
        raise RuntimeError(
            f"Bybit API error: {payload['retMsg']}"
        )

    rows = payload["result"]["list"]

    dataframe = pd.DataFrame(
        rows,
        columns=[
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "turnover",
        ],
    )

    dataframe["timestamp"] = pd.to_datetime(
        dataframe["timestamp"].astype("int64"),
        unit="ms",
        utc=True,
    )

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "turnover",
    ]

    dataframe[numeric_columns] = dataframe[numeric_columns].astype(float)

    # Bybit returns newest candle first.
    dataframe = dataframe.sort_values("timestamp").reset_index(drop=True)

    return dataframe


def main():
    dataframe = fetch_klines()

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(OUTPUT_FILE, index=False)

    print(f"Downloaded {len(dataframe)} candles.")
    print(f"Saved to: {OUTPUT_FILE}")
    print()
    print(dataframe.tail())


if __name__ == "__main__":
    main()