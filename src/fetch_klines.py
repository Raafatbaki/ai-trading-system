from datetime import datetime, timedelta, timezone
from pathlib import Path
import time

import pandas as pd
import requests

BASE_URL = "https://api.bybit.com"

# Market configuration
SYMBOL = "BTCUSDT"
CATEGORY = "spot"
INTERVAL = "15"

# Historical data configuration
LIMIT = 1000
DAYS_TO_DOWNLOAD = 365

# API protection
REQUEST_DELAY_SECONDS = 0.7
MAX_RETRIES = 6

# Output
OUTPUT_FILE = Path("data/BTCUSDT_spot_15m.csv")


def fetch_batch(end_time_ms: int) -> list:
    """
    Fetch one batch of historical Spot candles from Bybit.
    Retries automatically when the API rate limit is reached.
    """

    url = f"{BASE_URL}/v5/market/kline"

    params = {
        "category": CATEGORY,
        "symbol": SYMBOL,
        "interval": INTERVAL,
        "limit": LIMIT,
        "end": end_time_ms,
    }

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(
                url,
                params=params,
                timeout=20,
            )

            if response.status_code == 429:
                wait_seconds = 2**attempt

                print(f"HTTP rate limit reached. " f"Waiting {wait_seconds} seconds...")

                time.sleep(wait_seconds)
                continue

            response.raise_for_status()

            payload = response.json()

            ret_code = payload.get("retCode")
            ret_msg = payload.get("retMsg", "")

            if ret_code == 0:
                return payload["result"]["list"]

            if "rate limit" in ret_msg.lower() or "too many visits" in ret_msg.lower():
                wait_seconds = 2**attempt

                print(
                    f"Bybit rate limit reached. " f"Waiting {wait_seconds} seconds..."
                )

                time.sleep(wait_seconds)
                continue

            raise RuntimeError(f"Bybit API error " f"(retCode={ret_code}): {ret_msg}")

        except requests.RequestException as error:
            if attempt == MAX_RETRIES - 1:
                raise RuntimeError(
                    f"Network error while contacting Bybit: {error}"
                ) from error

            wait_seconds = 2**attempt

            print(f"Network error. " f"Waiting {wait_seconds} seconds before retry...")

            time.sleep(wait_seconds)

    raise RuntimeError("Bybit API failed after maximum retry attempts.")


def fetch_historical_klines() -> pd.DataFrame:
    """
    Download historical Spot candles for the configured period.
    """

    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=DAYS_TO_DOWNLOAD)

    start_time_ms = int(start_date.timestamp() * 1000)
    current_end_ms = int(end_date.timestamp() * 1000)

    all_rows = []

    while True:
        rows = fetch_batch(current_end_ms)

        if not rows:
            break

        all_rows.extend(rows)

        oldest_timestamp = min(int(row[0]) for row in rows)

        oldest_datetime = pd.to_datetime(
            oldest_timestamp,
            unit="ms",
            utc=True,
        )

        print(f"Downloaded: {len(all_rows)} candles | " f"Oldest: {oldest_datetime}")

        if oldest_timestamp <= start_time_ms:
            break

        current_end_ms = oldest_timestamp - 1

        time.sleep(REQUEST_DELAY_SECONDS)

    if not all_rows:
        raise RuntimeError("No candle data was returned by Bybit.")

    dataframe = pd.DataFrame(
        all_rows,
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

    # Keep only the requested time period.
    dataframe = dataframe[dataframe["timestamp"] >= start_date]

    # Remove duplicates and sort chronologically.
    dataframe = (
        dataframe.drop_duplicates(subset="timestamp")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    # Remove the currently open/incomplete 15-minute candle.
    now = pd.Timestamp.now(tz="UTC")
    current_candle_start = now.floor("15min")

    dataframe = dataframe[dataframe["timestamp"] < current_candle_start].reset_index(
        drop=True
    )

    return dataframe


def main() -> None:
    print("Market data configuration:")
    print(f"Symbol:     {SYMBOL}")
    print(f"Market:     {CATEGORY.upper()}")
    print(f"Timeframe:  {INTERVAL}m")
    print(f"History:    {DAYS_TO_DOWNLOAD} days")
    print()

    dataframe = fetch_historical_klines()

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print("Download completed.")
    print(f"Market:  {CATEGORY.upper()}")
    print(f"Candles: {len(dataframe)}")
    print(f"From:    {dataframe['timestamp'].min()}")
    print(f"To:      {dataframe['timestamp'].max()}")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
