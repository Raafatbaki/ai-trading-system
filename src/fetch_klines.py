import argparse
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://api.bybit.com"
CATEGORY = "spot"

SUPPORTED_SYMBOLS = [
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
]

TIMEFRAMES = {
    "15m": {
        "bybit_interval": "15",
        "pandas_frequency": "15min",
    },
    "1h": {
        "bybit_interval": "60",
        "pandas_frequency": "1h",
    },
    "4h": {
        "bybit_interval": "240",
        "pandas_frequency": "4h",
    },
}

LIMIT = 1000
DEFAULT_DAYS = 365

REQUEST_DELAY_SECONDS = 0.7
MAX_RETRIES = 6


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Download historical Bybit Spot candles."
    )

    parser.add_argument(
        "--symbol",
        choices=SUPPORTED_SYMBOLS,
        default="BTCUSDT",
        help="Trading pair.",
    )

    parser.add_argument(
        "--timeframe",
        choices=TIMEFRAMES.keys(),
        default="1h",
        help="Candle timeframe.",
    )

    parser.add_argument(
        "--days",
        type=int,
        default=DEFAULT_DAYS,
        help="Number of historical days.",
    )

    return parser.parse_args()


def fetch_batch(
    symbol: str,
    interval: str,
    end_time_ms: int,
) -> list:

    url = f"{BASE_URL}/v5/market/kline"

    params = {
        "category": CATEGORY,
        "symbol": symbol,
        "interval": interval,
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

                print(f"HTTP rate limit. " f"Waiting {wait_seconds}s...")

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

                print(f"Bybit rate limit. " f"Waiting {wait_seconds}s...")

                time.sleep(wait_seconds)
                continue

            raise RuntimeError(
                f"Bybit API error " f"(retCode={ret_code}): " f"{ret_msg}"
            )

        except requests.RequestException as error:
            if attempt == MAX_RETRIES - 1:
                raise RuntimeError(f"Network error: {error}") from error

            wait_seconds = 2**attempt

            print(f"Network error. " f"Waiting {wait_seconds}s...")

            time.sleep(wait_seconds)

    raise RuntimeError("Bybit API failed after maximum retries.")


def fetch_historical_klines(
    symbol: str,
    timeframe: str,
    days: int,
) -> pd.DataFrame:

    config = TIMEFRAMES[timeframe]

    interval = config["bybit_interval"]

    pandas_frequency = config["pandas_frequency"]

    end_date = datetime.now(timezone.utc)

    start_date = end_date - timedelta(days=days)

    start_time_ms = int(start_date.timestamp() * 1000)

    current_end_ms = int(end_date.timestamp() * 1000)

    all_rows = []

    while True:
        rows = fetch_batch(
            symbol=symbol,
            interval=interval,
            end_time_ms=current_end_ms,
        )

        if not rows:
            break

        all_rows.extend(rows)

        oldest_timestamp = min(int(row[0]) for row in rows)

        oldest_datetime = pd.to_datetime(
            oldest_timestamp,
            unit="ms",
            utc=True,
        )

        print(
            f"Downloaded: " f"{len(all_rows)} candles | " f"Oldest: {oldest_datetime}"
        )

        if oldest_timestamp <= start_time_ms:
            break

        current_end_ms = oldest_timestamp - 1

        time.sleep(REQUEST_DELAY_SECONDS)

    if not all_rows:
        raise RuntimeError(f"No data returned for " f"{symbol}.")

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

    dataframe = dataframe[dataframe["timestamp"] >= start_date]

    dataframe = (
        dataframe.drop_duplicates(subset="timestamp")
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    # Remove currently open candle.
    now = pd.Timestamp.now(tz="UTC")

    current_candle_start = now.floor(pandas_frequency)

    dataframe = dataframe[dataframe["timestamp"] < current_candle_start].reset_index(
        drop=True
    )

    return dataframe


def main():
    args = parse_arguments()

    symbol = args.symbol
    timeframe = args.timeframe
    days = args.days

    output_file = Path(f"data/" f"{symbol}_{CATEGORY}_" f"{timeframe}.csv")

    print()
    print("=" * 60)
    print("MARKET DATA DOWNLOAD")
    print("=" * 60)

    print()
    print(f"Symbol:    {symbol}")
    print(f"Market:    {CATEGORY.upper()}")
    print(f"Timeframe: {timeframe}")
    print(f"History:   {days} days")
    print()

    dataframe = fetch_historical_klines(
        symbol=symbol,
        timeframe=timeframe,
        days=days,
    )

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe.to_csv(
        output_file,
        index=False,
    )

    print()
    print("Download completed.")
    print(f"Candles: {len(dataframe)}")
    print(f"From:    " f"{dataframe['timestamp'].min()}")
    print(f"To:      " f"{dataframe['timestamp'].max()}")
    print(f"Saved to: {output_file}")


if __name__ == "__main__":
    main()
