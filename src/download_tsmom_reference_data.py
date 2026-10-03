from pathlib import Path
import time

import pandas as pd
import requests

BASE_URL = "https://api.bybit.com"
ENDPOINT = "/v5/market/kline"

CATEGORY = "spot"
INTERVAL = "D"
LIMIT = 1000

SYMBOLS = [
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
]

OUTPUT_DIR = Path("data/reference_tsmom")

REQUEST_DELAY_SECONDS = 0.15


def fetch_batch(
    symbol: str,
    end_ms: int | None = None,
) -> list:

    params = {
        "category": CATEGORY,
        "symbol": symbol,
        "interval": INTERVAL,
        "limit": LIMIT,
    }

    if end_ms is not None:
        params["end"] = end_ms

    response = requests.get(
        BASE_URL + ENDPOINT,
        params=params,
        timeout=30,
    )

    response.raise_for_status()

    payload = response.json()

    if payload.get("retCode") != 0:
        raise RuntimeError(f"Bybit error for {symbol}: " f"{payload}")

    return payload.get("result", {}).get("list", [])


def download_symbol(
    symbol: str,
) -> pd.DataFrame:

    print()
    print("=" * 80)
    print(f"Downloading {symbol}")
    print("=" * 80)

    all_rows = []

    end_ms = None
    request_number = 0

    while True:

        rows = fetch_batch(
            symbol=symbol,
            end_ms=end_ms,
        )

        request_number += 1

        if not rows:
            break

        all_rows.extend(rows)

        timestamps = [int(row[0]) for row in rows]

        earliest_ms = min(timestamps)

        latest_ms = max(timestamps)

        earliest = pd.to_datetime(
            earliest_ms,
            unit="ms",
            utc=True,
        )

        latest = pd.to_datetime(
            latest_ms,
            unit="ms",
            utc=True,
        )

        print(
            f"Request {request_number:>2}: "
            f"{earliest.date()} "
            f"→ {latest.date()} "
            f"| candles={len(rows)}"
        )

        if len(rows) < LIMIT:
            break

        end_ms = earliest_ms - 1

        time.sleep(REQUEST_DELAY_SECONDS)

    if not all_rows:
        raise RuntimeError(f"No data returned for {symbol}")

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

    for column in numeric_columns:

        dataframe[column] = pd.to_numeric(
            dataframe[column],
            errors="raise",
        )

    dataframe = (
        dataframe.drop_duplicates(subset=["timestamp"])
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    return dataframe


def validate_data(
    symbol: str,
    dataframe: pd.DataFrame,
):

    if dataframe.empty:
        raise RuntimeError(f"{symbol}: empty dataframe")

    if dataframe["timestamp"].duplicated().any():

        raise RuntimeError(f"{symbol}: duplicate timestamps")

    if not dataframe["timestamp"].is_monotonic_increasing:

        raise RuntimeError(f"{symbol}: timestamps " "are not sorted")

    if (
        dataframe[
            [
                "open",
                "high",
                "low",
                "close",
            ]
        ]
        .isna()
        .any()
        .any()
    ):

        raise RuntimeError(f"{symbol}: OHLC contains NaN")

    invalid_high = dataframe["high"] < dataframe[
        [
            "open",
            "close",
            "low",
        ]
    ].max(axis=1)

    invalid_low = dataframe["low"] > dataframe[
        [
            "open",
            "close",
            "high",
        ]
    ].min(axis=1)

    if invalid_high.any():
        raise RuntimeError(f"{symbol}: invalid high values")

    if invalid_low.any():
        raise RuntimeError(f"{symbol}: invalid low values")


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("TSMOM REFERENCE DATA DOWNLOAD")

    summaries = []

    for symbol in SYMBOLS:

        dataframe = download_symbol(symbol)

        validate_data(
            symbol,
            dataframe,
        )

        output_file = OUTPUT_DIR / (f"{symbol}_" "spot_1d.csv")

        dataframe.to_csv(
            output_file,
            index=False,
        )

        earliest = dataframe.iloc[0]["timestamp"]

        latest = dataframe.iloc[-1]["timestamp"]

        summaries.append(
            {
                "symbol": symbol,
                "candles": len(dataframe),
                "earliest": earliest,
                "latest": latest,
                "file": str(output_file),
            }
        )

    print()
    print("=" * 100)

    print("DOWNLOAD COMPLETE")

    print("=" * 100)

    for result in summaries:

        print()
        print(result["symbol"])

        print(f"  Candles:  " f"{result['candles']}")

        print(f"  Earliest: " f"{result['earliest']}")

        print(f"  Latest:   " f"{result['latest']}")

        print(f"  File:     " f"{result['file']}")


if __name__ == "__main__":
    main()
