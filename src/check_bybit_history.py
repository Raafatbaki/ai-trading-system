from datetime import datetime, timezone
from pathlib import Path
import time

import requests

BASE_URL = "https://api.bybit.com"
ENDPOINT = "/v5/market/kline"

SYMBOLS = [
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
]

CATEGORY = "spot"

# Daily candles are enough for discovering
# how far back Bybit's history extends.
INTERVAL = "D"

LIMIT = 1000

REQUEST_DELAY_SECONDS = 0.15


def timestamp_to_datetime(
    timestamp_ms: int,
) -> datetime:

    return datetime.fromtimestamp(
        timestamp_ms / 1000,
        tz=timezone.utc,
    )


def fetch_klines(
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


def find_history_range(
    symbol: str,
) -> dict:

    print()
    print("=" * 80)
    print(symbol)
    print("=" * 80)

    end_ms = None

    earliest_ms = None
    latest_ms = None

    total_candles = 0
    request_count = 0

    while True:

        rows = fetch_klines(
            symbol=symbol,
            end_ms=end_ms,
        )

        request_count += 1

        if not rows:
            break

        timestamps = [int(row[0]) for row in rows]

        batch_latest = max(timestamps)

        batch_earliest = min(timestamps)

        if latest_ms is None:
            latest_ms = batch_latest

        earliest_ms = batch_earliest

        total_candles += len(rows)

        print(
            f"Request {request_count:>2}: "
            f"{timestamp_to_datetime(batch_earliest).date()} "
            f"→ "
            f"{timestamp_to_datetime(batch_latest).date()} "
            f"| candles={len(rows)}"
        )

        # Fewer than LIMIT means we have reached
        # the beginning of available history.
        if len(rows) < LIMIT:
            break

        # Move one millisecond before the
        # earliest candle we just received.
        end_ms = batch_earliest - 1

        time.sleep(REQUEST_DELAY_SECONDS)

    if earliest_ms is None or latest_ms is None:

        raise RuntimeError(f"No history returned for {symbol}")

    earliest = timestamp_to_datetime(earliest_ms)

    latest = timestamp_to_datetime(latest_ms)

    history_days = (latest - earliest).days

    history_years = history_days / 365.25

    return {
        "symbol": symbol,
        "earliest": earliest,
        "latest": latest,
        "days": history_days,
        "years": history_years,
        "candles": total_candles,
        "requests": request_count,
    }


def main():

    results = []

    for symbol in SYMBOLS:

        result = find_history_range(symbol)

        results.append(result)

    print()
    print()
    print("=" * 100)
    print("BYBIT SPOT HISTORY RANGE")
    print("=" * 100)

    for result in results:

        print()

        print(f"{result['symbol']}")

        print(f"  Earliest: " f"{result['earliest']}")

        print(f"  Latest:   " f"{result['latest']}")

        print(
            f"  History:  " f"{result['years']:.2f} years " f"({result['days']} days)"
        )

        print(f"  Daily candles retrieved: " f"{result['candles']}")

        print(f"  API requests: " f"{result['requests']}")

    print()
    print("=" * 100)

    print()
    print(
        "TSMOM Reference requires approximately "
        "12 months of history before its first signal."
    )


if __name__ == "__main__":
    main()
