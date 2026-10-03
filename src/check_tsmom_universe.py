from datetime import datetime, timezone
from pathlib import Path
import time

import pandas as pd
import requests

BASE_URL = "https://api.bybit.com"

KLINE_ENDPOINT = "/v5/market/kline"
INSTRUMENT_ENDPOINT = "/v5/market/instruments-info"

CATEGORY = "spot"
INTERVAL = "D"
LIMIT = 1000

REQUEST_DELAY_SECONDS = 0.15


# ============================================================
# PRE-REGISTERED ACTIVITY-SCREENED UNIVERSE
#
# IMPORTANT:
# This list is frozen BEFORE seeing backtest results.
#
# The screening here concerns the project's primary/direct
# activity according to our strict first-pass filter.
# It is NOT a religious fatwa.
# ============================================================

CANDIDATES = {
    "BTCUSDT": ("Peer-to-peer digital money/payment network"),
    "ETHUSDT": (
        "General-purpose smart-contract " "and decentralized application platform"
    ),
    "SOLUSDT": (
        "General-purpose high-performance " "blockchain and application platform"
    ),
    "ADAUSDT": ("General-purpose blockchain platform"),
    "DOGEUSDT": ("Peer-to-peer digital currency"),
    "LTCUSDT": ("Decentralized peer-to-peer payment network"),
    "DOTUSDT": ("General-purpose interoperable blockchain platform"),
    "FILUSDT": ("Decentralized data storage network"),
    "ICPUSDT": ("Decentralized computing/cloud infrastructure"),
    "ARUSDT": ("Decentralized permanent data storage"),
    "ETCUSDT": ("General-purpose decentralized " "smart-contract platform"),
}


# ============================================================
# HISTORY REQUIREMENT
#
# 1 year  = signal warm-up
# 3 years = minimum actual evaluation period
#
# Therefore:
# total required history ≈ 4 years
# ============================================================

WARMUP_DAYS = 365

MIN_TEST_DAYS = 365 * 3

MIN_TOTAL_DAYS = WARMUP_DAYS + MIN_TEST_DAYS


OUTPUT_FILE = Path("data/reference_tsmom/" "universe_screen.csv")


def timestamp_to_datetime(
    timestamp_ms: int,
) -> datetime:

    return datetime.fromtimestamp(
        timestamp_ms / 1000,
        tz=timezone.utc,
    )


# ============================================================
# CHECK WHETHER BYBIT SPOT SYMBOL EXISTS
# ============================================================


def check_spot_symbol(
    symbol: str,
) -> bool:

    response = requests.get(
        BASE_URL + INSTRUMENT_ENDPOINT,
        params={
            "category": CATEGORY,
            "symbol": symbol,
        },
        timeout=30,
    )

    response.raise_for_status()

    payload = response.json()

    if payload.get("retCode") != 0:

        raise RuntimeError(f"Bybit error for " f"{symbol}: " f"{payload}")

    instruments = payload.get("result", {}).get("list", [])

    if not instruments:
        return False

    return True


# ============================================================
# FETCH ONE HISTORY BATCH
# ============================================================


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
        BASE_URL + KLINE_ENDPOINT,
        params=params,
        timeout=30,
    )

    response.raise_for_status()

    payload = response.json()

    if payload.get("retCode") != 0:

        raise RuntimeError(f"Bybit error for " f"{symbol}: " f"{payload}")

    return payload.get("result", {}).get("list", [])


# ============================================================
# DISCOVER FULL AVAILABLE HISTORY
# ============================================================


def find_history_range(
    symbol: str,
) -> dict:

    end_ms = None

    earliest_ms = None
    latest_ms = None

    total_candles = 0
    requests_count = 0

    while True:

        rows = fetch_klines(
            symbol=symbol,
            end_ms=end_ms,
        )

        requests_count += 1

        if not rows:
            break

        timestamps = [int(row[0]) for row in rows]

        batch_earliest = min(timestamps)

        batch_latest = max(timestamps)

        if latest_ms is None:

            latest_ms = batch_latest

        earliest_ms = batch_earliest

        total_candles += len(rows)

        earliest_date = timestamp_to_datetime(batch_earliest)

        latest_date = timestamp_to_datetime(batch_latest)

        print(
            f"    Request "
            f"{requests_count:>2}: "
            f"{earliest_date.date()} "
            f"→ "
            f"{latest_date.date()} "
            f"| candles="
            f"{len(rows)}"
        )

        if len(rows) < LIMIT:

            break

        end_ms = batch_earliest - 1

        time.sleep(REQUEST_DELAY_SECONDS)

    if earliest_ms is None or latest_ms is None:

        raise RuntimeError(f"No history for " f"{symbol}")

    earliest = timestamp_to_datetime(earliest_ms)

    latest = timestamp_to_datetime(latest_ms)

    total_days = (latest - earliest).days

    post_warmup_days = max(
        0,
        total_days - WARMUP_DAYS,
    )

    eligible = post_warmup_days >= MIN_TEST_DAYS

    return {
        "earliest": earliest,
        "latest": latest,
        "total_days": total_days,
        "post_warmup_days": (post_warmup_days),
        "candles": (total_candles),
        "requests": (requests_count),
        "eligible": (eligible),
    }


# ============================================================
# MAIN
# ============================================================


def main():

    print()
    print("=" * 100)

    print("TSMOM REFERENCE — " "PRE-REGISTERED UNIVERSE SCREEN")

    print("=" * 100)

    print()

    print(f"Warm-up requirement: " f"{WARMUP_DAYS} days")

    print(f"Minimum test period: " f"{MIN_TEST_DAYS} days")

    print(f"Minimum total history: " f"{MIN_TOTAL_DAYS} days")

    print()

    results = []

    for (
        symbol,
        screen_basis,
    ) in CANDIDATES.items():

        print()
        print("-" * 100)

        print(symbol)

        print(f"Activity screen: " f"{screen_basis}")

        try:

            exists = check_spot_symbol(symbol)

            if not exists:

                print("  Bybit Spot: " "NOT AVAILABLE")

                results.append(
                    {
                        "symbol": symbol,
                        "activity_screen": (screen_basis),
                        "bybit_spot": False,
                        "earliest": None,
                        "latest": None,
                        "total_days": 0,
                        "post_warmup_days": 0,
                        "post_warmup_years": 0.0,
                        "eligible": False,
                        "reason": ("Not available " "on Bybit Spot"),
                    }
                )

                continue

            print("  Bybit Spot: " "AVAILABLE")

            history = find_history_range(symbol)

            post_warmup_years = history["post_warmup_days"] / 365.25

            if history["eligible"]:

                reason = "PASS: sufficient " "history after " "12-month warm-up"

            else:

                reason = "FAIL: insufficient " "history after " "12-month warm-up"

            print(f"  Earliest: " f"{history['earliest']}")

            print(f"  Latest:   " f"{history['latest']}")

            print(f"  Total history: " f"{history['total_days']} " f"days")

            print(
                f"  Test history after "
                f"warm-up: "
                f"{history['post_warmup_days']} "
                f"days "
                f"({post_warmup_years:.2f} years)"
            )

            print(f"  ELIGIBLE: " f"{history['eligible']}")

            results.append(
                {
                    "symbol": symbol,
                    "activity_screen": (screen_basis),
                    "bybit_spot": True,
                    "earliest": (history["earliest"]),
                    "latest": (history["latest"]),
                    "total_days": (history["total_days"]),
                    "post_warmup_days": (history["post_warmup_days"]),
                    "post_warmup_years": (post_warmup_years),
                    "eligible": (history["eligible"]),
                    "reason": reason,
                }
            )

        except Exception as error:

            print(f"  ERROR: " f"{error}")

            results.append(
                {
                    "symbol": symbol,
                    "activity_screen": (screen_basis),
                    "bybit_spot": False,
                    "earliest": None,
                    "latest": None,
                    "total_days": 0,
                    "post_warmup_days": 0,
                    "post_warmup_years": 0.0,
                    "eligible": False,
                    "reason": (f"ERROR: {error}"),
                }
            )

        time.sleep(REQUEST_DELAY_SECONDS)

    # ========================================================
    # SAVE AUDIT TABLE
    # ========================================================

    dataframe = pd.DataFrame(results)

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print()
    print("=" * 100)

    print("FINAL UNIVERSE SCREEN")

    print("=" * 100)

    print()

    columns = [
        "symbol",
        "bybit_spot",
        "post_warmup_years",
        "eligible",
        "reason",
    ]

    print(dataframe[columns].to_string(index=False))

    eligible_symbols = dataframe[dataframe["eligible"]]["symbol"].tolist()

    print()
    print("ELIGIBLE FROZEN UNIVERSE:")

    if eligible_symbols:

        for symbol in eligible_symbols:

            print(f"  {symbol}")

    else:

        print("  NONE")

    print()

    print(f"Saved: " f"{OUTPUT_FILE}")


if __name__ == "__main__":
    main()
