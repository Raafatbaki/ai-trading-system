from pathlib import Path
import time

import pandas as pd
import requests

BASE_URL = "https://api.bybit.com"
ENDPOINT = "/v5/market/kline"

CATEGORY = "spot"
INTERVAL = "D"
LIMIT = 1000

REQUEST_DELAY_SECONDS = 0.15


# ============================================================
# PATHS
# ============================================================

UNIVERSE_FILE = Path("data/reference_tsmom/" "universe_screen.csv")

OUTPUT_DIR = Path("data/reference_tsmom")


# ============================================================
# LOAD FROZEN UNIVERSE
# ============================================================


def load_frozen_universe() -> list[str]:

    if not UNIVERSE_FILE.exists():

        raise FileNotFoundError(f"Universe file missing: " f"{UNIVERSE_FILE}")

    dataframe = pd.read_csv(UNIVERSE_FILE)

    required_columns = [
        "symbol",
        "eligible",
    ]

    missing = [column for column in required_columns if column not in dataframe.columns]

    if missing:

        raise RuntimeError(f"Universe file missing columns: " f"{missing}")

    eligible = dataframe[dataframe["eligible"] == True]["symbol"].tolist()

    if not eligible:

        raise RuntimeError("Frozen universe contains " "no eligible symbols.")

    return eligible


# ============================================================
# FETCH ONE BATCH
# ============================================================


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

        raise RuntimeError(f"Bybit error for " f"{symbol}: " f"{payload}")

    return payload.get("result", {}).get("list", [])


# ============================================================
# DOWNLOAD FULL HISTORY
# ============================================================


def download_symbol(
    symbol: str,
) -> pd.DataFrame:

    print()
    print("=" * 90)

    print(f"Downloading {symbol}")

    print("=" * 90)

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
            f"→ "
            f"{latest.date()} "
            f"| candles="
            f"{len(rows)}"
        )

        if len(rows) < LIMIT:
            break

        end_ms = earliest_ms - 1

        time.sleep(REQUEST_DELAY_SECONDS)

    if not all_rows:

        raise RuntimeError(f"No data returned " f"for {symbol}")

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


# ============================================================
# VALIDATION
# ============================================================


def validate_data(
    symbol: str,
    dataframe: pd.DataFrame,
):

    if dataframe.empty:

        raise RuntimeError(f"{symbol}: " "empty dataframe")

    if dataframe["timestamp"].duplicated().any():

        raise RuntimeError(f"{symbol}: " "duplicate timestamps")

    if not dataframe["timestamp"].is_monotonic_increasing:

        raise RuntimeError(f"{symbol}: " "timestamps not sorted")

    ohlc = [
        "open",
        "high",
        "low",
        "close",
    ]

    if dataframe[ohlc].isna().any().any():

        raise RuntimeError(f"{symbol}: " "OHLC contains NaN")

    if (dataframe[ohlc] <= 0).any().any():

        raise RuntimeError(f"{symbol}: " "OHLC contains " "non-positive prices")

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

        raise RuntimeError(f"{symbol}: " "invalid HIGH values")

    if invalid_low.any():

        raise RuntimeError(f"{symbol}: " "invalid LOW values")


# ============================================================
# MAIN
# ============================================================


def main():

    symbols = load_frozen_universe()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("=" * 100)

    print("TSMOM REFERENCE — " "FROZEN UNIVERSE DATA DOWNLOAD")

    print("=" * 100)

    print()

    print(f"Frozen universe size: " f"{len(symbols)}")

    print()

    for symbol in symbols:

        print(f"  {symbol}")

    summaries = []

    for symbol in symbols:

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

    summary = pd.DataFrame(summaries)

    summary_file = OUTPUT_DIR / "frozen_universe_data.csv"

    summary.to_csv(
        summary_file,
        index=False,
    )

    print()
    print()
    print("=" * 100)

    print("DOWNLOAD COMPLETE")

    print("=" * 100)

    print()

    print(summary.to_string(index=False))

    print()

    print(f"Summary saved: " f"{summary_file}")


if __name__ == "__main__":
    main()
