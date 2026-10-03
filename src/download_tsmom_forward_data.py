from __future__ import annotations

from datetime import (
    datetime,
    timedelta,
    timezone,
)
from pathlib import Path
import json
import time

import pandas as pd
import requests

# ============================================================
# PATHS
# ============================================================

REFERENCE_DIR = Path("data/reference_tsmom")

FREEZE_MANIFEST = REFERENCE_DIR / "freeze" / "tsmom_reference_candidate_v1.json"

FORWARD_DIR = Path("data/forward_tsmom")

OUTPUT_DIR = FORWARD_DIR / "daily"

SUMMARY_FILE = FORWARD_DIR / "forward_data_summary.csv"


# ============================================================
# BYBIT
# ============================================================

BASE_URL = "https://api.bybit.com"
ENDPOINT = "/v5/market/kline"

CATEGORY = "spot"
INTERVAL = "D"
LIMIT = 1000

REQUEST_DELAY_SECONDS = 0.15


# ============================================================
# LOAD FROZEN MANIFEST
# ============================================================


def load_manifest() -> dict:

    if not FREEZE_MANIFEST.exists():

        raise FileNotFoundError(f"Missing freeze manifest: " f"{FREEZE_MANIFEST}")

    with FREEZE_MANIFEST.open(
        "r",
        encoding="utf-8",
    ) as file:

        manifest = json.load(file)

    required = [
        "version",
        "research_data_end",
        "forward_test_start",
        "universe",
    ]

    missing = [key for key in required if key not in manifest]

    if missing:

        raise RuntimeError(f"Freeze manifest missing keys: " f"{missing}")

    return manifest


# ============================================================
# LAST FULLY CLOSED DAILY CANDLE
# ============================================================


def last_closed_daily_timestamp() -> pd.Timestamp:

    now_utc = datetime.now(timezone.utc)

    today_utc = datetime(
        year=now_utc.year,
        month=now_utc.month,
        day=now_utc.day,
        tzinfo=timezone.utc,
    )

    return pd.Timestamp(today_utc - timedelta(days=1))


# ============================================================
# FETCH LATEST BYBIT BATCH
# ============================================================


def fetch_latest_batch(
    symbol: str,
) -> list:

    response = requests.get(
        BASE_URL + ENDPOINT,
        params={
            "category": CATEGORY,
            "symbol": symbol,
            "interval": INTERVAL,
            "limit": LIMIT,
        },
        timeout=30,
    )

    response.raise_for_status()

    payload = response.json()

    if payload.get("retCode") != 0:

        raise RuntimeError(f"Bybit error for {symbol}: " f"{payload}")

    return payload.get("result", {}).get("list", [])


# ============================================================
# CONVERT BYBIT ROWS
# ============================================================


def convert_rows(
    rows: list,
) -> pd.DataFrame:

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

    for column in [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "turnover",
    ]:

        dataframe[column] = pd.to_numeric(
            dataframe[column],
            errors="raise",
        )

    return (
        dataframe.sort_values("timestamp")
        .drop_duplicates(subset=["timestamp"])
        .reset_index(drop=True)
    )


# ============================================================
# LOAD EXISTING FORWARD DATA
# ============================================================


def load_existing(
    file_path: Path,
) -> pd.DataFrame:

    if not file_path.exists():

        return pd.DataFrame()

    dataframe = pd.read_csv(
        file_path,
        parse_dates=["timestamp"],
    )

    return (
        dataframe.sort_values("timestamp")
        .drop_duplicates(subset=["timestamp"])
        .reset_index(drop=True)
    )


# ============================================================
# VALIDATION
# ============================================================


def validate_forward_data(
    symbol: str,
    dataframe: pd.DataFrame,
    forward_start: pd.Timestamp,
    last_closed: pd.Timestamp,
):

    if dataframe.empty:

        raise RuntimeError(f"{symbol}: no forward candles.")

    if (dataframe["timestamp"] < forward_start).any():

        raise RuntimeError(
            f"{symbol}: research-period data " "leaked into forward dataset."
        )

    if (dataframe["timestamp"] > last_closed).any():

        raise RuntimeError(f"{symbol}: incomplete current " "daily candle detected.")

    if dataframe["timestamp"].duplicated().any():

        raise RuntimeError(f"{symbol}: duplicate timestamps.")

    if not dataframe["timestamp"].is_monotonic_increasing:

        raise RuntimeError(f"{symbol}: timestamps not sorted.")

    for column in [
        "open",
        "high",
        "low",
        "close",
    ]:

        if (dataframe[column] <= 0).any():

            raise RuntimeError(f"{symbol}: invalid {column}.")


# ============================================================
# APPEND-ONLY UPDATE
# ============================================================


def update_symbol(
    symbol: str,
    forward_start: pd.Timestamp,
    last_closed: pd.Timestamp,
) -> tuple[pd.DataFrame, int]:

    output_file = OUTPUT_DIR / f"{symbol}_spot_1d_forward.csv"

    existing = load_existing(output_file)

    rows = fetch_latest_batch(symbol)

    latest = convert_rows(rows)

    latest = (
        latest[
            (latest["timestamp"] >= forward_start)
            & (latest["timestamp"] <= last_closed)
        ]
        .copy()
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # FIRST RUN
    # --------------------------------------------------------

    if existing.empty:

        combined = latest.copy()

        added = len(combined)

    # --------------------------------------------------------
    # SUBSEQUENT RUNS
    #
    # IMPORTANT:
    #
    # Existing recorded forward candles are NEVER replaced.
    #
    # We append only timestamps that did not previously exist.
    # --------------------------------------------------------

    else:

        existing_timestamps = set(existing["timestamp"])

        new_rows = latest[~latest["timestamp"].isin(existing_timestamps)].copy()

        added = len(new_rows)

        combined = pd.concat(
            [
                existing,
                new_rows,
            ],
            ignore_index=True,
        )

    combined = (
        combined.sort_values("timestamp")
        .drop_duplicates(
            subset=["timestamp"],
            keep="first",
        )
        .reset_index(drop=True)
    )

    validate_forward_data(
        symbol=symbol,
        dataframe=combined,
        forward_start=forward_start,
        last_closed=last_closed,
    )

    # --------------------------------------------------------
    # CONTINUITY CHECK
    #
    # Crypto daily data should contain one candle every day.
    # --------------------------------------------------------

    expected_dates = pd.date_range(
        start=forward_start,
        end=last_closed,
        freq="D",
        tz="UTC",
    )

    actual_dates = pd.DatetimeIndex(combined["timestamp"])

    missing_dates = expected_dates.difference(actual_dates)

    if len(missing_dates) > 0:

        raise RuntimeError(
            f"{symbol}: missing forward " f"daily candles: " f"{missing_dates.tolist()}"
        )

    combined.to_csv(
        output_file,
        index=False,
    )

    return (
        combined,
        added,
    )


# ============================================================
# MAIN
# ============================================================


def main():

    manifest = load_manifest()

    symbols = manifest["universe"]

    forward_start = pd.Timestamp(
        manifest["forward_test_start"],
        tz="UTC",
    )

    last_closed = last_closed_daily_timestamp()

    if last_closed < forward_start:

        raise RuntimeError("No fully closed forward " "daily candles exist yet.")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("=" * 110)

    print("TSMOM V1 — APPEND-ONLY " "FORWARD DATA UPDATE")

    print("=" * 110)

    print()

    print(f"Frozen version:      " f"{manifest['version']}")

    print(f"Research data end:   " f"{manifest['research_data_end']}")

    print(f"Forward test start:  " f"{forward_start}")

    print(f"Last closed candle:  " f"{last_closed}")

    print(f"Universe:            " f"{len(symbols)} assets")

    print()

    summaries = []

    total_added = 0

    for symbol in symbols:

        (
            dataframe,
            added,
        ) = update_symbol(
            symbol=symbol,
            forward_start=forward_start,
            last_closed=last_closed,
        )

        total_added += added

        output_file = OUTPUT_DIR / (f"{symbol}_" "spot_1d_forward.csv")

        summaries.append(
            {
                "symbol": symbol,
                "candles": (len(dataframe)),
                "new_candles": (added),
                "from": (dataframe.iloc[0]["timestamp"]),
                "to": (dataframe.iloc[-1]["timestamp"]),
                "file": str(output_file),
            }
        )

        print(
            f"{symbol:<10} "
            f"total={len(dataframe):>4} | "
            f"new={added:>2} | "
            f"{dataframe.iloc[0]['timestamp']} "
            f"→ "
            f"{dataframe.iloc[-1]['timestamp']}"
        )

        time.sleep(REQUEST_DELAY_SECONDS)

    summary = pd.DataFrame(summaries)

    summary.to_csv(
        SUMMARY_FILE,
        index=False,
    )

    print()
    print("=" * 110)

    print("FORWARD DATA UPDATE COMPLETE")

    print("=" * 110)

    print()

    print(f"Total new candles appended: " f"{total_added}")

    print()

    print(summary.to_string(index=False))

    print()

    print(f"Summary: " f"{SUMMARY_FILE}")


if __name__ == "__main__":
    main()
