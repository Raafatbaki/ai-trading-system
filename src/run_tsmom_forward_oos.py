from __future__ import annotations

from pathlib import Path
import hashlib
import json

import pandas as pd

from backtest_portfolio import (
    run_portfolio_backtest,
)

from strategies.tsmom_reference_risk_scaled import (
    build_monthly_decisions,
)

# ============================================================
# PATHS
# ============================================================

REFERENCE_DIR = Path("data/reference_tsmom")

FORWARD_DIR = Path("data/forward_tsmom")

MANIFEST_FILE = REFERENCE_DIR / "freeze" / "tsmom_reference_candidate_v1.json"

FORWARD_DAILY_DIR = FORWARD_DIR / "daily"

OUTPUT_DIR = FORWARD_DIR / "results"

WEIGHTS_FILE = FORWARD_DIR / "forward_portfolio_weights.csv"

DECISION_AUDIT_FILE = FORWARD_DIR / "forward_decision_audit.csv"

TRANSACTIONS_FILE = OUTPUT_DIR / "forward_transactions.csv"

EQUITY_FILE = OUTPUT_DIR / "forward_equity.csv"

SUMMARY_FILE = OUTPUT_DIR / "forward_summary.csv"


# ============================================================
# FORWARD CAPITAL
# ============================================================

INITIAL_CAPITAL = 100.0
FEE_RATE = 0.001

EPSILON = 1e-12


# ============================================================
# HASH
# ============================================================


def sha256_file(
    file_path: Path,
) -> str:

    hasher = hashlib.sha256()

    with file_path.open("rb") as file:

        while True:

            chunk = file.read(1024 * 1024)

            if not chunk:
                break

            hasher.update(chunk)

    return hasher.hexdigest()


# ============================================================
# LOAD MANIFEST
# ============================================================


def load_manifest() -> dict:

    if not MANIFEST_FILE.exists():

        raise FileNotFoundError(f"Missing manifest: " f"{MANIFEST_FILE}")

    with MANIFEST_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:

        manifest = json.load(file)

    required = [
        "version",
        "research_data_end",
        "forward_test_start",
        "universe",
        "file_sha256",
    ]

    missing = [key for key in required if key not in manifest]

    if missing:

        raise RuntimeError(f"Manifest missing keys: " f"{missing}")

    return manifest


# ============================================================
# VERIFY FROZEN CANDIDATE
# ============================================================


def verify_frozen_candidate(
    manifest: dict,
):

    print()
    print("VERIFYING FROZEN V1")

    print("-" * 110)

    mismatches = []

    for (
        relative_path,
        expected_hash,
    ) in manifest["file_sha256"].items():

        file_path = Path(relative_path)

        if not file_path.exists():

            mismatches.append(
                (
                    relative_path,
                    "FILE MISSING",
                    expected_hash,
                )
            )

            continue

        current_hash = sha256_file(file_path)

        if current_hash != expected_hash:

            mismatches.append(
                (
                    relative_path,
                    current_hash,
                    expected_hash,
                )
            )

        else:

            print(f"OK  {relative_path}")

    if mismatches:

        print()

        print("FROZEN CANDIDATE " "VERIFICATION FAILED")

        for (
            path,
            current,
            expected,
        ) in mismatches:

            print()

            print(path)

            print(f"  expected: " f"{expected}")

            print(f"  current:  " f"{current}")

        raise RuntimeError("Frozen TSMOM v1 files " "changed after freeze.")

    print()

    print("Frozen candidate verified.")


# ============================================================
# LOAD DAILY FILE
# ============================================================


def load_daily_file(
    file_path: Path,
) -> pd.DataFrame:

    if not file_path.exists():

        raise FileNotFoundError(f"Missing data file: " f"{file_path}")

    dataframe = pd.read_csv(
        file_path,
        parse_dates=["timestamp"],
    )

    dataframe = (
        dataframe.sort_values("timestamp")
        .drop_duplicates(subset=["timestamp"])
        .reset_index(drop=True)
    )

    return dataframe


# ============================================================
# LOAD RESEARCH HISTORY
# ============================================================


def load_reference_data(
    symbol: str,
) -> pd.DataFrame:

    file_path = REFERENCE_DIR / f"{symbol}_spot_1d.csv"

    return load_daily_file(file_path)


# ============================================================
# LOAD FORWARD DATA
# ============================================================


def load_forward_data(
    symbol: str,
) -> pd.DataFrame:

    file_path = FORWARD_DAILY_DIR / (f"{symbol}_" "spot_1d_forward.csv")

    return load_daily_file(file_path)


# ============================================================
# COMBINE HISTORY FOR INDICATORS
# ============================================================


def build_indicator_history(
    symbol: str,
    manifest: dict,
) -> pd.DataFrame:

    reference = load_reference_data(symbol)

    forward = load_forward_data(symbol)

    research_end = pd.Timestamp(
        manifest["research_data_end"],
        tz="UTC",
    )

    forward_start = pd.Timestamp(
        manifest["forward_test_start"],
        tz="UTC",
    )

    # --------------------------------------------------------
    # STRICT RESEARCH / FORWARD SEPARATION
    # --------------------------------------------------------

    if reference["timestamp"].max() > research_end:

        raise RuntimeError(
            f"{symbol}: reference data " "contains post-freeze observations."
        )

    if forward["timestamp"].min() < forward_start:

        raise RuntimeError(
            f"{symbol}: forward data contains " "research-period observations."
        )

    overlap = set(reference["timestamp"]).intersection(set(forward["timestamp"]))

    if overlap:

        raise RuntimeError(
            f"{symbol}: reference/forward " "timestamp overlap detected."
        )

    combined = pd.concat(
        [
            reference,
            forward,
        ],
        ignore_index=True,
    )

    combined = combined.sort_values("timestamp").reset_index(drop=True)

    return combined


# ============================================================
# BUILD FORWARD DECISIONS
# ============================================================


def build_forward_decisions(
    symbols: list[str],
    manifest: dict,
):

    forward_start = pd.Timestamp(
        manifest["forward_test_start"],
        tz="UTC",
    )

    audit_rows = []

    expected_dates = None

    for symbol in symbols:

        full_history = build_indicator_history(
            symbol,
            manifest,
        )

        monthly = build_monthly_decisions(full_history)

        forward_monthly = (
            monthly[monthly["timestamp"] >= forward_start].copy().reset_index(drop=True)
        )

        if forward_monthly.empty:

            raise RuntimeError(f"{symbol}: no completed " "forward month-end decision.")

        if forward_monthly["target_weight"].isna().any():

            raise RuntimeError(f"{symbol}: invalid forward " "target weight.")

        dates = forward_monthly["timestamp"].tolist()

        if expected_dates is None:

            expected_dates = dates

        elif dates != expected_dates:

            raise RuntimeError(
                f"{symbol}: forward month-end "
                "dates differ from the "
                "rest of the universe."
            )

        for _, row in forward_monthly.iterrows():

            audit_rows.append(
                {
                    "timestamp": (row["timestamp"]),
                    "symbol": symbol,
                    "close": (row["close"]),
                    "close_12m_ago": (row["close_12m_ago"]),
                    "momentum_12m_pct": (row["momentum_12m_pct"]),
                    "ewma_vol_annual_pct": (row["ewma_vol_annual"] * 100.0),
                    "raw_asset_weight": (row["target_weight"]),
                }
            )

    audit = pd.DataFrame(audit_rows)

    if audit.empty:

        raise RuntimeError("No forward decisions generated.")

    # ========================================================
    # CROSS-ASSET NORMALIZATION
    #
    # Frozen rule:
    #
    # normalized_i =
    # raw_i / max(1, sum(raw))
    # ========================================================

    raw_matrix = (
        audit.pivot(
            index="timestamp",
            columns="symbol",
            values="raw_asset_weight",
        )
        .reindex(columns=symbols)
        .sort_index()
    )

    if raw_matrix.isna().any().any():

        raise RuntimeError("Missing asset weight in " "forward decision matrix.")

    raw_total = raw_matrix.sum(axis=1)

    normalization_factor = raw_total.clip(lower=1.0)

    normalized = raw_matrix.div(
        normalization_factor,
        axis=0,
    )

    portfolio_exposure = normalized.sum(axis=1)

    if (portfolio_exposure > 1.0 + EPSILON).any():

        raise RuntimeError("Forward leverage detected.")

    # ========================================================
    # ADD NORMALIZED WEIGHTS TO AUDIT
    # ========================================================

    normalized_long = normalized.stack().rename("portfolio_weight").reset_index()

    audit = audit.merge(
        normalized_long,
        on=[
            "timestamp",
            "symbol",
        ],
        how="left",
    )

    # ========================================================
    # BACKTEST WEIGHT TABLE
    # ========================================================

    weights = pd.DataFrame({"timestamp": (normalized.index)})

    for symbol in symbols:

        weights[f"{symbol}_weight"] = normalized[symbol].values

    weights["raw_total_weight"] = raw_total.values

    weights["portfolio_exposure"] = portfolio_exposure.values

    weights["cash_weight"] = 1.0 - portfolio_exposure.values

    weights["active_assets"] = (normalized > EPSILON).sum(axis=1).values

    return (
        audit,
        weights,
    )


# ============================================================
# FORWARD PRICE DATA ONLY
# ============================================================


def load_forward_prices(
    symbols: list[str],
) -> dict[str, pd.DataFrame]:

    return {symbol: (load_forward_data(symbol)) for symbol in symbols}


# ============================================================
# SAVE RESULTS
# ============================================================


def save_results(
    audit: pd.DataFrame,
    weights: pd.DataFrame,
    transactions: pd.DataFrame,
    equity: pd.DataFrame,
    stats: dict,
    manifest: dict,
):

    FORWARD_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    audit.to_csv(
        DECISION_AUDIT_FILE,
        index=False,
    )

    weights.to_csv(
        WEIGHTS_FILE,
        index=False,
    )

    transactions.to_csv(
        TRANSACTIONS_FILE,
        index=False,
    )

    equity.to_csv(
        EQUITY_FILE,
        index=False,
    )

    summary = pd.DataFrame(
        [
            {
                "candidate_version": (manifest["version"]),
                "research_data_end": (manifest["research_data_end"]),
                "forward_test_start": (manifest["forward_test_start"]),
                "forward_data_end": (equity.iloc[-1]["timestamp"]),
                "initial_capital": (stats["initial_capital"]),
                "current_equity": (stats["final_capital"]),
                "forward_return_pct": (stats["total_return"]),
                "max_drawdown_pct": (stats["max_drawdown"]),
                "average_exposure_pct": (stats["average_exposure_pct"]),
                "invested_days_pct": (stats["invested_days_pct"]),
                "rebalance_events": (stats["rebalance_events"]),
                "transactions": (stats["transaction_count"]),
                "fees_usdt": (stats["total_fees"]),
                "forced_exits": (stats["forced_exits"]),
            }
        ]
    )

    summary.to_csv(
        SUMMARY_FILE,
        index=False,
    )


# ============================================================
# PRINT DECISION AUDIT
# ============================================================


def print_latest_decision(
    audit: pd.DataFrame,
    weights: pd.DataFrame,
):

    latest_date = audit["timestamp"].max()

    latest = audit[audit["timestamp"] == latest_date].copy()

    latest = latest.sort_values(
        "portfolio_weight",
        ascending=False,
    )

    weight_row = weights[weights["timestamp"] == latest_date].iloc[0]

    print()
    print("LATEST FORWARD DECISION")

    print("-" * 110)

    print(f"Decision timestamp:   " f"{latest_date}")

    print(f"Raw total weight:     " f"{weight_row['raw_total_weight'] * 100:.2f}%")

    print(f"Portfolio exposure:   " f"{weight_row['portfolio_exposure'] * 100:.2f}%")

    print(f"Cash weight:          " f"{weight_row['cash_weight'] * 100:.2f}%")

    print(f"Active assets:        " f"{int(weight_row['active_assets'])}")

    print()

    print(
        f"{'Asset':<10}"
        f"{'12M Mom':>12}"
        f"{'Vol':>12}"
        f"{'Raw W':>12}"
        f"{'Portfolio W':>15}"
    )

    print("-" * 65)

    for _, row in latest.iterrows():

        print(
            f"{row['symbol']:<10}"
            f"{row['momentum_12m_pct']:>11.2f}%"
            f"{row['ewma_vol_annual_pct']:>11.2f}%"
            f"{row['raw_asset_weight'] * 100:>11.2f}%"
            f"{row['portfolio_weight'] * 100:>14.2f}%"
        )


# ============================================================
# PRINT FORWARD RESULT
# ============================================================


def print_result(
    transactions: pd.DataFrame,
    equity: pd.DataFrame,
    stats: dict,
    manifest: dict,
):

    print()
    print("=" * 110)

    print("TSMOM V1 — FORWARD OUT-OF-SAMPLE")

    print("=" * 110)

    print()

    print(f"Frozen version:      " f"{manifest['version']}")

    print(f"Research data end:   " f"{manifest['research_data_end']}")

    print(f"Forward test start:  " f"{manifest['forward_test_start']}")

    print(f"Forward data end:    " f"{equity.iloc[-1]['timestamp']}")

    print()

    print("PERFORMANCE TO DATE")

    print("-" * 110)

    print(f"Initial capital:     " f"{stats['initial_capital']:.4f} USDT")

    print(f"Current equity:      " f"{stats['final_capital']:.4f} USDT")

    print(f"Forward return:      " f"{stats['total_return']:+.4f}%")

    print(f"Maximum drawdown:    " f"{stats['max_drawdown']:.4f}%")

    print(f"Average exposure:    " f"{stats['average_exposure_pct']:.2f}%")

    print(f"Fees:                " f"{stats['total_fees']:.4f} USDT")

    print(f"Forced liquidation:  " f"{stats['forced_exits']}")

    print()

    print("EXECUTION AUDIT")

    print("-" * 110)

    if transactions.empty:

        print("No forward transactions yet.")

    else:

        normal = transactions[transactions["forced"] == False]

        if not normal.empty:

            first = normal.iloc[0]

            print(f"First signal:        " f"{first['signal_timestamp']}")

            print(f"First execution:     " f"{first['execution_timestamp']}")

            print(f"First asset:         " f"{first['symbol']}")

            print(f"First side:          " f"{first['side']}")

            print()

            print(f"Transaction rows:    " f"{len(normal)}")

    print()

    print("IMPORTANT")

    print("-" * 110)

    print("This is an ongoing forward test.")

    print("Open positions are NOT forcibly " "liquidated at the current data end.")

    print("No strategy parameter may be changed " "based on these results.")

    print()

    print("OUTPUT")

    print("-" * 110)

    print(f"Decision audit: " f"{DECISION_AUDIT_FILE}")

    print(f"Weights:        " f"{WEIGHTS_FILE}")

    print(f"Transactions:   " f"{TRANSACTIONS_FILE}")

    print(f"Equity:         " f"{EQUITY_FILE}")

    print(f"Summary:        " f"{SUMMARY_FILE}")


# ============================================================
# MAIN
# ============================================================


def main():

    manifest = load_manifest()

    print()
    print("=" * 110)

    print("TSMOM V1 — " "FORWARD OOS RUN")

    print("=" * 110)

    verify_frozen_candidate(manifest)

    symbols = manifest["universe"]

    (
        audit,
        weights,
    ) = build_forward_decisions(
        symbols,
        manifest,
    )

    print_latest_decision(
        audit,
        weights,
    )

    forward_prices = load_forward_prices(symbols)

    (
        transactions,
        equity,
        stats,
    ) = run_portfolio_backtest(
        price_data=forward_prices,
        weights=weights,
        initial_capital=(INITIAL_CAPITAL),
        fee_rate=(FEE_RATE),
        # IMPORTANT:
        #
        # Forward test is still running.
        # We must NOT create an artificial
        # final sell at 2026-10-02.
        force_liquidation=False,
    )

    save_results(
        audit=audit,
        weights=weights,
        transactions=transactions,
        equity=equity,
        stats=stats,
        manifest=manifest,
    )

    print_result(
        transactions,
        equity,
        stats,
        manifest,
    )


if __name__ == "__main__":
    main()
