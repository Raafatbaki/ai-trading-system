from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pandas as pd

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(".")

FORWARD_DIR = Path("data/forward_tsmom")

SUMMARY_FILE = FORWARD_DIR / "results" / "forward_summary.csv"

WEIGHTS_FILE = FORWARD_DIR / "forward_portfolio_weights.csv"

TRACKING_DIR = FORWARD_DIR / "tracking"

SNAPSHOTS_FILE = TRACKING_DIR / "forward_snapshots.csv"


# ============================================================
# RUN EXISTING SCRIPT
# ============================================================


def run_script(
    script_path: str,
):

    print()
    print("=" * 100)

    print(f"RUNNING {script_path}")

    print("=" * 100)

    subprocess.run(
        [
            sys.executable,
            script_path,
        ],
        check=True,
    )


# ============================================================
# LOAD CURRENT RESULT
# ============================================================


def load_current_snapshot() -> dict:

    if not SUMMARY_FILE.exists():

        raise FileNotFoundError(f"Missing forward summary: " f"{SUMMARY_FILE}")

    if not WEIGHTS_FILE.exists():

        raise FileNotFoundError(f"Missing forward weights: " f"{WEIGHTS_FILE}")

    summary = pd.read_csv(SUMMARY_FILE)

    weights = pd.read_csv(
        WEIGHTS_FILE,
        parse_dates=["timestamp"],
    )

    if summary.empty:

        raise RuntimeError("Forward summary is empty.")

    if weights.empty:

        raise RuntimeError("Forward weights are empty.")

    current = summary.iloc[0]

    latest_weight = weights.sort_values("timestamp").iloc[-1]

    return {
        "forward_data_end": (current["forward_data_end"]),
        "candidate_version": (current["candidate_version"]),
        "latest_decision": (latest_weight["timestamp"]),
        "current_equity": float(current["current_equity"]),
        "forward_return_pct": float(current["forward_return_pct"]),
        "max_drawdown_pct": float(current["max_drawdown_pct"]),
        "average_exposure_pct": float(current["average_exposure_pct"]),
        "current_target_exposure_pct": (
            float(latest_weight["portfolio_exposure"]) * 100.0
        ),
        "current_cash_target_pct": (float(latest_weight["cash_weight"]) * 100.0),
        "active_assets": int(latest_weight["active_assets"]),
        "transactions": int(current["transactions"]),
        "fees_usdt": float(current["fees_usdt"]),
    }


# ============================================================
# SAVE HISTORICAL SNAPSHOT
# ============================================================


def save_snapshot(
    snapshot: dict,
):

    TRACKING_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    new_row = pd.DataFrame([snapshot])

    if SNAPSHOTS_FILE.exists():

        existing = pd.read_csv(SNAPSHOTS_FILE)

        # ----------------------------------------------------
        # If we run the tracker several times with the same
        # latest closed candle, replace that snapshot rather
        # than duplicating it.
        # ----------------------------------------------------

        existing = existing[
            existing["forward_data_end"].astype(str)
            != str(snapshot["forward_data_end"])
        ]

        combined = pd.concat(
            [
                existing,
                new_row,
            ],
            ignore_index=True,
        )

    else:

        combined = new_row

    combined = combined.sort_values("forward_data_end").reset_index(drop=True)

    combined.to_csv(
        SNAPSHOTS_FILE,
        index=False,
    )


# ============================================================
# PRINT CURRENT STATUS
# ============================================================


def print_status(
    snapshot: dict,
):

    print()
    print("=" * 100)

    print("TSMOM V1 — FORWARD TRACKER STATUS")

    print("=" * 100)

    print()

    print(f"Version:                 " f"{snapshot['candidate_version']}")

    print(f"Forward data through:    " f"{snapshot['forward_data_end']}")

    print(f"Latest monthly decision: " f"{snapshot['latest_decision']}")

    print()

    print(f"Current equity:          " f"{snapshot['current_equity']:.4f} USDT")

    print(f"Forward return:          " f"{snapshot['forward_return_pct']:+.4f}%")

    print(f"Maximum drawdown:        " f"{snapshot['max_drawdown_pct']:.4f}%")

    print()

    print(
        f"Current target exposure: " f"{snapshot['current_target_exposure_pct']:.2f}%"
    )

    print(f"Current cash target:     " f"{snapshot['current_cash_target_pct']:.2f}%")

    print(f"Active assets:           " f"{snapshot['active_assets']}")

    print()

    print(f"Transactions:            " f"{snapshot['transactions']}")

    print(f"Fees:                    " f"{snapshot['fees_usdt']:.4f} USDT")

    print()

    print(f"Tracking history: " f"{SNAPSHOTS_FILE}")


# ============================================================
# MAIN
# ============================================================


def main():

    # 1. Download only fully closed new daily candles.
    run_script("src/download_tsmom_forward_data.py")

    # 2. Verify freeze and rerun the OOS portfolio.
    run_script("src/run_tsmom_forward_oos.py")

    # 3. Store current state in persistent tracking history.
    snapshot = load_current_snapshot()

    save_snapshot(snapshot)

    print_status(snapshot)


if __name__ == "__main__":
    main()
