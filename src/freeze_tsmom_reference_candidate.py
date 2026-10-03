from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone

PROJECT_ROOT = Path(".")

FREEZE_DIR = Path("data/reference_tsmom/freeze")

MANIFEST_FILE = FREEZE_DIR / "tsmom_reference_candidate_v1.json"


# ============================================================
# FROZEN RESEARCH CUTOFF
# ============================================================

RESEARCH_DATA_END = "2026-09-29"

FORWARD_TEST_START = "2026-09-30"


# ============================================================
# FROZEN STRATEGY
# ============================================================

STRATEGY = {
    "name": ("TSMOM Reference Risk-Scaled Portfolio"),
    "version": "v1",
    "momentum_lookback_months": 12,
    "decision_frequency": ("calendar_month_end"),
    "positive_momentum": "LONG",
    "zero_or_negative_momentum": "CASH",
    "volatility_model": "EWMA",
    "ewma_com_days": 60,
    "target_annual_volatility": 0.40,
    "annualization_days": 365,
    "asset_weight_cap": 1.0,
    "portfolio_weight_rule": ("raw_weight_i / " "max(1, sum(raw_weights))"),
    "short_selling": False,
    "leverage": False,
    "execution": ("next_daily_candle_open"),
    "fee_rate": 0.001,
    "initial_capital_usdt": 100.0,
}


# ============================================================
# FROZEN UNIVERSE
# ============================================================

UNIVERSE = [
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "ADAUSDT",
    "DOGEUSDT",
    "LTCUSDT",
    "DOTUSDT",
    "FILUSDT",
    "ICPUSDT",
    "ARUSDT",
    "ETCUSDT",
]


# ============================================================
# IMPORTANT IMPLEMENTATION FILES
# ============================================================

FILES_TO_HASH = [
    "src/backtest_portfolio.py",
    ("src/strategies/" "tsmom_reference_risk_scaled.py"),
    ("src/" "build_tsmom_reference_portfolio_weights.py"),
    ("src/" "run_tsmom_reference_portfolio.py"),
    ("data/reference_tsmom/" "universe_screen.csv"),
    ("data/reference_tsmom/" "portfolio/" "tsmom_reference_portfolio_weights.csv"),
    (
        "data/reference_tsmom/"
        "portfolio/results/"
        "tsmom_reference_portfolio_summary.csv"
    ),
]


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
# MAIN
# ============================================================


def main():

    FREEZE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    hashes = {}

    for relative_path in FILES_TO_HASH:

        file_path = Path(relative_path)

        if not file_path.exists():

            raise FileNotFoundError(f"Missing frozen file: " f"{file_path}")

        hashes[relative_path] = sha256_file(file_path)

    manifest = {
        "candidate": ("TSMOM Reference " "Risk-Scaled Portfolio"),
        "version": "v1",
        "frozen_at_utc": (datetime.now(timezone.utc).isoformat()),
        "research_data_end": (RESEARCH_DATA_END),
        "forward_test_start": (FORWARD_TEST_START),
        "strategy": (STRATEGY),
        "universe": (UNIVERSE),
        "historical_result": {
            "initial_capital_usdt": (100.0),
            "final_capital_usdt": (173.33),
            "return_pct": (73.33),
            "max_drawdown_pct": (-54.73),
            "average_exposure_pct": (61.63),
        },
        "robustness": {
            "positive_rolling_12m_pct": (59.12),
            "median_rolling_12m_pct": (18.17),
            "worst_rolling_12m_pct": (-50.69),
        },
        "file_sha256": (hashes),
    }

    with MANIFEST_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            manifest,
            file,
            indent=2,
        )

    print()
    print("=" * 100)

    print("TSMOM REFERENCE CANDIDATE FROZEN")

    print("=" * 100)

    print()

    print("Version:             v1")

    print(f"Research data end:   " f"{RESEARCH_DATA_END}")

    print(f"Forward test start:  " f"{FORWARD_TEST_START}")

    print(f"Universe:            " f"{len(UNIVERSE)} assets")

    print()

    print("IMPORTANT:")

    print(
        "Do not modify strategy rules, "
        "parameters or universe based on "
        "forward-test results."
    )

    print()

    print(f"Manifest: " f"{MANIFEST_FILE}")

    print()

    print("SHA256 fingerprints:")

    for (
        path,
        digest,
    ) in hashes.items():

        print(f"{path}")

        print(f"  {digest}")


if __name__ == "__main__":
    main()
