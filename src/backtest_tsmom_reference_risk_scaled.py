from pathlib import Path

import pandas as pd

from backtest import calculate_buy_and_hold
from backtest_fractional import run_fractional_backtest

# ============================================================
# PATHS
# ============================================================

DATA_DIR = Path("data/reference_tsmom")

UNIVERSE_FILE = DATA_DIR / "universe_screen.csv"

SIGNALS_DIR = DATA_DIR / "risk_scaled_signals"

RESULTS_DIR = DATA_DIR / "risk_scaled_results"

BASELINE_FILE = DATA_DIR / "results" / "tsmom_reference_frozen_universe_summary.csv"

SUMMARY_FILE = RESULTS_DIR / "tsmom_reference_risk_scaled_summary.csv"


# ============================================================
# LOAD FROZEN UNIVERSE
# ============================================================


def load_frozen_universe() -> list[str]:

    if not UNIVERSE_FILE.exists():

        raise FileNotFoundError(f"Missing universe file: " f"{UNIVERSE_FILE}")

    dataframe = pd.read_csv(UNIVERSE_FILE)

    eligible_mask = dataframe["eligible"].astype(str).str.strip().str.lower().eq("true")

    symbols = (
        dataframe.loc[
            eligible_mask,
            "symbol",
        ]
        .astype(str)
        .tolist()
    )

    if not symbols:

        raise RuntimeError("No eligible symbols " "in frozen universe.")

    return symbols


# ============================================================
# LOAD ALL-IN BASELINE
# ============================================================


def load_baseline() -> pd.DataFrame:

    if not BASELINE_FILE.exists():

        raise FileNotFoundError(f"Missing baseline file: " f"{BASELINE_FILE}")

    dataframe = pd.read_csv(BASELINE_FILE)

    required = [
        "symbol",
        "return_pct",
        "max_drawdown_pct",
        "exposure_pct",
    ]

    missing = [column for column in required if column not in dataframe.columns]

    if missing:

        raise RuntimeError("Baseline file missing columns: " f"{missing}")

    return dataframe.set_index("symbol")


# ============================================================
# LOAD RISK-SCALED SIGNALS
# ============================================================


def load_signals(
    symbol: str,
) -> pd.DataFrame:

    file_path = SIGNALS_DIR / (
        f"{symbol}_" "spot_1d_" "tsmom_reference_" "risk_scaled_signals.csv"
    )

    if not file_path.exists():

        raise FileNotFoundError(f"Missing signal file: " f"{file_path}")

    dataframe = pd.read_csv(
        file_path,
        parse_dates=["timestamp"],
    )

    dataframe = dataframe.sort_values("timestamp").reset_index(drop=True)

    return dataframe


# ============================================================
# REMOVE 12-MONTH WARM-UP
# ============================================================


def trim_to_valid_period(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:

    valid = dataframe["target_weight"].notna()

    if not valid.any():

        raise RuntimeError("No valid target weights.")

    first_valid_index = valid.idxmax()

    return dataframe.loc[first_valid_index:].copy().reset_index(drop=True)


# ============================================================
# SAVE RESULT FILES
# ============================================================


def save_results(
    symbol: str,
    transactions: pd.DataFrame,
    equity: pd.DataFrame,
):

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    transactions_file = RESULTS_DIR / (f"{symbol}_" "risk_scaled_" "transactions.csv")

    equity_file = RESULTS_DIR / (f"{symbol}_" "risk_scaled_" "equity.csv")

    transactions.to_csv(
        transactions_file,
        index=False,
    )

    equity.to_csv(
        equity_file,
        index=False,
    )

    return (
        transactions_file,
        equity_file,
    )


# ============================================================
# RUN ONE SYMBOL
# ============================================================


def run_symbol(
    symbol: str,
    baseline: pd.DataFrame,
) -> dict:

    full_data = load_signals(symbol)

    data = trim_to_valid_period(full_data)

    (
        transactions,
        equity,
        stats,
    ) = run_fractional_backtest(data)

    buy_hold = calculate_buy_and_hold(data)

    (
        transactions_file,
        equity_file,
    ) = save_results(
        symbol,
        transactions,
        equity,
    )

    if symbol not in baseline.index:

        raise RuntimeError(f"{symbol} missing from " "all-in baseline.")

    baseline_row = baseline.loc[symbol]

    baseline_return = float(baseline_row["return_pct"])

    baseline_dd = float(baseline_row["max_drawdown_pct"])

    baseline_exposure = float(baseline_row["exposure_pct"])

    risk_return = float(stats["total_return"])

    risk_dd = float(stats["max_drawdown"])

    risk_exposure = float(stats["average_exposure_pct"])

    return_delta = risk_return - baseline_return

    # Example:
    #
    # Risk DD     = -35%
    # Baseline DD = -70%
    #
    # Improvement = +35 percentage points

    dd_improvement = risk_dd - baseline_dd

    monthly_decisions = int(data["target_weight"].notna().sum())

    print()
    print("=" * 110)

    print(symbol)

    print("=" * 110)

    print(
        f"Test period: "
        f"{data.iloc[0]['timestamp']} "
        f"→ "
        f"{data.iloc[-1]['timestamp']}"
    )

    print(f"Candles:            " f"{len(data)}")

    print(f"Monthly decisions:  " f"{monthly_decisions}")

    print(f"Rebalances:         " f"{stats['rebalance_count']}")

    print()

    print("RETURN")

    print("-" * 110)

    print(f"Risk-scaled:        " f"{risk_return:.2f}%")

    print(f"All-in baseline:    " f"{baseline_return:.2f}%")

    print(f"Difference:         " f"{return_delta:+.2f} pp")

    print(f"Buy & Hold:         " f"{buy_hold:.2f}%")

    print()

    print("RISK")

    print("-" * 110)

    print(f"Risk-scaled DD:     " f"{risk_dd:.2f}%")

    print(f"All-in DD:          " f"{baseline_dd:.2f}%")

    print(f"DD improvement:     " f"{dd_improvement:+.2f} pp")

    print()

    print("EXPOSURE")

    print("-" * 110)

    print(f"Average weight:     " f"{risk_exposure:.2f}%")

    print(f"All-in exposure:    " f"{baseline_exposure:.2f}%")

    print(f"Invested candles:   " f"{stats['invested_candles_pct']:.2f}%")

    print()

    print("COSTS")

    print("-" * 110)

    print(f"Fees:               " f"{stats['total_fees']:.4f} USDT")

    print(f"Total notional:     " f"{stats['total_notional']:.2f} USDT")

    print(f"Forced exits:       " f"{stats['forced_exits']}")

    print()

    print(f"Transactions: " f"{transactions_file}")

    print(f"Equity:       " f"{equity_file}")

    return {
        "symbol": symbol,
        "monthly_decisions": (monthly_decisions),
        "rebalances": (stats["rebalance_count"]),
        "risk_scaled_return_pct": (risk_return),
        "all_in_return_pct": (baseline_return),
        "return_difference_pp": (return_delta),
        "buy_hold_pct": (buy_hold),
        "risk_scaled_dd_pct": (risk_dd),
        "all_in_dd_pct": (baseline_dd),
        "dd_improvement_pp": (dd_improvement),
        "avg_weight_pct": (risk_exposure),
        "all_in_exposure_pct": (baseline_exposure),
        "invested_candles_pct": (stats["invested_candles_pct"]),
        "fees_usdt": (stats["total_fees"]),
        "total_notional_usdt": (stats["total_notional"]),
        "forced_exits": (stats["forced_exits"]),
    }


# ============================================================
# AGGREGATE SUMMARY
# ============================================================


def print_summary(
    summary: pd.DataFrame,
):

    print()
    print()
    print("=" * 110)

    print("TSMOM REFERENCE — " "RISK-SCALED RESULTS")

    print("=" * 110)

    print()

    display_columns = [
        "symbol",
        "risk_scaled_return_pct",
        "all_in_return_pct",
        "risk_scaled_dd_pct",
        "all_in_dd_pct",
        "avg_weight_pct",
        "rebalances",
    ]

    print(summary[display_columns].to_string(index=False))

    profitable = int((summary["risk_scaled_return_pct"] > 0).sum())

    losing = int((summary["risk_scaled_return_pct"] < 0).sum())

    higher_return_than_all_in = int((summary["return_difference_pp"] > 0).sum())

    improved_dd = int((summary["dd_improvement_pp"] > 0).sum())

    avg_return = float(summary["risk_scaled_return_pct"].mean())

    median_return = float(summary["risk_scaled_return_pct"].median())

    baseline_avg_return = float(summary["all_in_return_pct"].mean())

    baseline_median_return = float(summary["all_in_return_pct"].median())

    avg_dd = float(summary["risk_scaled_dd_pct"].mean())

    baseline_avg_dd = float(summary["all_in_dd_pct"].mean())

    avg_weight = float(summary["avg_weight_pct"].mean())

    total_rebalances = int(summary["rebalances"].sum())

    total_fees = float(summary["fees_usdt"].sum())

    print()
    print("AGGREGATE")

    print("-" * 110)

    print(f"Assets:                    " f"{len(summary)}")

    print(f"Profitable assets:         " f"{profitable}")

    print(f"Losing assets:             " f"{losing}")

    print()

    print(f"Risk-scaled avg return:    " f"{avg_return:.2f}%")

    print(f"All-in avg return:         " f"{baseline_avg_return:.2f}%")

    print(f"Risk-scaled median:        " f"{median_return:.2f}%")

    print(f"All-in median:             " f"{baseline_median_return:.2f}%")

    print()

    print(f"Risk-scaled avg DD:        " f"{avg_dd:.2f}%")

    print(f"All-in avg DD:             " f"{baseline_avg_dd:.2f}%")

    print()

    print(
        f"Return higher than all-in: " f"{higher_return_than_all_in}" f"/{len(summary)}"
    )

    print(f"Drawdown improved:         " f"{improved_dd}" f"/{len(summary)}")

    print()

    print(f"Average portfolio weight:  " f"{avg_weight:.2f}%")

    print(f"Total rebalances:          " f"{total_rebalances}")

    print(f"Total fees:                " f"{total_fees:.4f} USDT")


# ============================================================
# MAIN
# ============================================================


def main():

    symbols = load_frozen_universe()

    baseline = load_baseline()

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("=" * 110)

    print("TSMOM REFERENCE — " "RISK-SCALED BACKTEST")

    print("=" * 110)

    print()

    print("Frozen rules:")

    print("  Momentum:          12 months")

    print("  Decision:          month-end")

    print("  Direction:         LONG / CASH")

    print("  Target vol:        40%")

    print("  EWMA COM:          60 days")

    print("  Annualization:     365")

    print("  Max position:      100%")

    print("  Short:             NONE")

    print("  Leverage:          NONE")

    print("  Execution:         next daily OPEN")

    print()

    print(f"Frozen universe: " f"{len(symbols)} assets")

    results = []

    for symbol in symbols:

        result = run_symbol(
            symbol,
            baseline,
        )

        results.append(result)

    summary = pd.DataFrame(results)

    summary.to_csv(
        SUMMARY_FILE,
        index=False,
    )

    print_summary(summary)

    print()

    print(f"Summary saved: " f"{SUMMARY_FILE}")


if __name__ == "__main__":
    main()
