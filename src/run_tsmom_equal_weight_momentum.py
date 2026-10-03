from pathlib import Path

import pandas as pd

from backtest_portfolio import (
    run_portfolio_backtest,
)

# ============================================================
# FROZEN CONFIGURATION
# ============================================================

INITIAL_CAPITAL = 100.0
FEE_RATE = 0.001


# ============================================================
# PATHS
# ============================================================

DATA_DIR = Path("data/reference_tsmom")

UNIVERSE_FILE = DATA_DIR / "universe_screen.csv"

WEIGHTS_FILE = DATA_DIR / "benchmarks" / "tsmom_equal_weight_momentum_weights.csv"

OUTPUT_DIR = DATA_DIR / "benchmarks" / "equal_weight_momentum_results"

TRANSACTIONS_FILE = OUTPUT_DIR / "transactions.csv"

EQUITY_FILE = OUTPUT_DIR / "equity.csv"

SUMMARY_FILE = OUTPUT_DIR / "summary.csv"


# ============================================================
# RISK-SCALED REFERENCE RESULT
# ============================================================

RISK_SCALED_SUMMARY_FILE = (
    DATA_DIR / "portfolio" / "results" / "tsmom_reference_portfolio_summary.csv"
)


# ============================================================
# LOAD FROZEN UNIVERSE
# ============================================================


def load_frozen_universe() -> list[str]:

    if not UNIVERSE_FILE.exists():

        raise FileNotFoundError(f"Missing universe file: " f"{UNIVERSE_FILE}")

    dataframe = pd.read_csv(UNIVERSE_FILE)

    required = [
        "symbol",
        "eligible",
    ]

    missing = [column for column in required if column not in dataframe.columns]

    if missing:

        raise RuntimeError("Universe file missing columns: " f"{missing}")

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

        raise RuntimeError("Frozen universe is empty.")

    return symbols


# ============================================================
# LOAD PRICE DATA
# ============================================================


def load_price_data(
    symbol: str,
) -> pd.DataFrame:

    file_path = DATA_DIR / f"{symbol}_spot_1d.csv"

    if not file_path.exists():

        raise FileNotFoundError(f"Missing price file: " f"{file_path}")

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
# LOAD BENCHMARK WEIGHTS
# ============================================================


def load_weights(
    symbols: list[str],
) -> pd.DataFrame:

    if not WEIGHTS_FILE.exists():

        raise FileNotFoundError(f"Missing weights file: " f"{WEIGHTS_FILE}")

    dataframe = pd.read_csv(
        WEIGHTS_FILE,
        parse_dates=["timestamp"],
    )

    required = [
        "timestamp",
    ] + [f"{symbol}_weight" for symbol in symbols]

    missing = [column for column in required if column not in dataframe.columns]

    if missing:

        raise RuntimeError("Weights file missing columns: " f"{missing}")

    dataframe = dataframe.sort_values("timestamp").reset_index(drop=True)

    return dataframe


# ============================================================
# LOAD RISK-SCALED RESULT
# ============================================================


def load_risk_scaled_result():

    if not RISK_SCALED_SUMMARY_FILE.exists():

        return None

    dataframe = pd.read_csv(RISK_SCALED_SUMMARY_FILE)

    if dataframe.empty:
        return None

    return dataframe.iloc[0]


# ============================================================
# SAVE BENCHMARK RESULT
# ============================================================


def save_results(
    transactions: pd.DataFrame,
    equity: pd.DataFrame,
    stats: dict,
    symbols: list[str],
):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
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
                "universe_size": (len(symbols)),
                "initial_capital": (stats["initial_capital"]),
                "final_capital": (stats["final_capital"]),
                "return_pct": (stats["total_return"]),
                "max_drawdown_pct": (stats["max_drawdown"]),
                "average_exposure_pct": (stats["average_exposure_pct"]),
                "invested_days_pct": (stats["invested_days_pct"]),
                "rebalance_events": (stats["rebalance_events"]),
                "transactions": (stats["transaction_count"]),
                "fees_usdt": (stats["total_fees"]),
                "total_notional_usdt": (stats["total_notional"]),
                "forced_exits": (stats["forced_exits"]),
                "start": (equity.iloc[0]["timestamp"]),
                "end": (equity.iloc[-1]["timestamp"]),
            }
        ]
    )

    summary.to_csv(
        SUMMARY_FILE,
        index=False,
    )


# ============================================================
# PRINT RESULT
# ============================================================


def print_result(
    symbols: list[str],
    weights: pd.DataFrame,
    equity: pd.DataFrame,
    transactions: pd.DataFrame,
    stats: dict,
):

    print()
    print("=" * 110)

    print("TSMOM EQUAL-WEIGHT MOMENTUM " "BENCHMARK RESULT")

    print("=" * 110)

    print()

    print("BENCHMARK RULES")

    print("-" * 110)

    print("Momentum:          " "12 months")

    print("Decision:          " "month-end")

    print("Positive:          " "LONG")

    print("Zero/negative:     " "CASH")

    print("Position sizing:   " "EQUAL WEIGHT")

    print("Leverage:          " "NONE")

    print("Short:             " "NONE")

    print("Execution:         " "NEXT DAILY OPEN")

    print(f"Fee rate:          " f"{FEE_RATE * 100:.3f}%")

    print()

    print("UNIVERSE")

    print("-" * 110)

    print(f"Assets: " f"{len(symbols)}")

    print(", ".join(symbols))

    print()

    print("PERFORMANCE")

    print("-" * 110)

    print(f"Initial capital:     " f"{stats['initial_capital']:.2f} USDT")

    print(f"Final capital:       " f"{stats['final_capital']:.2f} USDT")

    print(f"Net return:          " f"{stats['total_return']:.2f}%")

    print(f"Maximum drawdown:    " f"{stats['max_drawdown']:.2f}%")

    print()

    print("EXPOSURE")

    print("-" * 110)

    print(f"Average exposure:    " f"{stats['average_exposure_pct']:.2f}%")

    print(f"Invested days:       " f"{stats['invested_days_pct']:.2f}%")

    print()

    print("TRADING")

    print("-" * 110)

    print(f"Monthly decisions:   " f"{len(weights)}")

    print(f"Rebalance events:    " f"{stats['rebalance_events']}")

    print(f"Transactions:        " f"{stats['transaction_count']}")

    print(f"Total traded value:  " f"{stats['total_notional']:.2f} USDT")

    print(f"Total fees:          " f"{stats['total_fees']:.4f} USDT")

    print(f"Forced liquidation:  " f"{stats['forced_exits']}")

    # ========================================================
    # COMPARE AGAINST RISK-SCALED VERSION
    # ========================================================

    risk_scaled = load_risk_scaled_result()

    if risk_scaled is not None:

        risk_return = float(risk_scaled["return_pct"])

        risk_dd = float(risk_scaled["max_drawdown_pct"])

        risk_exposure = float(risk_scaled["average_exposure_pct"])

        risk_fees = float(risk_scaled["fees_usdt"])

        print()
        print("=" * 110)

        print("DIRECT COMPARISON")

        print("=" * 110)

        print()

        print(
            f"{'Metric':<30}"
            f"{'Risk-Scaled':>18}"
            f"{'Equal-Weight':>18}"
            f"{'Difference':>18}"
        )

        print("-" * 84)

        print(
            f"{'Return':<30}"
            f"{risk_return:>17.2f}%"
            f"{stats['total_return']:>17.2f}%"
            f"{stats['total_return'] - risk_return:>+17.2f} pp"
        )

        print(
            f"{'Max Drawdown':<30}"
            f"{risk_dd:>17.2f}%"
            f"{stats['max_drawdown']:>17.2f}%"
            f"{stats['max_drawdown'] - risk_dd:>+17.2f} pp"
        )

        print(
            f"{'Average Exposure':<30}"
            f"{risk_exposure:>17.2f}%"
            f"{stats['average_exposure_pct']:>17.2f}%"
            f"{stats['average_exposure_pct'] - risk_exposure:>+17.2f} pp"
        )

        print(
            f"{'Fees':<30}"
            f"{risk_fees:>17.4f}"
            f"{stats['total_fees']:>18.4f}"
            f"{stats['total_fees'] - risk_fees:>+18.4f}"
        )

    print()

    print("EXECUTION AUDIT")

    print("-" * 110)

    if not transactions.empty:

        normal = transactions[transactions["forced"] == False]

        if not normal.empty:

            first = normal.iloc[0]

            print(f"First signal:    " f"{first['signal_timestamp']}")

            print(f"First execution: " f"{first['execution_timestamp']}")

            print(f"Asset:           " f"{first['symbol']}")

            print(f"Side:            " f"{first['side']}")

    print()

    print("OUTPUT")

    print("-" * 110)

    print(f"Transactions: " f"{TRANSACTIONS_FILE}")

    print(f"Equity:       " f"{EQUITY_FILE}")

    print(f"Summary:      " f"{SUMMARY_FILE}")


# ============================================================
# MAIN
# ============================================================


def main():

    symbols = load_frozen_universe()

    weights = load_weights(symbols)

    price_data = {symbol: (load_price_data(symbol)) for symbol in symbols}

    print()
    print("=" * 110)

    print("RUNNING EQUAL-WEIGHT " "MOMENTUM BENCHMARK")

    print("=" * 110)

    (
        transactions,
        equity,
        stats,
    ) = run_portfolio_backtest(
        price_data=price_data,
        weights=weights,
        initial_capital=(INITIAL_CAPITAL),
        fee_rate=(FEE_RATE),
        force_liquidation=True,
    )

    save_results(
        transactions=transactions,
        equity=equity,
        stats=stats,
        symbols=symbols,
    )

    print_result(
        symbols=symbols,
        weights=weights,
        equity=equity,
        transactions=transactions,
        stats=stats,
    )


if __name__ == "__main__":
    main()
