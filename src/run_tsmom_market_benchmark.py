from pathlib import Path

import pandas as pd

from backtest_portfolio import (
    run_portfolio_backtest,
)

# ============================================================
# CONFIGURATION
# ============================================================

INITIAL_CAPITAL = 100.0
FEE_RATE = 0.001


# ============================================================
# PATHS
# ============================================================

DATA_DIR = Path("data/reference_tsmom")

UNIVERSE_FILE = DATA_DIR / "universe_screen.csv"

WEIGHTS_FILE = DATA_DIR / "benchmarks" / "equal_weight_market_weights.csv"

OUTPUT_DIR = DATA_DIR / "benchmarks" / "equal_weight_market_results"

TRANSACTIONS_FILE = OUTPUT_DIR / "transactions.csv"

EQUITY_FILE = OUTPUT_DIR / "equity.csv"

SUMMARY_FILE = OUTPUT_DIR / "summary.csv"


# ============================================================
# EXISTING COMPARISON RESULTS
# ============================================================

RISK_SCALED_FILE = (
    DATA_DIR / "portfolio" / "results" / "tsmom_reference_portfolio_summary.csv"
)

MOMENTUM_EQUAL_WEIGHT_FILE = (
    DATA_DIR / "benchmarks" / "equal_weight_momentum_results" / "summary.csv"
)


# ============================================================
# LOAD FROZEN UNIVERSE
# ============================================================


def load_frozen_universe() -> list[str]:

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
# LOAD MARKET WEIGHTS
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

    return dataframe.sort_values("timestamp").reset_index(drop=True)


# ============================================================
# LOAD ONE EXISTING SUMMARY
# ============================================================


def load_summary(
    file_path: Path,
):

    if not file_path.exists():
        return None

    dataframe = pd.read_csv(file_path)

    if dataframe.empty:
        return None

    return dataframe.iloc[0]


# ============================================================
# SAVE MARKET RESULT
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
# THREE-WAY COMPARISON
# ============================================================


def print_comparison(
    market_stats: dict,
):

    risk = load_summary(RISK_SCALED_FILE)

    momentum_equal = load_summary(MOMENTUM_EQUAL_WEIGHT_FILE)

    print()
    print("=" * 110)

    print("THREE-WAY PORTFOLIO COMPARISON")

    print("=" * 110)

    if risk is None or momentum_equal is None:

        print("One or more previous " "benchmark summaries are missing.")

        return

    rows = [
        {
            "name": "Risk-Scaled Momentum",
            "return": float(risk["return_pct"]),
            "dd": float(risk["max_drawdown_pct"]),
            "exposure": float(risk["average_exposure_pct"]),
            "fees": float(risk["fees_usdt"]),
        },
        {
            "name": "Equal-Weight Momentum",
            "return": float(momentum_equal["return_pct"]),
            "dd": float(momentum_equal["max_drawdown_pct"]),
            "exposure": float(momentum_equal["average_exposure_pct"]),
            "fees": float(momentum_equal["fees_usdt"]),
        },
        {
            "name": "Equal-Weight Market",
            "return": float(market_stats["total_return"]),
            "dd": float(market_stats["max_drawdown"]),
            "exposure": float(market_stats["average_exposure_pct"]),
            "fees": float(market_stats["total_fees"]),
        },
    ]

    print()

    print(
        f"{'Portfolio':<28}"
        f"{'Return':>14}"
        f"{'Max DD':>14}"
        f"{'Exposure':>14}"
        f"{'Fees':>14}"
    )

    print("-" * 84)

    for row in rows:

        print(
            f"{row['name']:<28}"
            f"{row['return']:>13.2f}%"
            f"{row['dd']:>13.2f}%"
            f"{row['exposure']:>13.2f}%"
            f"{row['fees']:>14.4f}"
        )


# ============================================================
# PRINT MARKET RESULT
# ============================================================


def print_result(
    symbols,
    weights,
    equity,
    transactions,
    stats,
):

    print()
    print("=" * 110)

    print("EQUAL-WEIGHT MARKET " "BENCHMARK RESULT")

    print("=" * 110)

    print()

    print("RULES")

    print("-" * 110)

    print("Momentum filter:     NONE")

    print("Eligibility:         " "12 months history")

    print("Allocation:          " "equal weight")

    print("Exposure:            " "100% when eligible")

    print("Rebalance:           " "monthly")

    print("Leverage:            NONE")

    print("Short:               NONE")

    print("Execution:           " "NEXT DAILY OPEN")

    print(f"Fee rate:            " f"{FEE_RATE * 100:.3f}%")

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

    print_comparison(stats)

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

    print("RUNNING EQUAL-WEIGHT " "MARKET BENCHMARK")

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
