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

WEIGHTS_FILE = DATA_DIR / "portfolio" / "tsmom_reference_portfolio_weights.csv"

RESULTS_DIR = DATA_DIR / "portfolio" / "results"

TRANSACTIONS_FILE = RESULTS_DIR / "tsmom_reference_portfolio_transactions.csv"

EQUITY_FILE = RESULTS_DIR / "tsmom_reference_portfolio_equity.csv"

SUMMARY_FILE = RESULTS_DIR / "tsmom_reference_portfolio_summary.csv"


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
# LOAD DAILY PRICE DATA
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
# LOAD PORTFOLIO WEIGHTS
# ============================================================


def load_portfolio_weights(
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

        raise RuntimeError("Portfolio weights missing columns: " f"{missing}")

    dataframe = dataframe.sort_values("timestamp").reset_index(drop=True)

    return dataframe


# ============================================================
# SAVE RESULTS
# ============================================================


def save_results(
    transactions: pd.DataFrame,
    equity: pd.DataFrame,
    stats: dict,
    symbols: list[str],
):

    RESULTS_DIR.mkdir(
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
# TRANSACTION AUDIT
# ============================================================


def print_transaction_audit(
    transactions: pd.DataFrame,
):

    print()
    print("EXECUTION AUDIT")

    print("-" * 110)

    if transactions.empty:

        print("No transactions.")

        return

    normal = transactions[transactions["forced"] == False].copy()

    if not normal.empty:

        first = normal.iloc[0]

        last = normal.iloc[-1]

        print(f"First signal:       " f"{first['signal_timestamp']}")

        print(f"First execution:    " f"{first['execution_timestamp']}")

        print(f"First asset:        " f"{first['symbol']}")

        print(f"First side:         " f"{first['side']}")

        print()

        print(f"Last normal signal: " f"{last['signal_timestamp']}")

        print(f"Last execution:     " f"{last['execution_timestamp']}")

    print()

    forced_count = int((transactions["forced"] == True).sum())

    print(f"Forced transaction rows: " f"{forced_count}")


# ============================================================
# FINAL PORTFOLIO SUMMARY
# ============================================================


def print_summary(
    symbols: list[str],
    weights: pd.DataFrame,
    equity: pd.DataFrame,
    transactions: pd.DataFrame,
    stats: dict,
):

    print()
    print("=" * 110)

    print("TSMOM REFERENCE — " "MULTI-ASSET PORTFOLIO RESULT")

    print("=" * 110)

    print()

    print("FROZEN STRATEGY")

    print("-" * 110)

    print("Momentum:             " "12 months")

    print("Decision:             " "month-end")

    print("Direction:            " "LONG / CASH")

    print("Target volatility:    " "40% annualized")

    print("EWMA COM:             " "60 days")

    print("Annualization:        " "365 days")

    print("Maximum asset weight: " "100% before portfolio normalization")

    print("Portfolio leverage:   " "NONE")

    print("Short selling:        " "NONE")

    print("Execution:            " "NEXT DAILY OPEN")

    print(f"Fee rate:             " f"{FEE_RATE * 100:.3f}%")

    print()

    print("UNIVERSE")

    print("-" * 110)

    print(f"Assets: " f"{len(symbols)}")

    print(", ".join(symbols))

    print()

    print("PERIOD")

    print("-" * 110)

    print(f"First portfolio decision: " f"{weights.iloc[0]['timestamp']}")

    print(f"Last portfolio decision:  " f"{weights.iloc[-1]['timestamp']}")

    print(f"Equity start:             " f"{equity.iloc[0]['timestamp']}")

    print(f"Equity end:               " f"{equity.iloc[-1]['timestamp']}")

    print(f"Daily equity observations: " f"{len(equity)}")

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

    print_transaction_audit(transactions)

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

    weights = load_portfolio_weights(symbols)

    price_data = {symbol: (load_price_data(symbol)) for symbol in symbols}

    print()
    print("=" * 110)

    print("RUNNING TSMOM REFERENCE " "MULTI-ASSET PORTFOLIO")

    print("=" * 110)

    print()

    print(f"Initial capital: " f"{INITIAL_CAPITAL:.2f} USDT")

    print(f"Frozen assets:   " f"{len(symbols)}")

    print(f"Monthly targets: " f"{len(weights)}")

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

    print_summary(
        symbols=symbols,
        weights=weights,
        equity=equity,
        transactions=transactions,
        stats=stats,
    )


if __name__ == "__main__":
    main()
