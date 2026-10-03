from pathlib import Path

import pandas as pd

from backtest import (
    run_backtest,
    calculate_buy_and_hold,
)

# ============================================================
# PATHS
# ============================================================

DATA_DIR = Path("data/reference_tsmom")

SIGNALS_DIR = DATA_DIR / "signals"

OUTPUT_DIR = DATA_DIR / "results"

UNIVERSE_FILE = DATA_DIR / "universe_screen.csv"


# ============================================================
# STRATEGY DESCRIPTION
# ============================================================

TIMEFRAME = "1d"

STRATEGY_NAME = "TSMOM Reference " "(12M, Monthly, Long/Cash)"


# ============================================================
# LOAD FROZEN UNIVERSE
# ============================================================


def load_frozen_universe() -> list[str]:

    if not UNIVERSE_FILE.exists():

        raise FileNotFoundError(f"Missing universe file: " f"{UNIVERSE_FILE}")

    dataframe = pd.read_csv(UNIVERSE_FILE)

    required_columns = [
        "symbol",
        "eligible",
    ]

    missing = [column for column in required_columns if column not in dataframe.columns]

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

        raise RuntimeError("Frozen universe contains " "no eligible symbols.")

    return symbols


# ============================================================
# LOAD SIGNAL DATA
# ============================================================


def load_signals(
    symbol: str,
) -> pd.DataFrame:

    file_path = SIGNALS_DIR / (f"{symbol}_" "spot_1d_" "tsmom_reference_" "signals.csv")

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


def trim_to_valid_test_period(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:

    valid_positions = dataframe["target_position"].notna()

    if not valid_positions.any():

        raise RuntimeError("No valid TSMOM " "decisions found.")

    first_valid_index = valid_positions.idxmax()

    result = dataframe.loc[first_valid_index:].copy().reset_index(drop=True)

    return result


# ============================================================
# SAVE RESULTS
# ============================================================


def save_results(
    symbol: str,
    trades: pd.DataFrame,
    equity: pd.DataFrame,
):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    trades_file = OUTPUT_DIR / (f"{symbol}_" "spot_1d_" "tsmom_reference_" "trades.csv")

    equity_file = OUTPUT_DIR / (f"{symbol}_" "spot_1d_" "tsmom_reference_" "equity.csv")

    trades.to_csv(
        trades_file,
        index=False,
    )

    equity.to_csv(
        equity_file,
        index=False,
    )

    return (
        trades_file,
        equity_file,
    )


# ============================================================
# FORMAT PROFIT FACTOR
# ============================================================


def format_profit_factor(
    value,
) -> str:

    if pd.isna(value):
        return "N/A"

    if value == float("inf"):
        return "INF"

    return f"{value:.2f}"


# ============================================================
# RUN ONE SYMBOL
# ============================================================


def run_symbol(
    symbol: str,
) -> dict:

    full_dataframe = load_signals(symbol)

    dataframe = trim_to_valid_test_period(full_dataframe)

    required_columns = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "signal",
    ]

    missing = [column for column in required_columns if column not in dataframe.columns]

    if missing:

        raise RuntimeError(f"{symbol}: missing columns " f"{missing}")

    (
        trades,
        equity,
        stats,
    ) = run_backtest(dataframe)

    buy_and_hold = calculate_buy_and_hold(dataframe)

    (
        trades_file,
        equity_file,
    ) = save_results(
        symbol,
        trades,
        equity,
    )

    print()
    print("=" * 110)

    print(symbol)

    print("=" * 110)

    print(
        f"Test period: "
        f"{dataframe.iloc[0]['timestamp']} "
        f"→ "
        f"{dataframe.iloc[-1]['timestamp']}"
    )

    print(f"Candles: " f"{len(dataframe)}")

    print()

    print("TRADES")

    print("-" * 110)

    print(f"Completed:    " f"{stats['trades']}")

    print(f"Wins:         " f"{stats['wins']}")

    print(f"Losses:       " f"{stats['losses']}")

    print(f"Win rate:     " f"{stats['win_rate']:.2f}%")

    print(f"Forced exits: " f"{stats['forced_exits']}")

    print()

    print("PERFORMANCE")

    print("-" * 110)

    print(f"Final capital: " f"{stats['final_capital']:.2f}" " USDT")

    print(f"Net return:    " f"{stats['total_return']:.2f}%")

    print(f"Buy & Hold:    " f"{buy_and_hold:.2f}%")

    print("Profit factor: " f"{format_profit_factor(stats['profit_factor'])}")

    print()

    print("RISK")

    print("-" * 110)

    print(f"Max drawdown:  " f"{stats['max_drawdown']:.2f}%")

    print(f"Exposure:      " f"{stats['exposure_pct']:.2f}%")

    print(f"Fees:          " f"{stats['total_fees']:.4f}" " USDT")

    print()

    print(f"Trades file: " f"{trades_file}")

    print(f"Equity file: " f"{equity_file}")

    return {
        "symbol": symbol,
        "trades": (stats["trades"]),
        "wins": (stats["wins"]),
        "losses": (stats["losses"]),
        "win_rate_pct": (stats["win_rate"]),
        "return_pct": (stats["total_return"]),
        "buy_hold_pct": (buy_and_hold),
        "profit_factor": (stats["profit_factor"]),
        "max_drawdown_pct": (stats["max_drawdown"]),
        "exposure_pct": (stats["exposure_pct"]),
        "fees_usdt": (stats["total_fees"]),
        "forced_exits": (stats["forced_exits"]),
    }


# ============================================================
# CROSS-ASSET AGGREGATE
# ============================================================


def print_cross_asset_summary(
    summary: pd.DataFrame,
):

    print()
    print()
    print("=" * 110)

    print("TSMOM REFERENCE — " "FROZEN UNIVERSE RESULTS")

    print("=" * 110)

    print()

    display_columns = [
        "symbol",
        "trades",
        "wins",
        "losses",
        "return_pct",
        "buy_hold_pct",
        "profit_factor",
        "max_drawdown_pct",
        "exposure_pct",
    ]

    print(summary[display_columns].to_string(index=False))

    total_trades = int(summary["trades"].sum())

    total_wins = int(summary["wins"].sum())

    total_losses = int(summary["losses"].sum())

    profitable_assets = int((summary["return_pct"] > 0).sum())

    losing_assets = int((summary["return_pct"] < 0).sum())

    flat_assets = len(summary) - profitable_assets - losing_assets

    if total_trades > 0:

        aggregate_win_rate = total_wins / total_trades * 100

    else:

        aggregate_win_rate = 0.0

    average_return = float(summary["return_pct"].mean())

    median_return = float(summary["return_pct"].median())

    worst_return = float(summary["return_pct"].min())

    best_return = float(summary["return_pct"].max())

    average_drawdown = float(summary["max_drawdown_pct"].mean())

    average_exposure = float(summary["exposure_pct"].mean())

    print()
    print("AGGREGATE")

    print("-" * 110)

    print(f"Assets:           " f"{len(summary)}")

    print(f"Profitable:       " f"{profitable_assets}")

    print(f"Losing:           " f"{losing_assets}")

    print(f"Flat:             " f"{flat_assets}")

    print()

    print(f"Total trades:     " f"{total_trades}")

    print(f"Winning trades:   " f"{total_wins}")

    print(f"Losing trades:    " f"{total_losses}")

    print(f"Aggregate WR:     " f"{aggregate_win_rate:.2f}%")

    print()

    print(f"Average return:   " f"{average_return:.2f}%")

    print(f"Median return:    " f"{median_return:.2f}%")

    print(f"Worst asset:      " f"{worst_return:.2f}%")

    print(f"Best asset:       " f"{best_return:.2f}%")

    print(f"Average DD:       " f"{average_drawdown:.2f}%")

    print(f"Average exposure: " f"{average_exposure:.2f}%")


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
    print("=" * 110)

    print("TSMOM REFERENCE — " "FROZEN UNIVERSE BACKTEST")

    print("=" * 110)

    print()

    print("Frozen rules:")

    print("  Lookback:           12 months")

    print("  Decision:           month-end")

    print("  Positive momentum:  LONG")

    print("  Zero/negative:      CASH")

    print("  Execution:          next daily open")

    print("  Short:              disabled")

    print("  Leverage:           disabled")

    print("  MA filter:          none")

    print()

    print(f"Frozen universe: " f"{len(symbols)} assets")

    results = []

    for symbol in symbols:

        result = run_symbol(symbol)

        results.append(result)

    summary = pd.DataFrame(results)

    summary_file = OUTPUT_DIR / ("tsmom_reference_" "frozen_universe_" "summary.csv")

    summary.to_csv(
        summary_file,
        index=False,
    )

    print_cross_asset_summary(summary)

    print()

    print(f"Summary saved: " f"{summary_file}")


if __name__ == "__main__":
    main()
