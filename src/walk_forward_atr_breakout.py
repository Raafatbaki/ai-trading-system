from pathlib import Path

import pandas as pd

from backtest import run_backtest

from strategies.atr_breakout import (
    calculate_features,
    generate_signals,
)

MARKET = "spot"
TIMEFRAME = "4h"

SYMBOLS = [
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
]

INITIAL_CAPITAL = 100.0

# EMA slow = 200 DAYS.
# Therefore the first test starts only
# after 200 days of historical context.
INITIAL_CONTEXT_DAYS = 200

# Approximately six months per temporal fold.
FOLD_DAYS = 183


def get_dataset_file(
    symbol: str,
    dataset: str,
) -> Path:

    return Path(f"data/splits/" f"{symbol}_{MARKET}_{TIMEFRAME}_{dataset}.csv")


def load_csv(
    file_path: Path,
) -> pd.DataFrame:

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    dataframe = pd.read_csv(
        file_path,
        parse_dates=["timestamp"],
    )

    return (
        dataframe.sort_values("timestamp")
        .drop_duplicates(subset=["timestamp"])
        .reset_index(drop=True)
    )


def load_research_data(
    symbol: str,
) -> pd.DataFrame:

    development = load_csv(
        get_dataset_file(
            symbol,
            "development",
        )
    )

    validation = load_csv(
        get_dataset_file(
            symbol,
            "validation",
        )
    )

    dataframe = pd.concat(
        [
            development,
            validation,
        ],
        ignore_index=True,
    )

    dataframe = (
        dataframe.sort_values("timestamp")
        .drop_duplicates(
            subset=["timestamp"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    # historical_holdout is intentionally
    # NOT loaded here.
    return dataframe


def calculate_global_drawdown(
    equity_dataframe: pd.DataFrame,
) -> float:

    if equity_dataframe.empty:
        return float("nan")

    equity = equity_dataframe["continuous_equity"]

    peak = equity.cummax()

    drawdown = ((equity / peak) - 1) * 100

    return float(drawdown.min())


def run_symbol_walk_forward(
    symbol: str,
) -> tuple[
    list[dict],
    pd.DataFrame,
]:

    dataframe = load_research_data(symbol)

    # Features are calculated once on the
    # chronological Development + Validation
    # sequence.
    #
    # All indicators are causal:
    # they use only current / previous data.
    dataframe = calculate_features(
        dataframe=dataframe,
        timeframe=TIMEFRAME,
    )

    first_timestamp = dataframe["timestamp"].min()

    last_timestamp = dataframe["timestamp"].max()

    test_start = first_timestamp + pd.Timedelta(days=INITIAL_CONTEXT_DAYS)

    fold_number = 1

    continuous_capital = INITIAL_CAPITAL

    fold_results = []

    continuous_equity_parts = []

    while test_start <= last_timestamp:

        test_end = test_start + pd.Timedelta(days=FOLD_DAYS)

        fold = dataframe[
            (dataframe["timestamp"] >= test_start) & (dataframe["timestamp"] < test_end)
        ].copy()

        if fold.empty:
            break

        fold = fold.reset_index(drop=True)

        # Each fold starts CASH.
        #
        # Entry/exit parameters remain frozen:
        # 20-day breakout
        # 14-day ATR
        # EMA50d / EMA200d
        # 0.25 ATR entry buffer
        # 3 ATR trailing exit
        fold = generate_signals(fold)

        (
            trades,
            equity,
            stats,
        ) = run_backtest(fold)

        fold_start_capital = continuous_capital

        # Backtester internally starts each
        # independent run at 100 USDT.
        fold_multiplier = stats["final_capital"] / INITIAL_CAPITAL

        continuous_capital = fold_start_capital * fold_multiplier

        scale_factor = fold_start_capital / INITIAL_CAPITAL

        fold_equity = equity.copy()

        fold_equity["continuous_equity"] = fold_equity["equity"] * scale_factor

        fold_equity["symbol"] = symbol

        fold_equity["fold"] = fold_number

        continuous_equity_parts.append(
            fold_equity[
                [
                    "timestamp",
                    "symbol",
                    "fold",
                    "continuous_equity",
                ]
            ]
        )

        fold_results.append(
            {
                "symbol": symbol,
                "fold": fold_number,
                "start": (fold["timestamp"].min()),
                "end": (fold["timestamp"].max()),
                "candles": len(fold),
                "start_capital": (fold_start_capital),
                "end_capital": (continuous_capital),
                "return_pct": (stats["total_return"]),
                "trades": (stats["trades"]),
                "wins": (stats["wins"]),
                "losses": (stats["losses"]),
                "win_rate": (stats["win_rate"]),
                "profit_factor": (stats["profit_factor"]),
                "max_drawdown": (stats["max_drawdown"]),
                "exposure_pct": (stats["exposure_pct"]),
                "fees": (stats["total_fees"] * scale_factor),
                "forced_exits": (stats["forced_exits"]),
            }
        )

        fold_number += 1
        test_start = test_end

    if continuous_equity_parts:

        continuous_equity = pd.concat(
            continuous_equity_parts,
            ignore_index=True,
        )

    else:

        continuous_equity = pd.DataFrame()

    return (
        fold_results,
        continuous_equity,
    )


def format_pf(
    value: float,
) -> str:

    if pd.isna(value):
        return "N/A"

    if value == float("inf"):
        return "INF"

    return f"{value:.2f}"


def main():

    output_directory = Path("data/walk_forward")

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_results = []

    print()
    print("=" * 90)
    print("ATR BREAKOUT TEMPORAL ROBUSTNESS TEST")
    print("=" * 90)

    print()
    print("Data:             DEVELOPMENT + VALIDATION")
    print("Holdout:          NOT USED")
    print(f"Timeframe:        {TIMEFRAME}")

    print("Parameters:       FROZEN")

    print("Breakout:         20 days")
    print("ATR:              14 days")
    print("Trend:            EMA50d > EMA200d")
    print("Breakout buffer:  0.25 ATR")
    print("Trailing exit:    3 ATR")

    print(f"Initial context:  " f"{INITIAL_CONTEXT_DAYS} days")

    print(f"Fold size:        " f"{FOLD_DAYS} days")

    print("Fold state:       STARTS CASH")

    print("Capital:          CONTINUOUS BETWEEN FOLDS")

    for symbol in SYMBOLS:

        (
            results,
            equity,
        ) = run_symbol_walk_forward(symbol)

        all_results.extend(results)

        print()
        print("-" * 90)
        print(symbol)
        print("-" * 90)

        for result in results:

            print()

            print(f"Fold {result['fold']}: " f"{result['start']} " f"→ {result['end']}")

            print(f"  Return:        " f"{result['return_pct']:.2f}%")

            print(f"  Trades:        " f"{result['trades']}")

            print(f"  Win rate:      " f"{result['win_rate']:.2f}%")

            print(f"  Profit factor: " f"{format_pf(result['profit_factor'])}")

            print(f"  Max DD:        " f"{result['max_drawdown']:.2f}%")

            print(f"  Exposure:      " f"{result['exposure_pct']:.2f}%")

            print(f"  Forced exits:  " f"{result['forced_exits']}")

            print(
                f"  Capital:       "
                f"{result['start_capital']:.2f}"
                f" → "
                f"{result['end_capital']:.2f}"
            )

        if not equity.empty:

            symbol_drawdown = calculate_global_drawdown(equity)

            symbol_final_capital = float(equity.iloc[-1]["continuous_equity"])

            symbol_return = ((symbol_final_capital / INITIAL_CAPITAL) - 1) * 100

            print()

            print(f"{symbol} continuous return: " f"{symbol_return:.2f}%")

            print(f"{symbol} continuous max DD: " f"{symbol_drawdown:.2f}%")

            equity_file = output_directory / (
                f"{symbol}_{MARKET}_"
                f"{TIMEFRAME}_"
                f"atr_breakout_"
                f"walk_forward_equity.csv"
            )

            equity.to_csv(
                equity_file,
                index=False,
            )

    results_dataframe = pd.DataFrame(all_results)

    if results_dataframe.empty:

        raise RuntimeError("No walk-forward folds were generated.")

    results_file = output_directory / ("atr_breakout_4h_" "walk_forward_folds.csv")

    results_dataframe.to_csv(
        results_file,
        index=False,
    )

    total_folds = len(results_dataframe)

    profitable_folds = int((results_dataframe["return_pct"] > 0).sum())

    losing_folds = int((results_dataframe["return_pct"] < 0).sum())

    flat_folds = total_folds - profitable_folds - losing_folds

    total_trades = int(results_dataframe["trades"].sum())

    total_wins = int(results_dataframe["wins"].sum())

    total_losses = int(results_dataframe["losses"].sum())

    if total_trades > 0:

        total_win_rate = total_wins / total_trades * 100

    else:

        total_win_rate = 0.0

    pf_above_one = int((results_dataframe["profit_factor"] > 1).sum())

    print()
    print("=" * 90)
    print("AGGREGATE")
    print("=" * 90)

    print()

    print(f"Total folds:       " f"{total_folds}")

    print(f"Profitable folds:  " f"{profitable_folds}")

    print(f"Losing folds:      " f"{losing_folds}")

    print(f"Flat folds:        " f"{flat_folds}")

    print(f"PF > 1 folds:      " f"{pf_above_one}")

    print()

    print(f"Total trades:      " f"{total_trades}")

    print(f"Winning trades:    " f"{total_wins}")

    print(f"Losing trades:     " f"{total_losses}")

    print(f"Win rate:          " f"{total_win_rate:.2f}%")

    print()

    print(f"Average fold:      " f"{results_dataframe['return_pct'].mean():.2f}%")

    print(f"Median fold:       " f"{results_dataframe['return_pct'].median():.2f}%")

    print(f"Worst fold:        " f"{results_dataframe['return_pct'].min():.2f}%")

    print(f"Best fold:         " f"{results_dataframe['return_pct'].max():.2f}%")

    print(f"Average max DD:    " f"{results_dataframe['max_drawdown'].mean():.2f}%")

    print(f"Forced exits:      " f"{int(results_dataframe['forced_exits'].sum())}")

    print()

    print(f"Results saved to: " f"{results_file}")


if __name__ == "__main__":
    main()
