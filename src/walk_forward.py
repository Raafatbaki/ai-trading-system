import argparse
from pathlib import Path

import pandas as pd

from backtest import run_backtest
from strategies.time_series_momentum import (
    calculate_features,
    generate_signals,
    get_strategy_periods,
)

MARKET = "spot"
DATA_DIR = Path("data/splits")

SUPPORTED_SYMBOLS = [
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
]

SUPPORTED_TIMEFRAMES = [
    "15m",
    "1h",
    "4h",
]

SUPPORTED_REBALANCES = [
    "weekly",
    "daily",
]


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Continuous walk-forward evaluation for TSMOM V2. "
            "Uses development + validation only. "
            "Historical holdout is never loaded."
        )
    )

    parser.add_argument(
        "--symbols",
        nargs="+",
        choices=SUPPORTED_SYMBOLS,
        default=SUPPORTED_SYMBOLS,
    )

    parser.add_argument(
        "--timeframe",
        choices=SUPPORTED_TIMEFRAMES,
        default="1h",
    )

    parser.add_argument(
        "--rebalance",
        choices=SUPPORTED_REBALANCES,
        default="daily",
    )

    parser.add_argument(
        "--fold-days",
        type=int,
        default=183,
    )

    return parser.parse_args()


def load_research_data(
    symbol: str,
    timeframe: str,
) -> pd.DataFrame:
    development_file = DATA_DIR / f"{symbol}_{MARKET}_{timeframe}_development.csv"

    validation_file = DATA_DIR / f"{symbol}_{MARKET}_{timeframe}_validation.csv"

    if not development_file.exists():
        raise FileNotFoundError(f"Missing file: {development_file}")

    if not validation_file.exists():
        raise FileNotFoundError(f"Missing file: {validation_file}")

    development = pd.read_csv(
        development_file,
        parse_dates=["timestamp"],
    )

    validation = pd.read_csv(
        validation_file,
        parse_dates=["timestamp"],
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
        .drop_duplicates(subset="timestamp")
        .reset_index(drop=True)
    )

    return dataframe


def prepare_strategy(
    dataframe: pd.DataFrame,
    timeframe: str,
    rebalance: str,
):
    (
        momentum_lookback,
        long_trend_window,
    ) = get_strategy_periods(timeframe)

    dataframe = calculate_features(
        dataframe=dataframe,
        momentum_lookback=momentum_lookback,
        long_trend_window=long_trend_window,
        rebalance=rebalance,
    )

    required_history = max(
        momentum_lookback,
        long_trend_window,
    )

    if len(dataframe) <= required_history:
        raise RuntimeError("Not enough data for strategy warm-up.")

    # لا نبدأ التداول قبل اكتمال MA200.
    tradable = dataframe.iloc[required_history:].copy()

    tradable = tradable.reset_index(drop=True)

    # مهم:
    # الإشارات تُولد مرة واحدة على كامل الفترة،
    # وليس بشكل منفصل لكل Fold.
    tradable = generate_signals(tradable)

    return tradable


def build_fold_ranges(
    dataframe: pd.DataFrame,
    fold_days: int,
):
    first_time = pd.Timestamp(dataframe["timestamp"].min())

    final_time = pd.Timestamp(dataframe["timestamp"].max())

    fold_start = first_time
    fold_number = 1

    while fold_start <= final_time:
        fold_end = fold_start + pd.Timedelta(days=fold_days)

        yield (
            fold_number,
            fold_start,
            fold_end,
        )

        fold_start = fold_end
        fold_number += 1


def calculate_local_drawdown(
    equity_values: pd.Series,
    starting_equity: float,
) -> float:
    values = pd.concat(
        [
            pd.Series([starting_equity]),
            equity_values.reset_index(drop=True),
        ],
        ignore_index=True,
    )

    peaks = values.cummax()

    drawdowns = (values / peaks - 1) * 100

    return float(drawdowns.min())


def evaluate_fold(
    equity: pd.DataFrame,
    trades: pd.DataFrame,
    fold_start,
    fold_end,
):
    fold_equity = equity[
        (equity["timestamp"] >= fold_start) & (equity["timestamp"] < fold_end)
    ].copy()

    if fold_equity.empty:
        return None

    previous_equity = equity[equity["timestamp"] < fold_start]

    if previous_equity.empty:
        starting_equity = float(fold_equity.iloc[0]["equity"])
    else:
        starting_equity = float(previous_equity.iloc[-1]["equity"])

    ending_equity = float(fold_equity.iloc[-1]["equity"])

    fold_return = (ending_equity / starting_equity - 1) * 100

    max_drawdown = calculate_local_drawdown(
        fold_equity["equity"],
        starting_equity,
    )

    exposure_pct = fold_equity["position_open"].astype(bool).mean() * 100

    if trades.empty:
        fold_trades = trades.copy()
    else:
        trades = trades.copy()

        trades["exit_time"] = pd.to_datetime(
            trades["exit_time"],
            utc=True,
        )

        fold_trades = trades[
            (trades["exit_time"] >= fold_start) & (trades["exit_time"] < fold_end)
        ].copy()

    trade_count = len(fold_trades)

    if trade_count > 0:
        wins = int((fold_trades["pnl_usdt"] > 0).sum())

        losses = int((fold_trades["pnl_usdt"] < 0).sum())

        win_rate = wins / trade_count * 100

        winning = fold_trades[fold_trades["pnl_usdt"] > 0]

        losing = fold_trades[fold_trades["pnl_usdt"] < 0]

        gross_profit = float(winning["pnl_usdt"].sum())

        gross_loss = abs(float(losing["pnl_usdt"].sum()))

        if gross_loss > 0:
            profit_factor = gross_profit / gross_loss
        elif gross_profit > 0:
            profit_factor = float("inf")
        else:
            profit_factor = 0.0

        forced_exits = int(fold_trades["forced_exit"].sum())

    else:
        wins = 0
        losses = 0
        win_rate = 0.0
        profit_factor = float("nan")
        forced_exits = 0

    return {
        "candles": len(fold_equity),
        "trades": trade_count,
        "wins": wins,
        "losses": losses,
        "win_rate_pct": win_rate,
        "return_pct": fold_return,
        "profit_factor": profit_factor,
        "max_dd_pct": max_drawdown,
        "exposure_pct": exposure_pct,
        "starting_equity": starting_equity,
        "ending_equity": ending_equity,
        "forced_exits": forced_exits,
    }


def run_symbol(
    symbol: str,
    timeframe: str,
    rebalance: str,
    fold_days: int,
):
    raw_data = load_research_data(
        symbol=symbol,
        timeframe=timeframe,
    )

    signals = prepare_strategy(
        dataframe=raw_data,
        timeframe=timeframe,
        rebalance=rebalance,
    )

    # ---------------------------------------
    # ONE CONTINUOUS BACKTEST
    # ---------------------------------------

    (
        trades,
        equity,
        stats,
    ) = run_backtest(signals)

    equity["timestamp"] = pd.to_datetime(
        equity["timestamp"],
        utc=True,
    )

    results = []

    for (
        fold_number,
        fold_start,
        fold_end,
    ) in build_fold_ranges(
        dataframe=signals,
        fold_days=fold_days,
    ):
        result = evaluate_fold(
            equity=equity,
            trades=trades,
            fold_start=fold_start,
            fold_end=fold_end,
        )

        if result is None:
            continue

        result.update(
            {
                "symbol": symbol,
                "timeframe": timeframe,
                "rebalance": rebalance,
                "fold": fold_number,
                "from": fold_start,
                "to": min(
                    fold_end,
                    equity["timestamp"].max(),
                ),
            }
        )

        results.append(result)

    return (
        results,
        stats,
    )


def print_symbol_summary(
    dataframe: pd.DataFrame,
    symbol: str,
):
    current = dataframe[dataframe["symbol"] == symbol]

    print()
    print("=" * 120)
    print(f"{symbol} - CONTINUOUS WALK-FORWARD")
    print("=" * 120)

    columns = [
        "fold",
        "from",
        "to",
        "trades",
        "wins",
        "losses",
        "return_pct",
        "profit_factor",
        "max_dd_pct",
        "exposure_pct",
        "starting_equity",
        "ending_equity",
        "forced_exits",
    ]

    display = current[columns].copy()

    numeric_columns = [
        "return_pct",
        "profit_factor",
        "max_dd_pct",
        "exposure_pct",
        "starting_equity",
        "ending_equity",
    ]

    display[numeric_columns] = display[numeric_columns].round(2)

    print()
    print(display.to_string(index=False))


def print_global_summary(
    dataframe: pd.DataFrame,
):
    print()
    print("=" * 120)
    print("CROSS-ASSET CONTINUOUS " "WALK-FORWARD SUMMARY")
    print("=" * 120)

    total_trades = int(dataframe["trades"].sum())

    total_wins = int(dataframe["wins"].sum())

    total_losses = int(dataframe["losses"].sum())

    profitable_folds = int((dataframe["return_pct"] > 0).sum())

    losing_folds = int((dataframe["return_pct"] < 0).sum())

    flat_folds = int((dataframe["return_pct"] == 0).sum())

    fold_count = len(dataframe)

    if total_trades > 0:
        overall_win_rate = total_wins / total_trades * 100
    else:
        overall_win_rate = 0.0

    print()
    print(f"Fold results:       " f"{fold_count}")

    print(f"Profitable folds:   " f"{profitable_folds}")

    print(f"Losing folds:       " f"{losing_folds}")

    print(f"Flat folds:         " f"{flat_folds}")

    print(f"Total trades:       " f"{total_trades}")

    print(f"Total wins:         " f"{total_wins}")

    print(f"Total losses:       " f"{total_losses}")

    print(f"Overall win rate:   " f"{overall_win_rate:.2f}%")

    print(f"Average fold return:" f" " f"{dataframe['return_pct'].mean():.2f}%")

    print(f"Median fold return: " f"{dataframe['return_pct'].median():.2f}%")

    print(f"Worst fold return:  " f"{dataframe['return_pct'].min():.2f}%")

    print(f"Best fold return:   " f"{dataframe['return_pct'].max():.2f}%")

    print(f"Average max DD:     " f"{dataframe['max_dd_pct'].mean():.2f}%")


def main():
    args = parse_arguments()

    print()
    print("=" * 100)
    print("TSMOM V2 CONTINUOUS " "WALK-FORWARD RESEARCH")
    print("=" * 100)

    print("Symbols:      " + ", ".join(args.symbols))

    print(f"Timeframe:    " f"{args.timeframe}")

    print(f"Rebalance:    " f"{args.rebalance}")

    print(f"Fold length:  " f"{args.fold_days} days")

    print("Data used:    " "DEVELOPMENT + VALIDATION")

    print("Holdout:      NOT READ")

    print("Trading:      CONTINUOUS " "across fold boundaries")

    print("Forced close: ONLY at final " "end of research dataset")

    all_results = []

    for symbol in args.symbols:
        (
            symbol_results,
            full_stats,
        ) = run_symbol(
            symbol=symbol,
            timeframe=args.timeframe,
            rebalance=args.rebalance,
            fold_days=args.fold_days,
        )

        all_results.extend(symbol_results)

        print()
        print(f"{symbol} full continuous " f"backtest:")

        print(f"  Trades:      " f"{full_stats['trades']}")

        print(f"  Return:      " f"{full_stats['total_return']:.2f}%")

        print(f"  Max DD:      " f"{full_stats['max_drawdown']:.2f}%")

        print(f"  Exposure:    " f"{full_stats['exposure_pct']:.2f}%")

    if not all_results:
        raise RuntimeError("No walk-forward results.")

    results = pd.DataFrame(all_results)

    for symbol in args.symbols:
        print_symbol_summary(
            dataframe=results,
            symbol=symbol,
        )

    print_global_summary(results)

    output_file = DATA_DIR / (
        f"tsmom_v2_"
        f"{args.timeframe}_"
        f"{args.rebalance}_"
        "continuous_walk_forward.csv"
    )

    results.to_csv(
        output_file,
        index=False,
    )

    print()
    print(f"Saved to: " f"{output_file}")


if __name__ == "__main__":
    main()
