import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

MARKET = "spot"
STRATEGY = "tsmom_v2"
INITIAL_CAPITAL = 100.0
FEE_RATE = 0.001

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPLITS_DIRECTORY = PROJECT_ROOT / "data" / "splits"

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

DATASETS = [
    "development",
    "validation",
]


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Run TSMOM V2 research across "
            "multiple timeframes and rebalance frequencies."
        )
    )

    parser.add_argument(
        "--symbol",
        choices=SUPPORTED_SYMBOLS,
        default="BTCUSDT",
    )

    parser.add_argument(
        "--timeframes",
        nargs="+",
        choices=SUPPORTED_TIMEFRAMES,
        default=SUPPORTED_TIMEFRAMES,
    )

    parser.add_argument(
        "--rebalances",
        nargs="+",
        choices=SUPPORTED_REBALANCES,
        default=SUPPORTED_REBALANCES,
    )

    return parser.parse_args()


def run_command(command: list[str]) -> None:
    print()
    print("=" * 90)
    print("RUNNING")
    print(" ".join(command))
    print("=" * 90)

    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
    )

    if result.returncode != 0:
        raise RuntimeError("Command failed:\n" + " ".join(command))


def generate_signals(
    symbol: str,
    timeframe: str,
    dataset: str,
    rebalance: str,
) -> None:
    command = [
        sys.executable,
        "src/strategies/time_series_momentum.py",
        "--symbol",
        symbol,
        "--timeframe",
        timeframe,
        "--dataset",
        dataset,
        "--rebalance",
        rebalance,
    ]

    run_command(command)


def run_backtest(
    symbol: str,
    timeframe: str,
    dataset: str,
    rebalance: str,
) -> None:
    command = [
        sys.executable,
        "src/backtest.py",
        "--strategy",
        STRATEGY,
        "--symbol",
        symbol,
        "--timeframe",
        timeframe,
        "--dataset",
        dataset,
        "--rebalance",
        rebalance,
    ]

    run_command(command)


def get_base_name(
    symbol: str,
    timeframe: str,
    dataset: str,
    rebalance: str,
) -> str:
    return f"{symbol}_{MARKET}_{timeframe}_" f"{dataset}_{STRATEGY}_{rebalance}"


def calculate_buy_and_hold(
    signals: pd.DataFrame,
) -> float:
    first_price = float(signals.iloc[0]["open"])

    last_price = float(signals.iloc[-1]["close"])

    buy_fee = INITIAL_CAPITAL * FEE_RATE

    investable_capital = INITIAL_CAPITAL - buy_fee

    quantity = investable_capital / first_price

    gross_final_value = quantity * last_price

    exit_fee = gross_final_value * FEE_RATE

    final_value = gross_final_value - exit_fee

    return (final_value / INITIAL_CAPITAL - 1) * 100


def calculate_summary(
    symbol: str,
    timeframe: str,
    dataset: str,
    rebalance: str,
) -> dict:
    base = get_base_name(
        symbol=symbol,
        timeframe=timeframe,
        dataset=dataset,
        rebalance=rebalance,
    )

    signals_file = SPLITS_DIRECTORY / f"{base}_signals.csv"

    trades_file = SPLITS_DIRECTORY / f"{base}_trades.csv"

    equity_file = SPLITS_DIRECTORY / f"{base}_equity.csv"

    for file_path in [
        signals_file,
        trades_file,
        equity_file,
    ]:
        if not file_path.exists():
            raise FileNotFoundError(f"Missing file: {file_path}")

    signals = pd.read_csv(signals_file)

    trades = pd.read_csv(trades_file)

    equity = pd.read_csv(equity_file)

    trade_count = len(trades)

    if trade_count > 0:
        wins = int((trades["pnl_usdt"] > 0).sum())

        losses = int((trades["pnl_usdt"] < 0).sum())

        win_rate = wins / trade_count * 100

        final_capital = float(trades.iloc[-1]["capital_after_trade"])

        strategy_return = (final_capital / INITIAL_CAPITAL - 1) * 100

        winning_trades = trades[trades["pnl_usdt"] > 0]

        losing_trades = trades[trades["pnl_usdt"] < 0]

        gross_profit = float(winning_trades["pnl_usdt"].sum())

        gross_loss = abs(float(losing_trades["pnl_usdt"].sum()))

        if gross_loss > 0:
            profit_factor = gross_profit / gross_loss
        elif gross_profit > 0:
            profit_factor = float("inf")
        else:
            profit_factor = 0.0

        average_trade = float(trades["net_return_pct"].mean())

        worst_trade = float(trades["net_return_pct"].min())

        fees = INITIAL_CAPITAL - float(
            (
                trades["pnl_usdt"]
                + trades.get(
                    "gross_pnl_usdt",
                    trades["pnl_usdt"],
                )
            ).sum()
        )

    else:
        wins = 0
        losses = 0
        win_rate = 0.0
        final_capital = INITIAL_CAPITAL
        strategy_return = 0.0
        profit_factor = 0.0
        average_trade = 0.0
        worst_trade = 0.0

    max_drawdown = float(equity["drawdown_pct"].min())

    if "position_open" in equity.columns:
        exposure = equity["position_open"].astype(bool).mean() * 100
    else:
        exposure = float("nan")

    buy_hold = calculate_buy_and_hold(signals)

    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "rebalance": rebalance,
        "dataset": dataset,
        "trades": trade_count,
        "wins": wins,
        "losses": losses,
        "win_rate_pct": win_rate,
        "return_pct": strategy_return,
        "buy_hold_pct": buy_hold,
        "profit_factor": profit_factor,
        "avg_trade_pct": average_trade,
        "worst_trade_pct": worst_trade,
        "max_dd_pct": max_drawdown,
        "exposure_pct": exposure,
    }


def print_summary(
    results: list[dict],
    symbol: str,
) -> None:
    dataframe = pd.DataFrame(results)

    numeric_columns = [
        "win_rate_pct",
        "return_pct",
        "buy_hold_pct",
        "profit_factor",
        "avg_trade_pct",
        "worst_trade_pct",
        "max_dd_pct",
        "exposure_pct",
    ]

    dataframe[numeric_columns] = dataframe[numeric_columns].round(2)

    print()
    print("=" * 120)
    print(f"{symbol} - TSMOM V2 " "MULTI-TIMEFRAME RESEARCH")
    print("=" * 120)

    print()
    print(dataframe.to_string(index=False))

    output_file = SPLITS_DIRECTORY / (f"{symbol}_tsmom_v2_" f"multiframe_research.csv")

    dataframe.to_csv(
        output_file,
        index=False,
    )

    print()
    print("=" * 120)
    print(f"Summary saved to: " f"{output_file}")


def main():
    args = parse_arguments()

    symbol = args.symbol
    timeframes = args.timeframes
    rebalances = args.rebalances

    results = []

    print()
    print("=" * 90)
    print("TSMOM V2 MULTI-TIMEFRAME RESEARCH")
    print("=" * 90)

    print(f"Symbol:       {symbol}")
    print("Timeframes:   " + ", ".join(timeframes))
    print("Rebalances:   " + ", ".join(rebalances))
    print("Datasets:     development, validation")
    print("HOLDOUT:      NOT USED")

    for timeframe in timeframes:
        for rebalance in rebalances:
            for dataset in DATASETS:
                generate_signals(
                    symbol=symbol,
                    timeframe=timeframe,
                    dataset=dataset,
                    rebalance=rebalance,
                )

                run_backtest(
                    symbol=symbol,
                    timeframe=timeframe,
                    dataset=dataset,
                    rebalance=rebalance,
                )

                result = calculate_summary(
                    symbol=symbol,
                    timeframe=timeframe,
                    dataset=dataset,
                    rebalance=rebalance,
                )

                results.append(result)

    print_summary(
        results=results,
        symbol=symbol,
    )


if __name__ == "__main__":
    main()
