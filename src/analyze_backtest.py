import argparse
from pathlib import Path

import pandas as pd

SYMBOL = "BTCUSDT"
MARKET = "spot"

SUPPORTED_TIMEFRAMES = [
    "15m",
    "1h",
    "4h",
]


def parse_arguments():
    parser = argparse.ArgumentParser(description="Analyze backtest results by year.")

    parser.add_argument(
        "--timeframe",
        choices=SUPPORTED_TIMEFRAMES,
        default="1h",
        help="Timeframe to analyze.",
    )

    return parser.parse_args()


def calculate_profit_factor(
    trades: pd.DataFrame,
) -> float:
    winners = trades[trades["pnl_usdt"] > 0]

    losers = trades[trades["pnl_usdt"] < 0]

    gross_profit = winners["pnl_usdt"].sum()

    gross_loss = abs(losers["pnl_usdt"].sum())

    if gross_loss == 0:
        return float("inf")

    return gross_profit / gross_loss


def calculate_compounded_return(
    trades: pd.DataFrame,
) -> float:
    capital = 1.0

    for net_return in trades["net_return_pct"]:
        capital *= 1 + net_return / 100

    return (capital - 1) * 100


def calculate_max_drawdown(
    trades: pd.DataFrame,
) -> float:
    capital = 1.0
    equity_curve = [capital]

    for net_return in trades["net_return_pct"]:
        capital *= 1 + net_return / 100

        equity_curve.append(capital)

    equity = pd.Series(equity_curve)

    running_max = equity.cummax()

    drawdown = equity / running_max - 1

    return drawdown.min() * 100


def analyze_year(
    year: int,
    trades: pd.DataFrame,
) -> dict:
    winners = trades[trades["pnl_usdt"] > 0]

    losers = trades[trades["pnl_usdt"] < 0]

    total_trades = len(trades)

    win_rate = len(winners) / total_trades * 100 if total_trades > 0 else 0

    return {
        "year": year,
        "trades": total_trades,
        "wins": len(winners),
        "losses": len(losers),
        "win_rate": win_rate,
        "return": (calculate_compounded_return(trades)),
        "profit_factor": (calculate_profit_factor(trades)),
        "max_drawdown": (calculate_max_drawdown(trades)),
        "average_trade": trades["net_return_pct"].mean(),
        "best_trade": trades["net_return_pct"].max(),
        "worst_trade": trades["net_return_pct"].min(),
    }


def main():
    args = parse_arguments()

    timeframe = args.timeframe

    trades_file = Path(f"data/{SYMBOL}_{MARKET}_" f"{timeframe}_trades.csv")

    if not trades_file.exists():
        raise FileNotFoundError(f"Trades file not found: " f"{trades_file}")

    trades = pd.read_csv(
        trades_file,
        parse_dates=[
            "entry_signal_time",
            "entry_time",
            "exit_time",
        ],
    )

    # We attribute a trade to the year
    # in which it was closed.
    trades["year"] = trades["exit_time"].dt.year

    results = []

    for year in sorted(trades["year"].unique()):
        year_trades = trades[trades["year"] == year].copy()

        results.append(
            analyze_year(
                int(year),
                year_trades,
            )
        )

    print()
    print("=" * 82)
    print(f"{SYMBOL} {MARKET.upper()} " f"{timeframe} - YEARLY PERFORMANCE")
    print("=" * 82)
    print()

    print(
        f"{'Year':<6}"
        f"{'Trades':>8}"
        f"{'Win %':>10}"
        f"{'Return %':>12}"
        f"{'PF':>10}"
        f"{'Max DD %':>12}"
        f"{'Avg %':>10}"
    )

    print("-" * 82)

    for result in results:
        print(
            f"{result['year']:<6}"
            f"{result['trades']:>8}"
            f"{result['win_rate']:>10.2f}"
            f"{result['return']:>12.2f}"
            f"{result['profit_factor']:>10.2f}"
            f"{result['max_drawdown']:>12.2f}"
            f"{result['average_trade']:>10.3f}"
        )

    print()
    print("DETAILS")
    print("-" * 82)

    for result in results:
        print()
        print(f"Year: {result['year']}")
        print(f"Trades:        " f"{result['trades']}")
        print(f"Wins / Losses: " f"{result['wins']} / " f"{result['losses']}")
        print(f"Win rate:      " f"{result['win_rate']:.2f}%")
        print(f"Return:        " f"{result['return']:.2f}%")
        print(f"Profit factor: " f"{result['profit_factor']:.2f}")
        print(f"Max drawdown:  " f"{result['max_drawdown']:.2f}%")
        print(f"Average trade: " f"{result['average_trade']:.3f}%")
        print(f"Best trade:    " f"{result['best_trade']:.3f}%")
        print(f"Worst trade:   " f"{result['worst_trade']:.3f}%")

    print()
    print("Note: trades are assigned to the year " "in which they EXIT.")


if __name__ == "__main__":
    main()
