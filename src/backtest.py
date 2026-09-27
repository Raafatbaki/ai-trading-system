import argparse
from pathlib import Path

import pandas as pd

MARKET = "spot"

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

SUPPORTED_STRATEGIES = [
    "atr_breakout",
    "trend",
    "breakout",
    "donchian",
    "bollinger",
    "rsi_mr",
    "trend_pullback",
    "tsmom",
    "tsmom_v2",
]

SUPPORTED_DATASETS = [
    "development",
    "validation",
    "historical_holdout",
]

INITIAL_CAPITAL = 100.0

# 0.1% per transaction side.
FEE_RATE = 0.001

# Currently disabled.
SLIPPAGE_RATE = 0.0


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Generic long-only strategy backtester."
    )

    parser.add_argument(
        "--symbol",
        choices=SUPPORTED_SYMBOLS,
        default="BTCUSDT",
    )

    parser.add_argument(
        "--timeframe",
        choices=SUPPORTED_TIMEFRAMES,
        default="1h",
    )

    parser.add_argument(
        "--strategy",
        choices=SUPPORTED_STRATEGIES,
        required=True,
    )

    parser.add_argument(
        "--dataset",
        choices=SUPPORTED_DATASETS,
        default="development",
    )

    parser.add_argument(
        "--rebalance",
        choices=["weekly", "daily"],
        default="weekly",
    )

    return parser.parse_args()


def get_files(
    symbol: str,
    timeframe: str,
    strategy: str,
    dataset: str,
    rebalance: str,
) -> tuple[Path, Path, Path]:
    if strategy == "trend":
        signals_file = Path(f"data/{symbol}_{MARKET}_{timeframe}_trend_signals.csv")

        trades_file = Path(f"data/{symbol}_{MARKET}_{timeframe}_trades.csv")

        equity_file = Path(f"data/{symbol}_{MARKET}_{timeframe}_equity.csv")

    elif strategy == "breakout":
        signals_file = Path(f"data/{symbol}_{MARKET}_{timeframe}_breakout_signals.csv")

        trades_file = Path(f"data/{symbol}_{MARKET}_{timeframe}_breakout_trades.csv")

        equity_file = Path(f"data/{symbol}_{MARKET}_{timeframe}_breakout_equity.csv")

    elif strategy == "donchian":
        base = f"{symbol}_{MARKET}_{timeframe}_" f"{dataset}_donchian_20_10"

        signals_file = Path(f"data/splits/{base}_signals.csv")

        trades_file = Path(f"data/splits/{base}_trades.csv")

        equity_file = Path(f"data/splits/{base}_equity.csv")

    elif strategy == "tsmom":
        base = f"{symbol}_{MARKET}_{timeframe}_{dataset}_tsmom"

        signals_file = Path(f"data/splits/{base}_signals.csv")

        trades_file = Path(f"data/splits/{base}_trades.csv")

        equity_file = Path(f"data/splits/{base}_equity.csv")

    elif strategy == "bollinger":
        base = f"{symbol}_{MARKET}_{timeframe}_" f"{dataset}_bollinger_mr"

        signals_file = Path(f"data/splits/{base}_signals.csv")

        trades_file = Path(f"data/splits/{base}_trades.csv")

        equity_file = Path(f"data/splits/{base}_equity.csv")
    elif strategy == "rsi_mr":
        base = f"{symbol}_{MARKET}_{timeframe}_" f"{dataset}_rsi_mr"

        signals_file = Path(f"data/splits/{base}_signals.csv")

        trades_file = Path(f"data/splits/{base}_trades.csv")

        equity_file = Path(f"data/splits/{base}_equity.csv")

    elif strategy == "atr_breakout":
        base = (
            f"{symbol}_{MARKET}_{timeframe}_"
            f"{dataset}_atr_breakout"
        )

        signals_file = Path(
            f"data/splits/{base}_signals.csv"
        )

        trades_file = Path(
            f"data/splits/{base}_trades.csv"
        )

        equity_file = Path(
            f"data/splits/{base}_equity.csv"
        )
    
    elif strategy == "trend_pullback":
        base = (
            f"{symbol}_{MARKET}_{timeframe}_"
            f"{dataset}_trend_pullback"
        )

        signals_file = Path(
            f"data/splits/{base}_signals.csv"
        )

        trades_file = Path(
            f"data/splits/{base}_trades.csv"
        )

        equity_file = Path(
            f"data/splits/{base}_equity.csv"
        )
    
    elif strategy == "tsmom_v2":
        base = f"{symbol}_{MARKET}_{timeframe}_" f"{dataset}_tsmom_v2_{rebalance}"

        signals_file = Path(f"data/splits/{base}_signals.csv")

        trades_file = Path(f"data/splits/{base}_trades.csv")

        equity_file = Path(f"data/splits/{base}_equity.csv")

    else:
        raise ValueError(f"Unsupported strategy: {strategy}")

    return (
        signals_file,
        trades_file,
        equity_file,
    )


def safe_mean(
    series: pd.Series,
) -> float:
    if series.empty:
        return float("nan")

    return float(series.mean())


def calculate_drawdown(
    equity_dataframe: pd.DataFrame,
) -> pd.DataFrame:
    dataframe = equity_dataframe.copy()

    dataframe["equity_peak"] = dataframe["equity"].cummax()

    dataframe["drawdown_pct"] = (
        (dataframe["equity"] / dataframe["equity_peak"]) - 1
    ) * 100

    return dataframe


def close_trade(
    *,
    asset_quantity: float,
    entry_signal_time,
    entry_time,
    entry_price: float,
    entry_capital: float,
    exit_signal_time,
    exit_time,
    exit_price: float,
    forced_exit: bool,
) -> tuple[dict, float, float]:
    gross_exit_value = asset_quantity * exit_price

    exit_fee = gross_exit_value * FEE_RATE

    capital_after_trade = gross_exit_value - exit_fee

    gross_return = exit_price / entry_price - 1

    net_return = capital_after_trade / entry_capital - 1

    pnl = capital_after_trade - entry_capital

    duration = pd.Timestamp(exit_time) - pd.Timestamp(entry_time)

    trade = {
        "entry_signal_time": entry_signal_time,
        "entry_time": entry_time,
        "entry_price": entry_price,
        "exit_signal_time": exit_signal_time,
        "exit_time": exit_time,
        "exit_price": exit_price,
        "duration": duration,
        "gross_return_pct": gross_return * 100,
        "net_return_pct": net_return * 100,
        "pnl_usdt": pnl,
        "capital_after_trade": capital_after_trade,
        "forced_exit": forced_exit,
    }

    return (
        trade,
        capital_after_trade,
        exit_fee,
    )


def run_backtest(
    dataframe: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict,
]:
    if dataframe.empty:
        raise RuntimeError("Signal dataframe is empty.")

    required_columns = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "signal",
    ]

    missing_columns = [
        column for column in required_columns if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError("Missing required columns: " f"{missing_columns}")

    dataframe = dataframe.sort_values("timestamp").reset_index(drop=True)

    capital = INITIAL_CAPITAL

    position_open = False
    asset_quantity = 0.0

    entry_signal_time = None
    entry_time = None
    entry_price = None
    entry_capital = None

    total_fees = 0.0
    exposure_candles = 0

    trades = []
    equity_rows = []

    for index in range(len(dataframe)):
        current = dataframe.iloc[index]

        # ----------------------------------
        # MARK-TO-MARKET EQUITY
        # ----------------------------------

        if position_open:
            exposure_candles += 1

            current_equity = asset_quantity * float(current["close"])

        else:
            current_equity = capital

        equity_rows.append(
            {
                "timestamp": current["timestamp"],
                "close": float(current["close"]),
                "equity": current_equity,
                "position_open": position_open,
            }
        )

        # Last candle has no next open.
        if index == (len(dataframe) - 1):
            continue

        signal = str(current["signal"])

        next_candle = dataframe.iloc[index + 1]

        # ----------------------------------
        # BUY AT NEXT CANDLE OPEN
        # ----------------------------------

        if signal == "BUY" and not position_open:
            raw_entry_price = float(next_candle["open"])

            entry_price = raw_entry_price * (1 + SLIPPAGE_RATE)

            entry_signal_time = current["timestamp"]

            entry_time = next_candle["timestamp"]

            entry_capital = capital

            entry_fee = entry_capital * FEE_RATE

            investable_capital = entry_capital - entry_fee

            asset_quantity = investable_capital / entry_price

            total_fees += entry_fee

            position_open = True

        # ----------------------------------
        # EXIT AT NEXT CANDLE OPEN
        # ----------------------------------

        elif signal == "EXIT" and position_open:
            raw_exit_price = float(next_candle["open"])

            exit_price = raw_exit_price * (1 - SLIPPAGE_RATE)

            trade, capital, exit_fee = close_trade(
                asset_quantity=(asset_quantity),
                entry_signal_time=(entry_signal_time),
                entry_time=entry_time,
                entry_price=entry_price,
                entry_capital=(entry_capital),
                exit_signal_time=(current["timestamp"]),
                exit_time=(next_candle["timestamp"]),
                exit_price=(exit_price),
                forced_exit=False,
            )

            trades.append(trade)

            total_fees += exit_fee

            position_open = False
            asset_quantity = 0.0

            entry_signal_time = None
            entry_time = None
            entry_price = None
            entry_capital = None

    # --------------------------------------
    # FORCE CLOSE FINAL OPEN POSITION
    # --------------------------------------

    if position_open:
        last_candle = dataframe.iloc[-1]

        exit_price = float(last_candle["close"]) * (1 - SLIPPAGE_RATE)

        trade, capital, exit_fee = close_trade(
            asset_quantity=(asset_quantity),
            entry_signal_time=(entry_signal_time),
            entry_time=entry_time,
            entry_price=entry_price,
            entry_capital=(entry_capital),
            exit_signal_time=None,
            exit_time=(last_candle["timestamp"]),
            exit_price=exit_price,
            forced_exit=True,
        )

        trades.append(trade)

        total_fees += exit_fee

        # Include the exit fee in final
        # mark-to-market equity.
        equity_rows[-1]["equity"] = capital

    trades_dataframe = pd.DataFrame(trades)

    equity_dataframe = pd.DataFrame(equity_rows)

    equity_dataframe = calculate_drawdown(equity_dataframe)

    # --------------------------------------
    # STATISTICS
    # --------------------------------------

    if trades_dataframe.empty:
        winning_trades = trades_dataframe

        losing_trades = trades_dataframe

        profit_factor = float("nan")

        average_trade = float("nan")

        average_win = float("nan")

        average_loss = float("nan")

        best_trade = float("nan")

        worst_trade = float("nan")

        forced_exits = 0

    else:
        winning_trades = trades_dataframe[trades_dataframe["pnl_usdt"] > 0]

        losing_trades = trades_dataframe[trades_dataframe["pnl_usdt"] < 0]

        gross_profit = float(winning_trades["pnl_usdt"].sum())

        gross_loss = abs(float(losing_trades["pnl_usdt"].sum()))

        if gross_loss > 0:
            profit_factor = gross_profit / gross_loss
        elif gross_profit > 0:
            profit_factor = float("inf")
        else:
            profit_factor = 0.0

        average_trade = safe_mean(trades_dataframe["net_return_pct"])

        average_win = safe_mean(winning_trades["net_return_pct"])

        average_loss = safe_mean(losing_trades["net_return_pct"])

        best_trade = float(trades_dataframe["net_return_pct"].max())

        worst_trade = float(trades_dataframe["net_return_pct"].min())

        forced_exits = int(trades_dataframe["forced_exit"].sum())

    completed_trades = len(trades_dataframe)

    wins = len(winning_trades)

    losses = len(losing_trades)

    if completed_trades > 0:
        win_rate = wins / completed_trades * 100
    else:
        win_rate = 0.0

    total_return = (capital / INITIAL_CAPITAL - 1) * 100

    max_drawdown = float(equity_dataframe["drawdown_pct"].min())

    exposure_pct = exposure_candles / len(dataframe) * 100

    statistics = {
        "trades": completed_trades,
        "wins": wins,
        "losses": losses,
        "win_rate": win_rate,
        "initial_capital": INITIAL_CAPITAL,
        "final_capital": capital,
        "total_return": total_return,
        "profit_factor": profit_factor,
        "average_trade": average_trade,
        "average_win": average_win,
        "average_loss": average_loss,
        "best_trade": best_trade,
        "worst_trade": worst_trade,
        "max_drawdown": max_drawdown,
        "total_fees": total_fees,
        "forced_exits": forced_exits,
        "exposure_pct": exposure_pct,
    }

    return (
        trades_dataframe,
        equity_dataframe,
        statistics,
    )


def calculate_buy_and_hold(
    dataframe: pd.DataFrame,
) -> float:
    first_price = float(dataframe.iloc[0]["open"])

    last_price = float(dataframe.iloc[-1]["close"])

    buy_fee = INITIAL_CAPITAL * FEE_RATE

    investable_capital = INITIAL_CAPITAL - buy_fee

    quantity = investable_capital / first_price

    value_before_exit_fee = quantity * last_price

    exit_fee = value_before_exit_fee * FEE_RATE

    final_value = value_before_exit_fee - exit_fee

    return (final_value / INITIAL_CAPITAL - 1) * 100


def format_number(
    value: float,
    decimals: int = 2,
) -> str:
    if pd.isna(value):
        return "N/A"

    if value == float("inf"):
        return "INF"

    return f"{value:.{decimals}f}"


def main():
    args = parse_arguments()

    symbol = args.symbol
    timeframe = args.timeframe
    strategy = args.strategy
    dataset = args.dataset
    rebalance = args.rebalance
    (
        signals_file,
        trades_file,
        equity_file,
    ) = get_files(
        symbol=symbol,
        timeframe=timeframe,
        strategy=strategy,
        dataset=dataset,
        rebalance=rebalance,
    )

    if not signals_file.exists():
        raise FileNotFoundError(f"Signal file not found: " f"{signals_file}")

    dataframe = pd.read_csv(
        signals_file,
        parse_dates=["timestamp"],
    )

    (
        trades,
        equity,
        stats,
    ) = run_backtest(dataframe)

    buy_and_hold_return = calculate_buy_and_hold(dataframe)

    trades_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    trades.to_csv(
        trades_file,
        index=False,
    )

    equity.to_csv(
        equity_file,
        index=False,
    )

    print()
    print("=" * 65)
    print("STRATEGY BACKTEST")
    print("=" * 65)

    print()
    print(f"Strategy:     " f"{strategy.upper()}")
    print(f"Symbol:       " f"{symbol}")
    print(f"Market:       " f"{MARKET.upper()}")
    print(f"Timeframe:    " f"{timeframe}")

    if strategy in [
        "atr_breakout",
        "trend_pullback",
        "rsi_mr",
        "donchian",
        "bollinger",
        "tsmom",
        "tsmom_v2",
    ]:
        print(f"Dataset:      " f"{dataset.upper()}")

    print("Mode:         LONG ONLY")
    print("Short:        DISABLED")
    print("Leverage:     DISABLED")

    print()
    print("-" * 65)
    print("TRADES")
    print("-" * 65)

    print(f"Completed trades: " f"{stats['trades']}")
    print(f"Winning trades:   " f"{stats['wins']}")
    print(f"Losing trades:    " f"{stats['losses']}")
    print(f"Win rate:         " f"{stats['win_rate']:.2f}%")
    print(f"Forced exits:     " f"{stats['forced_exits']}")

    print()
    print("-" * 65)
    print("PERFORMANCE")
    print("-" * 65)

    print(f"Initial capital:  " f"{stats['initial_capital']:.2f} USDT")
    print(f"Final capital:    " f"{stats['final_capital']:.2f} USDT")
    print(f"Net return:       " f"{stats['total_return']:.2f}%")
    print(f"Buy & Hold:       " f"{buy_and_hold_return:.2f}%")
    print(f"Profit factor:    " f"{format_number(stats['profit_factor'])}")

    print()
    print("-" * 65)
    print("TRADE QUALITY")
    print("-" * 65)

    print(f"Average trade:    " f"{format_number(stats['average_trade'], 3)}%")
    print(f"Average win:      " f"{format_number(stats['average_win'], 3)}%")
    print(f"Average loss:     " f"{format_number(stats['average_loss'], 3)}%")
    print(f"Best trade:       " f"{format_number(stats['best_trade'], 3)}%")
    print(f"Worst trade:      " f"{format_number(stats['worst_trade'], 3)}%")

    print()
    print("-" * 65)
    print("RISK & EXPOSURE")
    print("-" * 65)

    print(f"Max drawdown:     " f"{stats['max_drawdown']:.2f}%")
    print(f"Market exposure:  " f"{stats['exposure_pct']:.2f}%")
    print(f"Trading fees:     " f"{stats['total_fees']:.2f} USDT")

    print()
    print("Drawdown uses mark-to-market " "equity on every candle close.")

    print()
    print(f"Trades saved to: " f"{trades_file}")
    print(f"Equity saved to: " f"{equity_file}")


if __name__ == "__main__":
    main()
