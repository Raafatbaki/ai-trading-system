import json
import time
from pathlib import Path

import pandas as pd
import requests

from strategies.time_series_momentum import (
    calculate_features,
    generate_signals,
    get_strategy_periods,
)

# =========================================================
# FROZEN STRATEGY CONFIGURATION
# =========================================================

SYMBOLS = [
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
]

MARKET = "spot"

TIMEFRAME = "1h"
BYBIT_INTERVAL = "60"

REBALANCE = "daily"

INITIAL_CAPITAL = 100.0

FEE_RATE = 0.001
SLIPPAGE_RATE = 0.0

# نحتاج أكثر من 200 يوم لأن الاستراتيجية
# تستخدم MA 200 days.
HISTORY_DAYS = 260

# لكي لا نزعم أننا نفذنا على Open قديم.
# يجب تشغيل البرنامج قريبًا من بداية الساعة.
EXECUTION_GRACE_MINUTES = 10


# =========================================================
# BYBIT
# =========================================================

BASE_URL = "https://api.bybit.com"
KLINE_URL = f"{BASE_URL}/v5/market/kline"


# =========================================================
# FILES
# =========================================================

PAPER_DIR = Path("data/paper")

STATE_FILE = PAPER_DIR / "state.json"
TRADES_FILE = PAPER_DIR / "trades.csv"
EQUITY_FILE = PAPER_DIR / "equity.csv"


TRADE_COLUMNS = [
    "symbol",
    "entry_signal_time",
    "entry_time",
    "entry_price",
    "exit_signal_time",
    "exit_time",
    "exit_price",
    "quantity",
    "entry_fee_usdt",
    "exit_fee_usdt",
    "capital_before_entry",
    "capital_after_trade",
    "net_pnl_usdt",
    "net_return_pct",
]


EQUITY_COLUMNS = [
    "run_time",
    "symbol",
    "market_price",
    "position",
    "cash",
    "quantity",
    "equity",
]


# =========================================================
# HELPERS
# =========================================================


def utc_now() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


def iso_timestamp(
    value,
) -> str:
    return pd.Timestamp(value).isoformat()


def append_csv(
    file_path: Path,
    row: dict,
):
    dataframe = pd.DataFrame([row])

    dataframe.to_csv(
        file_path,
        mode="a",
        header=not file_path.exists(),
        index=False,
    )


def ensure_files():
    PAPER_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not TRADES_FILE.exists():
        pd.DataFrame(columns=TRADE_COLUMNS).to_csv(
            TRADES_FILE,
            index=False,
        )

    if not EQUITY_FILE.exists():
        pd.DataFrame(columns=EQUITY_COLUMNS).to_csv(
            EQUITY_FILE,
            index=False,
        )


# =========================================================
# PAPER STATE
# =========================================================


def create_symbol_state() -> dict:
    return {
        "cash": INITIAL_CAPITAL,
        "quantity": 0.0,
        "position_open": False,
        "entry_signal_time": None,
        "entry_time": None,
        "entry_price": None,
        "entry_fee_usdt": 0.0,
        "capital_before_entry": None,
        "last_processed_signal_time": None,
    }


def create_state() -> dict:
    return {
        "strategy": "TSMOM_V2",
        "exchange": "BYBIT",
        "market": MARKET,
        "timeframe": TIMEFRAME,
        "rebalance": REBALANCE,
        "created_at": iso_timestamp(utc_now()),
        "symbols": {symbol: create_symbol_state() for symbol in SYMBOLS},
    }


def load_state() -> dict:
    if not STATE_FILE.exists():
        return create_state()

    with open(
        STATE_FILE,
        "r",
        encoding="utf-8",
    ) as file:
        state = json.load(file)

    if "symbols" not in state:
        raise RuntimeError(
            "Existing paper state has an "
            "incompatible format. "
            "Remove data/paper/state.json "
            "before starting this version."
        )

    for symbol in SYMBOLS:
        if symbol not in state["symbols"]:
            state["symbols"][symbol] = create_symbol_state()

    return state


def save_state(
    state: dict,
):
    with open(
        STATE_FILE,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            state,
            file,
            indent=2,
        )


# =========================================================
# BYBIT MARKET DATA
# =========================================================


def fetch_bybit_klines(
    symbol: str,
) -> pd.DataFrame:
    now = utc_now()

    start_target = now - pd.Timedelta(days=HISTORY_DAYS)

    start_target_ms = int(start_target.timestamp() * 1000)

    end_ms = int(now.timestamp() * 1000)

    rows = []

    while True:
        response = requests.get(
            KLINE_URL,
            params={
                "category": MARKET,
                "symbol": symbol,
                "interval": BYBIT_INTERVAL,
                "end": end_ms,
                "limit": 1000,
            },
            timeout=30,
        )

        response.raise_for_status()

        payload = response.json()

        if payload.get("retCode") != 0:
            raise RuntimeError("Bybit error: " f"{payload}")

        batch = payload.get(
            "result",
            {},
        ).get(
            "list",
            [],
        )

        if not batch:
            break

        rows.extend(batch)

        oldest_timestamp_ms = min(int(row[0]) for row in batch)

        if oldest_timestamp_ms <= start_target_ms:
            break

        end_ms = oldest_timestamp_ms - 1

        time.sleep(0.05)

    if not rows:
        raise RuntimeError(f"No Bybit candles returned " f"for {symbol}.")

    dataframe = pd.DataFrame(
        rows,
        columns=[
            "timestamp_ms",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "turnover",
        ],
    )

    dataframe["timestamp"] = pd.to_datetime(
        dataframe["timestamp_ms"].astype("int64"),
        unit="ms",
        utc=True,
    )

    for column in [
        "open",
        "high",
        "low",
        "close",
        "volume",
    ]:
        dataframe[column] = pd.to_numeric(
            dataframe[column],
            errors="raise",
        )

    dataframe = (
        dataframe[
            [
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume",
            ]
        ]
        .drop_duplicates(subset=["timestamp"])
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    dataframe = dataframe[dataframe["timestamp"] >= start_target].reset_index(drop=True)

    return dataframe


# =========================================================
# STRATEGY
# =========================================================


def prepare_signals(
    closed_candles: pd.DataFrame,
) -> pd.DataFrame:
    (
        momentum_lookback,
        trend_window,
    ) = get_strategy_periods(TIMEFRAME)

    dataframe = calculate_features(
        dataframe=closed_candles.copy(),
        momentum_lookback=momentum_lookback,
        long_trend_window=trend_window,
        rebalance=REBALANCE,
    )

    required_history = max(
        momentum_lookback,
        trend_window,
    )

    if len(dataframe) <= required_history:
        raise RuntimeError("Not enough candles for " "TSMOM V2 warm-up.")

    dataframe = dataframe.iloc[required_history:].copy()

    dataframe = dataframe.reset_index(drop=True)

    dataframe = generate_signals(dataframe)

    return dataframe


# =========================================================
# PAPER PORTFOLIO
# =========================================================


def calculate_equity(
    symbol_state: dict,
    market_price: float,
) -> float:
    if symbol_state["position_open"]:
        return float(symbol_state["quantity"]) * market_price

    return float(symbol_state["cash"])


def paper_buy(
    symbol: str,
    symbol_state: dict,
    signal_time,
    execution_time,
    execution_price: float,
):
    cash = float(symbol_state["cash"])

    capital_before_entry = cash

    price = execution_price * (1 + SLIPPAGE_RATE)

    entry_fee = cash * FEE_RATE

    investable_cash = cash - entry_fee

    quantity = investable_cash / price

    symbol_state["cash"] = 0.0

    symbol_state["quantity"] = quantity

    symbol_state["position_open"] = True

    symbol_state["entry_signal_time"] = iso_timestamp(signal_time)

    symbol_state["entry_time"] = iso_timestamp(execution_time)

    symbol_state["entry_price"] = price

    symbol_state["entry_fee_usdt"] = entry_fee

    symbol_state["capital_before_entry"] = capital_before_entry

    print()
    print("PAPER BUY")
    print(f"Quantity:       " f"{quantity:.8f}")
    print(f"Entry price:    " f"{price:.2f}")
    print(f"Entry fee:      " f"{entry_fee:.4f} USDT")


def paper_exit(
    symbol: str,
    symbol_state: dict,
    signal_time,
    execution_time,
    execution_price: float,
):
    quantity = float(symbol_state["quantity"])

    price = execution_price * (1 - SLIPPAGE_RATE)

    gross_value = quantity * price

    exit_fee = gross_value * FEE_RATE

    capital_after_trade = gross_value - exit_fee

    capital_before_entry = float(symbol_state["capital_before_entry"])

    net_pnl = capital_after_trade - capital_before_entry

    net_return_pct = net_pnl / capital_before_entry * 100

    trade = {
        "symbol": symbol,
        "entry_signal_time": symbol_state["entry_signal_time"],
        "entry_time": symbol_state["entry_time"],
        "entry_price": symbol_state["entry_price"],
        "exit_signal_time": iso_timestamp(signal_time),
        "exit_time": iso_timestamp(execution_time),
        "exit_price": price,
        "quantity": quantity,
        "entry_fee_usdt": symbol_state["entry_fee_usdt"],
        "exit_fee_usdt": exit_fee,
        "capital_before_entry": capital_before_entry,
        "capital_after_trade": capital_after_trade,
        "net_pnl_usdt": net_pnl,
        "net_return_pct": net_return_pct,
    }

    append_csv(
        TRADES_FILE,
        trade,
    )

    symbol_state["cash"] = capital_after_trade

    symbol_state["quantity"] = 0.0

    symbol_state["position_open"] = False

    symbol_state["entry_signal_time"] = None

    symbol_state["entry_time"] = None

    symbol_state["entry_price"] = None

    symbol_state["entry_fee_usdt"] = 0.0

    symbol_state["capital_before_entry"] = None

    print()
    print("PAPER EXIT")
    print(f"Exit price:     " f"{price:.2f}")
    print(f"Exit fee:       " f"{exit_fee:.4f} USDT")
    print(f"Net P/L:        " f"{net_pnl:.2f} USDT")
    print(f"Net return:     " f"{net_return_pct:.2f}%")


# =========================================================
# SYMBOL PROCESSING
# =========================================================


def process_symbol(
    symbol: str,
    state: dict,
):
    print()
    print("=" * 75)
    print(symbol)
    print("=" * 75)

    market_data = fetch_bybit_klines(symbol)

    now = utc_now()

    current_hour = now.floor("h")

    closed_candles = market_data[market_data["timestamp"] < current_hour].copy()

    current_candle = market_data[market_data["timestamp"] == current_hour].copy()

    if closed_candles.empty:
        raise RuntimeError(f"No closed candles for " f"{symbol}.")

    signals = prepare_signals(closed_candles)

    latest_signal_row = signals.iloc[-1]

    latest_signal_time = latest_signal_row["timestamp"]

    signal = (
        str(
            latest_signal_row.get(
                "signal",
                "NO SIGNAL",
            )
        )
        .strip()
        .upper()
    )

    symbol_state = state["symbols"][symbol]

    # Current market price for
    # mark-to-market equity.
    if not current_candle.empty:
        market_price = float(current_candle.iloc[-1]["close"])
    else:
        market_price = float(closed_candles.iloc[-1]["close"])

    print(f"Closed candles: " f"{len(closed_candles)}")

    print(f"Latest closed:  " f"{latest_signal_time}")

    print(f"Signal:         " f"{signal}")

    print(f"Position:       " f"{'LONG' if symbol_state['position_open'] else 'CASH'}")

    print(f"Market price:   " f"{market_price:.2f}")

    # ---------------------------------------------
    # Only BUY / EXIT require an execution.
    # ---------------------------------------------

    if signal in [
        "BUY",
        "EXIT",
    ]:

        signal_key = iso_timestamp(latest_signal_time)

        if symbol_state["last_processed_signal_time"] == signal_key:
            print("Action:         " "ALREADY PROCESSED")

        elif current_candle.empty:
            print("Action:         " "WAIT - current 1h " "candle unavailable")

        else:
            execution_row = current_candle.iloc[-1]

            execution_time = execution_row["timestamp"]

            expected_execution_time = latest_signal_time + pd.Timedelta(hours=1)

            # يجب أن يكون التنفيذ فعلًا
            # على الـnext candle.
            if execution_time != expected_execution_time:
                print("Action:         " "SKIPPED - execution " "candle mismatch")

            else:
                minutes_after_open = (now - execution_time).total_seconds() / 60

                if minutes_after_open > EXECUTION_GRACE_MINUTES:
                    print("Action:         " "MISSED")

                    print("Reason:         " "next-candle open " "is already too old")

                    symbol_state["last_processed_signal_time"] = signal_key

                else:
                    execution_price = float(execution_row["open"])

                    print(f"Execution time: " f"{execution_time}")

                    print(f"Execution open: " f"{execution_price:.2f}")

                    if signal == "BUY" and not symbol_state["position_open"]:
                        paper_buy(
                            symbol=symbol,
                            symbol_state=symbol_state,
                            signal_time=latest_signal_time,
                            execution_time=execution_time,
                            execution_price=execution_price,
                        )

                    elif signal == "EXIT" and symbol_state["position_open"]:
                        paper_exit(
                            symbol=symbol,
                            symbol_state=symbol_state,
                            signal_time=latest_signal_time,
                            execution_time=execution_time,
                            execution_price=execution_price,
                        )

                    else:
                        print("Action:         " "NO POSITION CHANGE")

                    symbol_state["last_processed_signal_time"] = signal_key

    else:
        print("Action:         NONE")

    # ---------------------------------------------
    # Mark-to-market equity snapshot
    # ---------------------------------------------

    equity = calculate_equity(
        symbol_state,
        market_price,
    )

    append_csv(
        EQUITY_FILE,
        {
            "run_time": iso_timestamp(now),
            "symbol": symbol,
            "market_price": market_price,
            "position": ("LONG" if symbol_state["position_open"] else "CASH"),
            "cash": symbol_state["cash"],
            "quantity": symbol_state["quantity"],
            "equity": equity,
        },
    )

    print(f"Paper equity:   " f"{equity:.2f} USDT")


# =========================================================
# MAIN
# =========================================================


def main():
    ensure_files()

    state = load_state()

    print()
    print("=" * 75)
    print("TSMOM V2 - BYBIT PAPER TRADING")
    print("=" * 75)

    print()
    print("Exchange:       BYBIT")
    print("Market:         SPOT")
    print("Real orders:    DISABLED")
    print("API key:        NOT REQUIRED")
    print()

    print("Symbols:        " + ", ".join(SYMBOLS))

    print(f"Timeframe:      " f"{TIMEFRAME}")

    print(f"Rebalance:      " f"{REBALANCE.upper()}")

    print("Momentum:       " "28 days")

    print("Trend filter:   " "200 days")

    print(f"Starting paper capital: " f"{INITIAL_CAPITAL:.2f} USDT " "per symbol")

    for symbol in SYMBOLS:
        try:
            process_symbol(
                symbol=symbol,
                state=state,
            )

            save_state(state)

        except Exception as error:
            print()
            print(f"ERROR {symbol}: " f"{error}")

    save_state(state)

    print()
    print("=" * 75)

    print(f"State:  " f"{STATE_FILE}")

    print(f"Trades: " f"{TRADES_FILE}")

    print(f"Equity: " f"{EQUITY_FILE}")

    print("=" * 75)


if __name__ == "__main__":
    main()
