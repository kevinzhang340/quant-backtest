import pandas as pd
import numpy as np

def run_backtest(
    close: pd.DataFrame,
    open_: pd.DataFrame,
    signals: pd.DataFrame,
    initial_capital: float = 100_000.0,
    commission: float = 0.001,
) -> tuple:
    """
    Simulates trading day by day.
    Signals are computed from close prices (already shifted by 1 in the strategy).
    Trades execute at the SAME day's open price — since the signal was already
    shifted forward by one day in the strategy, this open price is the next
    available price after the signal was known.
    Portfolio value (equity) is marked to market using close prices.
    """
    cash = initial_capital
    holdings = pd.Series(0.0, index=close.columns)

    equity_records = []
    trade_records = []

    for date in close.index:
        close_row = close.loc[date]
        open_row  = open_.loc[date]
        signal_row = signals.loc[date]

        # --- Step 1: Determine target holdings using TODAY'S OPEN as the fill price ---
        # Portfolio value at the moment of trading (using open price, since that's
        # the price at which we're about to transact)
        portfolio_value_at_open = cash + (holdings * open_row).sum()

        active_tickers = signal_row[signal_row == 1].index.tolist()
        target_holdings = pd.Series(0.0, index=close.columns)

        if len(active_tickers) > 0:
            allocation_per_asset = portfolio_value_at_open / len(active_tickers)
            for ticker in active_tickers:
                if open_row[ticker] > 0:
                    target_holdings[ticker] = allocation_per_asset / open_row[ticker]

        # --- Step 2: Execute trades AT THE OPEN PRICE ---
        trades_needed = target_holdings - holdings

        for ticker, shares_to_trade in trades_needed.items():
            if abs(shares_to_trade) < 1e-6:
                continue

            trade_value = abs(shares_to_trade) * open_row[ticker]
            fee = trade_value * commission

            cash -= shares_to_trade * open_row[ticker] + fee
            holdings[ticker] += shares_to_trade

            trade_records.append({
                "date": date,
                "ticker": ticker,
                "shares": shares_to_trade,
                "price": open_row[ticker],   # filled at open, not close
                "commission": fee,
            })

        # --- Step 3: Mark to market using CLOSE price (end-of-day valuation) ---
        equity_records.append({
            "date": date,
            "equity": cash + (holdings * close_row).sum(),
            "cash": cash,
        })

    equity_curve = pd.DataFrame(equity_records).set_index("date")
    trade_log = pd.DataFrame(trade_records)

    return equity_curve, trade_log