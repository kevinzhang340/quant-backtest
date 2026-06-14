import pandas as pd
import numpy as np

def run_backtest(
    prices: pd.DataFrame,
    signals: pd.DataFrame,
    initial_capital: float = 100_000.0,
    commission: float = 0.001,
) -> tuple:
    """
    Simulates trading day by day.
    Returns (equity_curve DataFrame, trade_log DataFrame).
    """
    cash = initial_capital
    holdings = pd.Series(0.0, index=prices.columns)

    equity_records = []
    trade_records = []

    for date in prices.index:
        price_row = prices.loc[date]
        signal_row = signals.loc[date]

        # Step 1: mark to market
        portfolio_value = cash + (holdings * price_row).sum()

        # Step 2: determine target holdings (equal weight across signalled assets)
        active_tickers = signal_row[signal_row == 1].index.tolist()
        target_holdings = pd.Series(0.0, index=prices.columns)

        if len(active_tickers) > 0:
            allocation_per_asset = portfolio_value / len(active_tickers)
            for ticker in active_tickers:
                if price_row[ticker] > 0:
                    target_holdings[ticker] = allocation_per_asset / price_row[ticker]

        # Step 3: execute trades
        trades_needed = target_holdings - holdings

        for ticker, shares_to_trade in trades_needed.items():
            if abs(shares_to_trade) < 1e-6:
                continue

            trade_value = abs(shares_to_trade) * price_row[ticker]
            fee = trade_value * commission

            cash -= shares_to_trade * price_row[ticker] + fee
            holdings[ticker] += shares_to_trade

            trade_records.append({
                "date": date,
                "ticker": ticker,
                "shares": shares_to_trade,
                "price": price_row[ticker],
                "commission": fee,
            })

        # Step 4: record equity
        equity_records.append({
            "date": date,
            "equity": cash + (holdings * price_row).sum(),
            "cash": cash,
        })

    equity_curve = pd.DataFrame(equity_records).set_index("date")
    trade_log = pd.DataFrame(trade_records)

    return equity_curve, trade_log