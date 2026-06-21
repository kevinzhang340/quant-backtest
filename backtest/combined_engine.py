import pandas as pd

def run_combined_backtest(
    close: pd.DataFrame,
    open_: pd.DataFrame,
    signals: pd.DataFrame,
    contribution_schedule: pd.DataFrame,   # NEW: per-ticker $ amounts per day
    initial_capital: float = 10_000.0,
    commission: float = 0.001,
    rebalance_threshold: float = 0.02,
) -> tuple:
    cash = initial_capital
    holdings = pd.Series(0.0, index=close.columns)
    cash_invested = initial_capital

    equity_records = []
    trade_records = []
    prev_signal_row = pd.Series(0, index=close.columns)

    for date in close.index:
        close_row = close.loc[date]
        open_row  = open_.loc[date]
        signal_row = signals.loc[date]
        contribution_row = contribution_schedule.loc[date]

        # --- Step 1: apply today's contributions (per ticker, from the strategy) ---
        todays_contribution = contribution_row.sum()
        if todays_contribution > 0:
            cash += todays_contribution
            cash_invested += todays_contribution

        signal_changed = not signal_row.equals(prev_signal_row)
        is_contribution_day = todays_contribution > 0

        if is_contribution_day or signal_changed:
            portfolio_value_at_open = cash + (holdings * open_row).sum()
            active_tickers = signal_row[signal_row == 1].index.tolist()
            target_holdings = pd.Series(0.0, index=close.columns)

            if len(active_tickers) > 0:
                allocation_per_asset = portfolio_value_at_open / len(active_tickers)
                for ticker in active_tickers:
                    if open_row[ticker] > 0:
                        target_holdings[ticker] = allocation_per_asset / open_row[ticker]

            trades_needed = target_holdings - holdings

            for ticker, shares_to_trade in trades_needed.items():
                trade_value = abs(shares_to_trade) * open_row[ticker]

                if trade_value < rebalance_threshold * portfolio_value_at_open:
                    continue

                fee = trade_value * commission
                cash -= shares_to_trade * open_row[ticker] + fee
                holdings[ticker] += shares_to_trade

                trade_records.append({
                    "date": date,
                    "ticker": ticker,
                    "shares": shares_to_trade,
                    "price": open_row[ticker],
                    "commission": fee,
                })

        prev_signal_row = signal_row

        equity_records.append({
            "date": date,
            "equity": cash + (holdings * close_row).sum(),
            "cash": cash,
            "cash_invested": cash_invested,
        })

    equity_curve = pd.DataFrame(equity_records).set_index("date")
    trade_log = pd.DataFrame(trade_records)

    return equity_curve, trade_log