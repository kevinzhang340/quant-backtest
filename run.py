from data.data import get_prices
from strategies.base import MACrossover, WithMonthlyContribution
from backtest.combined_engine import run_combined_backtest
from analytics.metrics import compute_dca_metrics, plot_equity

TICKERS = ["SPY", "QQQ", "IEF"]
START   = "2015-01-01"
END     = "2024-01-01"

prices = get_prices(TICKERS, START, END)
close  = prices["close"]
open_  = prices["open"]

strategy = WithMonthlyContribution(
    base_strategy=MACrossover(fast=20, slow=50),
    contributions={"SPY": 100, "QQQ": 100, "IEF": 100},
)

signals = strategy.generate_signals(close)
contribution_schedule = strategy.get_contribution_schedule(close)

equity, trades = run_combined_backtest(
    close, open_, signals, contribution_schedule,
    initial_capital=10_000,
)

metrics = compute_dca_metrics(equity)
print(metrics)
print(f"\nTotal trades: {len(trades)}")

plot_equity(equity["equity"], title="MA Crossover + $100/month per ticker")