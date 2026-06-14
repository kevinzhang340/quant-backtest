from data.data import get_prices
from strategies.base import MACrossover
from backtest.engine import run_backtest
from analytics.metrics import compute_metrics, plot_equity

# --- Configuration ---
TICKERS = ["SPY", "QQQ", "IEF"]
START   = "2015-01-01"
END     = "2024-01-01"

# --- Run the pipeline ---
prices   = get_prices(TICKERS, START, END)
strategy = MACrossover(fast=20, slow=50)
signals  = strategy.generate_signals(prices)

equity, trades = run_backtest(prices, signals, initial_capital=100_000)

print(compute_metrics(equity["equity"]))
print(f"\nTotal trades: {len(trades)}")

plot_equity(equity["equity"], title="MA Crossover — SPY, QQQ, IEF")