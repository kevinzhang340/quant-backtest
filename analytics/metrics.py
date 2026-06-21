import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def compute_metrics(equity: pd.Series, risk_free_rate: float = 0.04) -> dict:
    """
    Computes standard performance metrics from an equity curve.
    Assumes a single fixed lump sum at the start — no ongoing contributions.
    Use this for pure signal-based strategies (e.g. MACrossover with a
    fixed initial_capital and no monthly contributions).
    """
    returns = equity.pct_change().dropna()

    # Sharpe ratio (annualised)
    daily_rf = risk_free_rate / 252
    excess_returns = returns - daily_rf
    sharpe = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252)

    # Maximum drawdown
    rolling_peak = equity.cummax()
    drawdown = (equity - rolling_peak) / rolling_peak
    max_drawdown = drawdown.min()

    # CAGR
    n_years = len(equity) / 252
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / n_years) - 1

    # Total return
    total_return = (equity.iloc[-1] / equity.iloc[0]) - 1

    return {
        "total_return":  round(total_return, 4),
        "cagr":          round(cagr, 4),
        "sharpe_ratio":  round(sharpe, 3),
        "max_drawdown":  round(max_drawdown, 4),
    }


def compute_dca_metrics(equity_curve: pd.DataFrame) -> dict:
    """
    For strategies with ongoing contributions (DCA or combined).
    Compares final value to total amount actually invested.
    equity_curve must have columns 'equity' and 'cash_invested'.
    """
    final_value = equity_curve["equity"].iloc[-1]
    total_invested = equity_curve["cash_invested"].iloc[-1]

    return {
        "total_invested": round(total_invested, 2),
        "final_value":     round(final_value, 2),
        "total_return":    round((final_value / total_invested) - 1, 4),
    }


def compute_returns_net_of_contributions(equity_curve: pd.DataFrame) -> pd.Series:
    """
    Computes daily percentage returns, excluding the effect of new cash
    contributions. This isolates how well the STRATEGY performed,
    independent of how much money was added and when.
    equity_curve must have columns 'equity' and 'cash_invested'.
    """
    equity = equity_curve["equity"]
    invested = equity_curve["cash_invested"]

    # Daily contribution = how much MORE was invested today vs yesterday
    daily_contribution = invested.diff().fillna(0)

    # True gain = today's equity, minus today's contribution, minus yesterday's equity
    prev_equity = equity.shift(1)
    true_gain = (equity - daily_contribution) - prev_equity

    true_return = true_gain / prev_equity
    return true_return.dropna()


def compute_metrics_from_returns(returns: pd.Series, risk_free_rate: float = 0.04) -> dict:
    """
    Computes Sharpe, max drawdown (on cumulative growth), and CAGR
    directly from a pre-computed daily return series. Works whether
    the returns came from a simple lump-sum equity curve or from
    compute_returns_net_of_contributions().
    """
    daily_rf = risk_free_rate / 252
    excess_returns = returns - daily_rf
    sharpe = (excess_returns.mean() / excess_returns.std()) * (252 ** 0.5)

    # Reconstruct a "growth of $1" curve from returns to measure drawdown
    growth = (1 + returns).cumprod()
    rolling_peak = growth.cummax()
    drawdown = (growth - rolling_peak) / rolling_peak
    max_drawdown = drawdown.min()

    n_years = len(returns) / 252
    cagr = growth.iloc[-1] ** (1 / n_years) - 1

    return {
        "sharpe_ratio": round(sharpe, 3),
        "max_drawdown": round(max_drawdown, 4),
        "cagr": round(cagr, 4),
    }


def plot_equity(equity: pd.Series, title: str = "Equity curve"):
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

    # Top panel: equity curve
    axes[0].plot(equity.index, equity.values)
    axes[0].set_title(title)
    axes[0].set_ylabel("Portfolio value ($)")
    axes[0].grid(True, alpha=0.3)

    # Bottom panel: drawdown
    rolling_peak = equity.cummax()
    drawdown = (equity - rolling_peak) / rolling_peak
    axes[1].fill_between(drawdown.index, drawdown.values, 0, color="red", alpha=0.3)
    axes[1].set_ylabel("Drawdown")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_returns_comparison(returns_dict: dict, title: str = "Strategy comparison"):
    """
    Plots growth-of-$1 curves for multiple return series on the same chart,
    so strategies with different cashflows can be compared fairly.
    returns_dict: {"label": returns_series, ...}
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    for label, returns in returns_dict.items():
        growth = (1 + returns).cumprod()
        ax.plot(growth.index, growth.values, label=label)

    ax.set_title(title)
    ax.set_ylabel("Growth of $1")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()