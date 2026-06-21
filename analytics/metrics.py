import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def compute_metrics(equity: pd.Series, risk_free_rate: float = 0.04) -> dict:
    """
    Computes standard performance metrics from an equity curve.
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

def compute_dca_metrics(equity_curve: pd.DataFrame) -> dict:
    """
    For strategies with ongoing contributions (DCA or combined).
    Compares final value to total amount actually invested.
    """
    final_value = equity_curve["equity"].iloc[-1]
    total_invested = equity_curve["cash_invested"].iloc[-1]

    return {
        "total_invested": round(total_invested, 2),
        "final_value":     round(final_value, 2),
        "total_return":    round((final_value / total_invested) - 1, 4),
    }