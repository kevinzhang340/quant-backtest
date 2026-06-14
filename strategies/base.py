import pandas as pd

class Strategy:
    """
    Base class for all strategies.
    Subclasses must implement generate_signals().
    """
    def generate_signals(self, prices: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError("Subclasses must implement generate_signals()")


class MACrossover(Strategy):
    """
    Dual moving average crossover.
    Signal = 1 (long) when fast MA > slow MA, else 0 (flat).
    """
    def __init__(self, fast: int = 20, slow: int = 50):
        self.fast = fast
        self.slow = slow

    def generate_signals(self, prices: pd.DataFrame) -> pd.DataFrame:
        fast_ma = prices.rolling(window=self.fast).mean()
        slow_ma = prices.rolling(window=self.slow).mean()

        signals = (fast_ma > slow_ma).astype(int)

        # Shift by 1 to prevent look-ahead bias
        signals = signals.shift(1)
        signals.fillna(0, inplace=True)

        return signals