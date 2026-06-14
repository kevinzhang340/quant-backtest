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


class RSIStrategy(Strategy):
    """
    Mean reversion strategy using the Relative Strength Index.
    Buy when RSI < oversold threshold, sell when RSI > overbought threshold.
    """
    def __init__(self, period: int = 14, oversold: int = 30, overbought: int = 70):
        self.period = period
        self.oversold = oversold
        self.overbought = overbought

    def compute_rsi(self, prices: pd.Series) -> pd.Series:
        delta = prices.diff()

        gains = delta.clip(lower=0)   # keep only positive moves
        losses = -delta.clip(upper=0) # keep only negative moves (make positive)

        avg_gain = gains.rolling(window=self.period).mean()
        avg_loss = losses.rolling(window=self.period).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def generate_signals(self, prices: pd.DataFrame) -> pd.DataFrame:
        signals = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

        for ticker in prices.columns:
            rsi = self.compute_rsi(prices[ticker])

            # 1 when RSI crosses back above oversold, 0 when it crosses above overbought
            position = 0
            signal_list = []

            for val in rsi:
                if pd.isna(val):
                    signal_list.append(0)
                elif val < self.oversold:
                    position = 1    # enter long
                    signal_list.append(position)
                elif val > self.overbought:
                    position = 0    # exit
                    signal_list.append(position)
                else:
                    signal_list.append(position)  # hold current position

            signals[ticker] = signal_list

        # Shift by 1 to prevent look-ahead bias — same as MA crossover
        signals = signals.shift(1).fillna(0)
        return signals
    