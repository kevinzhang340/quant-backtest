import pandas as pd

class Strategy:
    def generate_signals(self, close: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError("Subclasses must implement generate_signals()")

    def get_contribution_schedule(self, close: pd.DataFrame) -> pd.DataFrame:
        """
        Returns a DataFrame, same shape as `close`, where each entry is the
        dollar amount to contribute to that ticker on that date.
        Default: no contributions at all (all zeros).
        Override this in subclasses that want DCA behavior.
        """
        return pd.DataFrame(0.0, index=close.index, columns=close.columns)

class WithMonthlyContribution(Strategy):
    """
    Wraps any existing strategy and adds a monthly dollar contribution
    on top of its signals. Lets you mix-and-match: any signal strategy
    + any per-ticker contribution amounts.
    """
    def __init__(self, base_strategy: Strategy, contributions: dict):
        """
        base_strategy: any Strategy instance (e.g. MACrossover(20, 50))
        contributions: dict mapping ticker -> dollar amount per month
                       e.g. {"SPY": 150, "QQQ": 50, "IEF": 0}
        """
        self.base_strategy = base_strategy
        self.contributions = contributions

    def generate_signals(self, close: pd.DataFrame) -> pd.DataFrame:
        # Just defer to the wrapped strategy — no changes to signal logic
        return self.base_strategy.generate_signals(close)

    def get_contribution_schedule(self, close: pd.DataFrame) -> pd.DataFrame:
        schedule = pd.DataFrame(0.0, index=close.index, columns=close.columns)

        months_seen = set()
        for date in close.index:
            month_key = (date.year, date.month)
            if month_key not in months_seen:
                months_seen.add(month_key)
                for ticker in close.columns:
                    schedule.loc[date, ticker] = self.contributions.get(ticker, 0.0)

        return schedule


class MACrossover(Strategy):
    def __init__(self, fast: int = 20, slow: int = 50):
        self.fast = fast
        self.slow = slow

    def generate_signals(self, close: pd.DataFrame) -> pd.DataFrame:
        fast_ma = close.rolling(window=self.fast).mean()
        slow_ma = close.rolling(window=self.slow).mean()

        signals = (fast_ma > slow_ma).astype(int)
        signals = signals.shift(1)
        signals.fillna(0, inplace=True)

        return signals


class RSIStrategy(Strategy):
    def __init__(self, period: int = 14, oversold: int = 30, overbought: int = 70):
        self.period = period
        self.oversold = oversold
        self.overbought = overbought

    def compute_rsi(self, close: pd.Series) -> pd.Series:
        delta = close.diff()
        gains = delta.clip(lower=0)
        losses = -delta.clip(upper=0)

        avg_gain = gains.rolling(window=self.period).mean()
        avg_loss = losses.rolling(window=self.period).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def generate_signals(self, close: pd.DataFrame) -> pd.DataFrame:
        signals = pd.DataFrame(0.0, index=close.index, columns=close.columns)

        for ticker in close.columns:
            rsi = self.compute_rsi(close[ticker])
            position = 0
            signal_list = []

            for val in rsi:
                if pd.isna(val):
                    signal_list.append(0)
                elif val < self.oversold:
                    position = 1
                    signal_list.append(position)
                elif val > self.overbought:
                    position = 0
                    signal_list.append(position)
                else:
                    signal_list.append(position)

            signals[ticker] = signal_list

        signals = signals.shift(1).fillna(0)
        return signals