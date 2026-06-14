import yfinance as yf
import pandas as pd
import os

CACHE_DIR = "data"

def get_prices(tickers: list, start: str, end: str) -> pd.DataFrame:
    """
    Download adjusted closing prices for a list of tickers.
    Caches to a Parquet file so you don't hit the network every run.
    Returns a DataFrame of shape (trading_days, n_tickers).
    """
    cache_file = os.path.join(CACHE_DIR, f"prices_{start}_{end}.parquet")

    if os.path.exists(cache_file):
        print(f"Loading from cache: {cache_file}")
        return pd.read_parquet(cache_file)

    print(f"Downloading {tickers} from {start} to {end}...")
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True)

    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]].rename(columns={"Close": tickers[0]})

    prices.dropna(how="all", inplace=True)
    prices.to_parquet(cache_file)
    return prices