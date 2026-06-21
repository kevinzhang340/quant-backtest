import yfinance as yf
import pandas as pd
import os

CACHE_DIR = "data"

def get_prices(tickers: list, start: str, end: str) -> dict:
    """
    Download adjusted Open and Close prices for a list of tickers.
    Caches to Parquet files so you don't hit the network every run.
    Returns a dict: {"close": DataFrame, "open": DataFrame}
    """
    close_cache = os.path.join(CACHE_DIR, f"close_{start}_{end}.parquet")
    open_cache  = os.path.join(CACHE_DIR, f"open_{start}_{end}.parquet")

    if os.path.exists(close_cache) and os.path.exists(open_cache):
        print(f"Loading from cache: {close_cache}, {open_cache}")
        return {
            "close": pd.read_parquet(close_cache),
            "open":  pd.read_parquet(open_cache),
        }

    print(f"Downloading {tickers} from {start} to {end}...")
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True)

    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"]
        open_ = raw["Open"]
    else:
        close = raw[["Close"]].rename(columns={"Close": tickers[0]})
        open_ = raw[["Open"]].rename(columns={"Open": tickers[0]})

    close.dropna(how="all", inplace=True)
    open_.dropna(how="all", inplace=True)

    close.to_parquet(close_cache)
    open_.to_parquet(open_cache)

    return {"close": close, "open": open_}