"""
Data Loader Module
Downloads historical data from Yahoo Finance and performs data cleaning (NAs only, no outlier removal).
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Tuple, Optional


def download_data(
    tickers: List[str],
    years: int = 10,
    end_date: Optional[datetime] = None
) -> pd.DataFrame:
    """Download historical price data from Yahoo Finance."""
    if end_date is None:
        end_date = datetime.now()

    start_date = end_date - timedelta(days=years * 365)

    print(f"Downloading data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

    data = yf.download(
        tickers,
        start=start_date.strftime('%Y-%m-%d'),
        end=end_date.strftime('%Y-%m-%d'),
        progress=True,
        auto_adjust=True
    )

    if isinstance(data.columns, pd.MultiIndex):
        if 'Close' in data.columns.get_level_values(0):
            prices = data['Close']
        elif 'Adj Close' in data.columns.get_level_values(0):
            prices = data['Adj Close']
        else:
            raise ValueError(f"Could not find price column.")
    else:
        if 'Close' in data.columns:
            prices = data[['Close']]
        elif 'Adj Close' in data.columns:
            prices = data[['Adj Close']]
        else:
            raise ValueError(f"Could not find price column.")
        prices.columns = tickers

    return prices


def download_benchmark(
    benchmark_ticker: str = '^GSPC',
    years: int = 10,
    end_date: Optional[datetime] = None
) -> pd.DataFrame:
    """Download benchmark (S&P 500) data."""
    if end_date is None:
        end_date = datetime.now()

    start_date = end_date - timedelta(days=years * 365)

    print(f"Downloading benchmark ({benchmark_ticker}) data...")

    data = yf.download(
        benchmark_ticker,
        start=start_date.strftime('%Y-%m-%d'),
        end=end_date.strftime('%Y-%m-%d'),
        progress=False,
        auto_adjust=True
    )

    if 'Close' in data.columns:
        prices = data[['Close']]
    else:
        prices = data[['Adj Close']]

    prices.columns = [benchmark_ticker]
    return prices


def clean_data(
    prices: pd.DataFrame,
    min_data_pct: float = 0.7
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """Clean price data by removing NAs only (no outlier removal)."""
    print("\n=== Data Cleaning (NAs only, no outlier removal) ===")
    print(f"Initial shape: {prices.shape}")
    print(f"Initial NA counts:\n{prices.isna().sum()}")

    data_availability = 1 - (prices.isna().sum() / len(prices))
    valid_tickers = data_availability[data_availability >= min_data_pct].index.tolist()
    removed_tickers = [t for t in prices.columns if t not in valid_tickers]

    if removed_tickers:
        print(f"\nRemoved tickers due to insufficient data: {removed_tickers}")

    prices_filtered = prices[valid_tickers]
    prices_cleaned = prices_filtered.ffill().bfill()
    returns = prices_cleaned.pct_change().dropna()
    prices_cleaned = prices_cleaned.loc[returns.index]

    print(f"\nFinal shape: {prices_cleaned.shape}")
    print(f"Date range: {prices_cleaned.index[0]} to {prices_cleaned.index[-1]}")

    return prices_cleaned, returns, removed_tickers


def calculate_statistics(returns: pd.DataFrame) -> dict:
    """Calculate statistical measures for returns data."""
    annual_factor = 252

    mean_returns = returns.mean() * annual_factor
    cov_matrix = returns.cov() * annual_factor
    std_returns = returns.std() * np.sqrt(annual_factor)
    corr_matrix = returns.corr()

    return {
        'mean_returns': mean_returns,
        'cov_matrix': cov_matrix,
        'std_returns': std_returns,
        'corr_matrix': corr_matrix,
        'daily_returns': returns,
        'annual_factor': annual_factor
    }


def get_thai_tickers(tickers: List[str]) -> List[str]:
    """Identify Thai-related tickers from the list."""
    thai_tickers = []
    for ticker in tickers:
        if ticker.endswith('.BK') or ticker in ['THD', 'THB=X']:
            thai_tickers.append(ticker)
    return thai_tickers


def load_and_prepare_data(
    tickers: List[str],
    years: int = 10,
    end_date: Optional[datetime] = None,
    min_data_pct: float = 0.7,
    include_benchmark: bool = True
) -> dict:
    """Main function to load and prepare all data for portfolio optimization."""
    prices = download_data(tickers, years, end_date)

    benchmark_prices = None
    benchmark_returns = None
    if include_benchmark:
        benchmark_prices = download_benchmark('^GSPC', years, end_date)

    prices_cleaned, returns_cleaned, removed_tickers = clean_data(prices, min_data_pct)

    if benchmark_prices is not None:
        common_dates = prices_cleaned.index.intersection(benchmark_prices.index)
        prices_cleaned = prices_cleaned.loc[common_dates]
        returns_cleaned = returns_cleaned.loc[common_dates]
        benchmark_prices = benchmark_prices.loc[common_dates]
        benchmark_returns = benchmark_prices.pct_change().dropna()

        common_dates = returns_cleaned.index.intersection(benchmark_returns.index)
        returns_cleaned = returns_cleaned.loc[common_dates]
        benchmark_returns = benchmark_returns.loc[common_dates]
        prices_cleaned = prices_cleaned.loc[common_dates]
        benchmark_prices = benchmark_prices.loc[common_dates]

    stats = calculate_statistics(returns_cleaned)
    valid_tickers = prices_cleaned.columns.tolist()
    thai_tickers = get_thai_tickers(valid_tickers)

    print(f"\n=== Data Summary ===")
    print(f"Valid tickers: {valid_tickers}")
    print(f"Thai tickers: {thai_tickers}")
    print(f"Number of observations: {len(returns_cleaned)}")

    return {
        'prices': prices_cleaned,
        'returns': returns_cleaned,
        'tickers': valid_tickers,
        'thai_tickers': thai_tickers,
        'removed_tickers': removed_tickers,
        'statistics': stats,
        'benchmark_prices': benchmark_prices,
        'benchmark_returns': benchmark_returns
    }
