"""
Create a small *synthetic* DuckDB OHLCV database so the notebook runs
end-to-end before you point it at your real file.

Nothing here is real market data. Three teaching tickers have known Hurst:

    DEMO_RW       white-noise log returns            true H = 0.50
    DEMO_PERSIST  fractional Gaussian noise returns  true H = 0.80
    DEMO_MEANREV  fractional Gaussian noise returns  true H = 0.30

plus ``--n-tickers`` synthetic names (SYN001...) driven by a common market
factor + idiosyncratic noise (true H ~ 0.5), which is roughly what index
constituents look like next to the teaching tickers.

Usage:
    python make_demo_duckdb.py                       # -> demo_sp500.duckdb
    python make_demo_duckdb.py --out my.duckdb --n-tickers 50
"""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from rolling_hurst import simulate_fgn


def build_prices(n_tickers: int = 27, start: str = "2015-01-01", end: str = "2024-12-31", seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, end)
    n = len(dates)

    # ticker -> daily log-return series (length n-1)
    series: dict[str, np.ndarray] = {}
    print("Simulating teaching tickers (exact fGn via Cholesky - a few seconds)...")
    series["DEMO_RW"] = rng.standard_normal(n - 1) * 0.010
    series["DEMO_PERSIST"] = simulate_fgn(n - 1, 0.80, rng) * 0.010
    series["DEMO_MEANREV"] = simulate_fgn(n - 1, 0.30, rng) * 0.010

    market = rng.standard_normal(n - 1) * 0.008 + 0.0003
    for i in range(1, n_tickers + 1):
        beta = rng.uniform(0.6, 1.4)
        idio_vol = rng.uniform(0.006, 0.016)
        drift = rng.uniform(-0.0001, 0.0006)
        series[f"SYN{i:03d}"] = drift + beta * market + rng.standard_normal(n - 1) * idio_vol

    frames = []
    for ticker, log_ret in series.items():
        close = 100.0 * np.exp(np.concatenate([[0.0], np.cumsum(log_ret)]))
        open_ = np.concatenate([[close[0]], close[:-1]]) * np.exp(rng.standard_normal(n) * 0.001)
        high = np.maximum(open_, close) * np.exp(np.abs(rng.standard_normal(n)) * 0.002)
        low = np.minimum(open_, close) * np.exp(-np.abs(rng.standard_normal(n)) * 0.002)
        volume = rng.lognormal(mean=15.0, sigma=0.5, size=n).astype(np.int64)
        frames.append(
            pd.DataFrame(
                {
                    "ticker": ticker,
                    "date": dates,
                    "open": open_,
                    "high": high,
                    "low": low,
                    "close": close,
                    "adj_close": close,  # no splits/dividends in synthetic land
                    "volume": volume,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="demo_sp500.duckdb")
    ap.add_argument("--n-tickers", type=int, default=27, help="synthetic tickers in addition to the 3 DEMO_* ones")
    ap.add_argument("--start", default="2015-01-01")
    ap.add_argument("--end", default="2024-12-31")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    prices = build_prices(args.n_tickers, args.start, args.end, args.seed)
    out = Path(args.out)
    if out.exists():
        out.unlink()
    con = duckdb.connect(str(out))
    try:
        con.register("prices_df", prices)
        con.execute(
            "CREATE TABLE prices AS SELECT ticker, CAST(date AS DATE) AS date, "
            '"open", high, low, "close", adj_close, volume FROM prices_df'
        )
        n_rows, n_tickers = con.execute("SELECT COUNT(*), COUNT(DISTINCT ticker) FROM prices").fetchone()
    finally:
        con.close()
    print(f"Wrote {out}  ({n_rows:,} rows, {n_tickers} tickers, {args.start}..{args.end})")


if __name__ == "__main__":
    main()
