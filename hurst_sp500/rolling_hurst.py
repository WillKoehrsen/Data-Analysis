"""
Rolling Hurst exponent (Rescaled Range, R/S) for S&P 500 prices stored in DuckDB.

What this module does
---------------------
* Finds the prices table in a ``.duckdb`` file automatically (ticker / date /
  close-style columns, case-insensitive) -- or takes explicit overrides.
* Loads it read-only into a tidy ``ticker, date, price`` DataFrame
  (adjusted close preferred over raw close: splits/dividends otherwise look
  like giant "mean-reverting" jumps).
* Computes the rolling Hurst exponent of **log returns** for several windows
  (default 126 / 252 / 504 trading days) with the classical R/S estimator.

The finite-sample correction (important)
-----------------------------------------
The textbook R/S estimator is badly biased upward on short windows: fed with
pure white-noise returns (true H = 0.5) it returns ~0.58-0.61 for the windows
used here, so a naive "H > 0.55 = trending" rule flags almost everything as
trending. This module therefore defaults to ``corrected=True``: each scale's
mean R/S is divided by its *expected* value under iid Gaussian noise
(Anis & Lloyd, 1976) and multiplied back by ``sqrt(n)``, which forces the
estimator to be centred on 0.5 for a random walk at every window size.

Measured on simulated fractional Gaussian noise (200 replicates, see
``test_rolling_hurst.py`` and the notebook's validation section):

    =========  ========  ========  ========
    true H      126d      252d      504d     (AL-corrected, mean estimate)
    =========  ========  ========  ========
    0.30       0.34      0.34      0.34
    0.50       0.50      0.49      0.49
    0.80       0.68      0.69      0.71
    =========  ========  ========  ========

i.e. estimates are correctly *ordered* and unbiased at H = 0.5, but
compressed toward 0.5 at the extremes. Read H as a regime indicator
("more/less persistent than a random walk"), not as a precise physical
constant. ``corrected=False`` recovers the textbook estimator for comparison
with other libraries (expect it to read ~0.08 higher on noise-like data).

Speed
-----
If ``numba`` is installed the rolling loop is JIT-compiled (500 tickers x
10 years x 3 windows at ``step=5`` takes seconds). Without numba a NumPy
fallback is used -- still fine, but prefer ``step=5`` (weekly Hurst points)
or ``step=21`` there. Rolling windows overlap almost completely from one day
to the next, so daily points add compute, not information.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "HAS_NUMBA",
    "PriceSchema",
    "make_scales",
    "expected_rs_anis_lloyd",
    "hurst_rs",
    "hurst_rs_details",
    "rolling_hurst",
    "discover_price_schema",
    "load_prices",
    "compute_rolling_hurst",
    "simulate_fgn",
    "white_noise_null_bands",
    "latest_snapshot",
]

# --------------------------------------------------------------------------
# Optional numba acceleration
# --------------------------------------------------------------------------
try:
    from numba import njit

    HAS_NUMBA = True
except Exception:  # pragma: no cover - exercised only without numba
    HAS_NUMBA = False

    def njit(*args, **kwargs):  # type: ignore[no-redef]
        def wrap(fn):
            return fn

        return wrap(args[0]) if args and callable(args[0]) else wrap


# --------------------------------------------------------------------------
# Core R/S estimator
# --------------------------------------------------------------------------
def make_scales(window: int, n_scales: int = 8, min_scale: int = 10) -> np.ndarray:
    """Log-spaced sub-period lengths used inside one rolling window.

    Scales run from ``min_scale`` to ``window // 2`` (at least two blocks per
    scale, so every R/S point is an average, not a single-block fluke).
    """
    if window < 4 * min_scale:
        raise ValueError(
            f"window={window} too small for min_scale={min_scale}; "
            "Hurst needs window >= ~4x min_scale to fit a scaling line."
        )
    scales = np.unique(
        np.logspace(np.log10(min_scale), np.log10(window // 2), num=n_scales).astype(int)
    )
    if len(scales) < 4:
        raise ValueError(f"Only {len(scales)} distinct scales for window={window}")
    return scales


def expected_rs_anis_lloyd(n: int) -> float:
    """Expected R/S of iid Gaussian noise for sub-period length ``n``.

    Anis & Lloyd (1976), with the (n - 0.5)/n prefactor that matches the
    sample standard deviation (ddof=1) used here:

        E[(R/S)_n] = ((n-0.5)/n) * Gamma((n-1)/2) / (sqrt(pi) Gamma(n/2))
                     * sum_{i=1}^{n-1} sqrt((n-i)/i)
    """
    if n < 2:
        raise ValueError("n must be >= 2")
    i = np.arange(1, n, dtype=np.float64)
    total = float(np.sum(np.sqrt((n - i) / i)))
    log_pref = math.lgamma((n - 1) / 2.0) - 0.5 * math.log(math.pi) - math.lgamma(n / 2.0)
    return ((n - 0.5) / n) * math.exp(log_pref) * total


def _scale_arrays(scales: np.ndarray, corrected: bool) -> tuple[np.ndarray, np.ndarray]:
    """Return (log_scales, adjust) where log(R/S) + adjust is what gets regressed."""
    log_scales = np.log(scales.astype(np.float64))
    if corrected:
        log_expected = np.log([expected_rs_anis_lloyd(int(s)) for s in scales])
        adjust = -log_expected + 0.5 * log_scales
    else:
        adjust = np.zeros_like(log_scales)
    return log_scales, adjust


def _mean_rs_at_scales(x: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """Mean R/S at each scale (NumPy reference implementation)."""
    out = np.full(len(scales), np.nan)
    for j, n in enumerate(scales):
        n = int(n)
        n_blocks = len(x) // n
        if n_blocks == 0:
            continue
        blocks = x[: n_blocks * n].reshape(n_blocks, n)
        demeaned = blocks - blocks.mean(axis=1, keepdims=True)
        cumdev = np.cumsum(demeaned, axis=1)
        r = cumdev.max(axis=1) - cumdev.min(axis=1)
        s = blocks.std(axis=1, ddof=1)
        ok = s > 1e-12
        if ok.any():
            out[j] = float(np.mean(r[ok] / s[ok]))
    return out


def _slope(log_x: np.ndarray, log_y: np.ndarray) -> tuple[float, float]:
    ok = np.isfinite(log_x) & np.isfinite(log_y)
    if ok.sum() < 4:
        return float("nan"), float("nan")
    lx, ly = log_x[ok], log_y[ok]
    slope, intercept = np.polyfit(lx, ly, 1)
    return float(slope), float(intercept)


def hurst_rs(
    x: Sequence[float] | np.ndarray,
    scales: np.ndarray | None = None,
    corrected: bool = True,
) -> float:
    """Hurst exponent of a return series ``x`` via classical R/S analysis.

    Parameters
    ----------
    x : 1-D array of (log) returns -- NOT price levels. Feeding prices
        double-integrates the series and pushes H toward 1.
    scales : sub-period lengths; defaults to ``make_scales(len(x))``.
    corrected : apply the Anis-Lloyd finite-sample correction (recommended).
    """
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if len(x) < 40 or np.std(x) <= 1e-12:
        return float("nan")
    if scales is None:
        scales = make_scales(len(x))
    mean_rs = _mean_rs_at_scales(x, scales)
    log_scales, adjust = _scale_arrays(scales, corrected)
    h, _ = _slope(log_scales, np.log(mean_rs) + adjust)
    return h


def hurst_rs_details(
    x: Sequence[float] | np.ndarray,
    scales: np.ndarray | None = None,
    corrected: bool = True,
) -> dict:
    """Like :func:`hurst_rs` but also returns the log-log scaling points.

    Useful for plotting the regression that a single H value summarises --
    if those points are not roughly on a straight line, don't trust H.
    """
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if scales is None:
        scales = make_scales(len(x))
    mean_rs = _mean_rs_at_scales(x, scales)
    expected = np.array([expected_rs_anis_lloyd(int(s)) for s in scales])
    log_scales, adjust = _scale_arrays(scales, corrected)
    log_rs_adj = np.log(mean_rs) + adjust
    h, intercept = _slope(log_scales, log_rs_adj)
    return {
        "hurst": h,
        "intercept": intercept,
        "scales": scales,
        "mean_rs": mean_rs,
        "expected_rs_white_noise": expected,
        "log_scales": log_scales,
        "log_rs_adjusted": log_rs_adj,
        "corrected": corrected,
    }


# --------------------------------------------------------------------------
# Fast rolling implementation (numba when available)
# --------------------------------------------------------------------------
@njit(cache=False, fastmath=True)
def _hurst_one_numba(x: np.ndarray, scales: np.ndarray, log_scales: np.ndarray, adjust: np.ndarray) -> float:
    n_scales = scales.shape[0]
    log_rs = np.empty(n_scales, dtype=np.float64)
    valid = np.empty(n_scales, dtype=np.float64)
    w = x.shape[0]
    n_valid = 0
    for si in range(n_scales):
        n = scales[si]
        n_blocks = w // n
        rs_sum = 0.0
        rs_cnt = 0
        for b in range(n_blocks):
            start = b * n
            mean = 0.0
            for i in range(n):
                mean += x[start + i]
            mean /= n
            var_sum = 0.0
            cum = 0.0
            cmax = -1e300
            cmin = 1e300
            for i in range(n):
                d = x[start + i] - mean
                var_sum += d * d
                cum += d
                if cum > cmax:
                    cmax = cum
                if cum < cmin:
                    cmin = cum
            if n > 1:
                s = np.sqrt(var_sum / (n - 1))
                if s > 1e-12:
                    rs_sum += (cmax - cmin) / s
                    rs_cnt += 1
        if rs_cnt > 0 and rs_sum > 0.0:
            log_rs[n_valid] = np.log(rs_sum / rs_cnt) + adjust[si]
            valid[n_valid] = log_scales[si]
            n_valid += 1
    if n_valid < 4:
        return np.nan
    mx = 0.0
    my = 0.0
    for i in range(n_valid):
        mx += valid[i]
        my += log_rs[i]
    mx /= n_valid
    my /= n_valid
    num = 0.0
    den = 0.0
    for i in range(n_valid):
        dx = valid[i] - mx
        num += dx * (log_rs[i] - my)
        den += dx * dx
    if den <= 0.0:
        return np.nan
    return num / den


@njit(cache=False, fastmath=True)
def _rolling_numba(
    x: np.ndarray,
    window: int,
    step: int,
    first_end: int,
    scales: np.ndarray,
    log_scales: np.ndarray,
    adjust: np.ndarray,
    out: np.ndarray,
    positions: np.ndarray,
) -> int:
    k = 0
    end = first_end
    while end <= x.shape[0]:
        out[k] = _hurst_one_numba(x[end - window : end], scales, log_scales, adjust)
        positions[k] = end - 1
        k += 1
        end += step
    return k


def rolling_hurst(
    returns: Sequence[float] | np.ndarray,
    window: int,
    step: int = 5,
    corrected: bool = True,
    n_scales: int = 8,
    min_scale: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Rolling Hurst of one return series.

    Returns ``(positions, values)`` where ``positions[i]`` is the index of
    the *last* return in the i-th window (use it to align H with dates:
    H at position p uses returns[p-window+1 .. p], no look-ahead).

    Evaluation points sit on a grid anchored to the start of the series
    (window-end indices are multiples of ``step``), so different ``window``
    values on the same series are evaluated on the *same* dates and can be
    merged column-wise without misalignment.
    """
    x = np.asarray(returns, dtype=np.float64)
    if len(x) < window:
        return np.empty(0, dtype=int), np.empty(0, dtype=np.float64)
    scales = make_scales(window, n_scales=n_scales, min_scale=min_scale)
    log_scales, adjust = _scale_arrays(scales, corrected)
    first_end = window + ((step - window % step) % step)
    ends = np.arange(first_end, len(x) + 1, step)
    out = np.full(len(ends), np.nan)
    positions = (ends - 1).astype(int)
    if HAS_NUMBA:
        _rolling_numba(x, int(window), int(step), int(first_end), scales.astype(np.int64), log_scales, adjust, out, positions)
    else:
        for k, end in enumerate(ends):
            out[k] = hurst_rs(x[end - window : end], scales=scales, corrected=corrected)
    return positions, out


# --------------------------------------------------------------------------
# DuckDB discovery + loading
# --------------------------------------------------------------------------
def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


_PRICE_PREFERENCE = [
    "adjclose", "adjustedclose", "adjcloseprice", "closeadj", "adjprice",
    "close", "closingprice", "closeprice", "last", "price", "pxlast",
]
_TICKER_ALIASES = {"ticker", "symbol", "ric", "code", "asset", "security", "instrument", "stock"}
_DATE_ALIASES = {"date", "tradingdate", "tradedate", "day", "timestamp", "datetime", "time", "businessdate"}


@dataclass
class PriceSchema:
    table: str
    ticker_col: str | None
    date_col: str
    price_col: str
    all_columns: list[str] = field(default_factory=list)
    candidate_tables: list[str] = field(default_factory=list)

    def __str__(self) -> str:  # pragma: no cover - display helper
        return (
            f"table={self.table!r}  ticker_col={self.ticker_col!r}  "
            f"date_col={self.date_col!r}  price_col={self.price_col!r}\n"
            f"all columns: {self.all_columns}\n"
            f"other candidate tables seen: {self.candidate_tables}"
        )


def _quote_ident(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def discover_price_schema(
    db_path: str | Path,
    table: str | None = None,
    ticker_col: str | None = None,
    date_col: str | None = None,
    price_col: str | None = None,
) -> PriceSchema:
    """Inspect a DuckDB file and locate the OHLCV/prices table + key columns.

    Any argument left as ``None`` is auto-detected (case/underscore
    insensitive). Auto-detection prefers an adjusted-close column over a raw
    close, and among tables prefers the one containing ticker+date+close.
    """
    import duckdb

    con = duckdb.connect(str(db_path), read_only=True)
    try:
        tables = [r[0] for r in con.execute("SHOW TABLES").fetchall()]
        if not tables:
            # views only?
            tables = [
                r[0]
                for r in con.execute(
                    "SELECT table_name FROM information_schema.tables"
                ).fetchall()
            ]
        if not tables:
            raise ValueError(f"No tables found in {db_path}")

        def columns_of(t: str) -> list[str]:
            return [r[0] for r in con.execute(f"DESCRIBE {_quote_ident(t)}").fetchall()]

        def score(t: str) -> tuple[int, list[str]]:
            cols = columns_of(t)
            norms = {_norm(c) for c in cols}
            s = 0
            if norms & _TICKER_ALIASES:
                s += 2
            if norms & _DATE_ALIASES:
                s += 2
            if norms & set(_PRICE_PREFERENCE):
                s += 3
            return s, cols

        chosen = table
        if chosen is None:
            ranked = sorted(((score(t)[0], t) for t in tables), reverse=True)
            if ranked[0][0] < 5:
                raise ValueError(
                    "Could not find a table with date + close-like columns. "
                    f"Tables seen: {tables}. Pass table=/date_col=/price_col= explicitly."
                )
            chosen = ranked[0][1]

        cols = columns_of(chosen)
        by_norm = {}
        for c in cols:  # first spelling wins per normalised name
            by_norm.setdefault(_norm(c), c)

        def pick(explicit: str | None, aliases: Iterable[str], what: str, required: bool = True):
            if explicit is not None:
                if explicit not in cols:
                    raise ValueError(f"{what} column {explicit!r} not in {chosen!r}; columns are {cols}")
                return explicit
            for a in aliases:
                if a in by_norm:
                    return by_norm[a]
            if required:
                raise ValueError(f"Could not auto-detect {what} column in {chosen!r}; columns are {cols}")
            return None

        return PriceSchema(
            table=chosen,
            ticker_col=pick(ticker_col, sorted(_TICKER_ALIASES, key=lambda a: a != "ticker"), "ticker", required=False),
            date_col=pick(date_col, sorted(_DATE_ALIASES, key=lambda a: a != "date"), "date"),
            price_col=pick(price_col, _PRICE_PREFERENCE, "price/close"),
            all_columns=cols,
            candidate_tables=tables,
        )
    finally:
        con.close()


def load_prices(
    db_path: str | Path,
    table: str | None = None,
    ticker_col: str | None = None,
    date_col: str | None = None,
    price_col: str | None = None,
    tickers: Sequence[str] | None = None,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Load prices from DuckDB as a tidy DataFrame: ``ticker, date, price``.

    The file is opened **read-only** and is never modified. Rows are sorted
    by ticker/date, duplicate (ticker, date) rows keep the last value, and
    non-positive / missing prices are dropped (log returns need price > 0).
    """
    import duckdb

    schema = discover_price_schema(db_path, table, ticker_col, date_col, price_col)
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        ticker_expr = (
            f"CAST({_quote_ident(schema.ticker_col)} AS VARCHAR) AS ticker"
            if schema.ticker_col
            else "'SP500' AS ticker"
        )
        sql = (
            f"SELECT {ticker_expr}, CAST({_quote_ident(schema.date_col)} AS DATE) AS date, "
            f"CAST({_quote_ident(schema.price_col)} AS DOUBLE) AS price "
            f"FROM {_quote_ident(schema.table)} WHERE {_quote_ident(schema.price_col)} IS NOT NULL"
        )
        params: list = []
        if tickers and schema.ticker_col:
            placeholders = ", ".join("?" for _ in tickers)
            sql += f" AND {_quote_ident(schema.ticker_col)} IN ({placeholders})"
            params.extend(list(tickers))
        if start:
            sql += f" AND CAST({_quote_ident(schema.date_col)} AS DATE) >= CAST(? AS DATE)"
            params.append(start)
        if end:
            sql += f" AND CAST({_quote_ident(schema.date_col)} AS DATE) <= CAST(? AS DATE)"
            params.append(end)
        sql += " ORDER BY 1, 2"
        df = con.execute(sql, params).fetchdf()
    finally:
        con.close()

    df.columns = ["ticker", "date", "price"]
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["price"] > 0]
    df = df.drop_duplicates(["ticker", "date"], keep="last").sort_values(["ticker", "date"])
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------
# Universe-wide rolling Hurst
# --------------------------------------------------------------------------
def compute_rolling_hurst(
    prices: pd.DataFrame,
    windows: Sequence[int] = (126, 252, 504),
    step: int = 5,
    corrected: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """Rolling Hurst for every ticker in a tidy ``ticker, date, price`` frame.

    Windows count **log returns** (a 252 window uses the last 252 daily log
    returns, i.e. 253 prices). Rolling is over each ticker's own observation
    count, not calendar days, so a ticker with trading halts/missing rows
    gets a window of its last ``window`` available returns.

    Returns a wide DataFrame: ``ticker, date, hurst_126, hurst_252, hurst_504``
    where ``date`` is the last date inside the window (no look-ahead).
    """
    required = {"ticker", "date", "price"}
    missing = required - set(prices.columns)
    if missing:
        raise ValueError(f"prices is missing columns {missing}; expected ticker/date/price")

    prices = prices.sort_values(["ticker", "date"])
    per_window: list[pd.DataFrame] = []
    tickers = prices["ticker"].unique()
    if verbose:
        print(
            f"Rolling Hurst: {len(tickers)} tickers x windows={tuple(windows)} "
            f"x step={step} (numba={'on' if HAS_NUMBA else 'OFF - numpy fallback, slower'})"
        )

    grouped = dict(tuple(prices.groupby("ticker", sort=True)))
    frames: dict[int, list[pd.DataFrame]] = {int(w): [] for w in windows}
    for i, ticker in enumerate(tickers, start=1):
        g = grouped[ticker]
        p = g["price"].to_numpy(dtype=np.float64)
        dates = g["date"].to_numpy()
        if len(p) < 2:
            continue
        log_ret = np.diff(np.log(p))          # length n-1
        ret_dates = dates[1:]                  # return i is realised on dates[i+1]
        for w in windows:
            w = int(w)
            positions, values = rolling_hurst(log_ret, window=w, step=step, corrected=corrected)
            if len(positions) == 0:
                continue
            frames[w].append(
                pd.DataFrame(
                    {"ticker": ticker, "date": ret_dates[positions], f"hurst_{w}": values}
                )
            )
        if verbose and (i % 100 == 0 or i == len(tickers)):
            print(f"  {i}/{len(tickers)} tickers done")

    result: pd.DataFrame | None = None
    for w in windows:
        w = int(w)
        if not frames[w]:
            continue
        part = pd.concat(frames[w], ignore_index=True)
        result = part if result is None else result.merge(part, on=["ticker", "date"], how="outer")
    if result is None:
        cols = ["ticker", "date"] + [f"hurst_{int(w)}" for w in windows]
        return pd.DataFrame(columns=cols)
    return result.sort_values(["ticker", "date"]).reset_index(drop=True)


def latest_snapshot(result: pd.DataFrame, window: int = 252) -> pd.DataFrame:
    """Most recent Hurst value per ticker for one window, sorted descending."""
    col = f"hurst_{window}"
    snap = (
        result.dropna(subset=[col])
        .sort_values("date")
        .groupby("ticker", as_index=False)
        .tail(1)[["ticker", "date", col]]
        .sort_values(col, ascending=False)
        .reset_index(drop=True)
    )
    return snap.rename(columns={"date": "as_of", col: "hurst"})


# --------------------------------------------------------------------------
# Simulation helpers (validation + demo data)
# --------------------------------------------------------------------------
_fgn_cholesky_cache: dict[tuple[int, float], np.ndarray] = {}


def simulate_fgn(n: int, hurst_true: float, rng: np.random.Generator | None = None) -> np.ndarray:
    """Fractional Gaussian noise with the given true Hurst (exact Cholesky).

    Exact but O(n^3) to set up (cached per (n, H)); keep n <= ~4000. fGn is
    the *increment* process of fractional Brownian motion, so
    ``cumsum(simulate_fgn(...))`` is an fBm price-like path and the fGn
    itself plays the role of returns with Hurst ``hurst_true``.
    """
    if not (0.05 < hurst_true < 0.95):
        raise ValueError("hurst_true must be in (0.05, 0.95)")
    rng = rng or np.random.default_rng()
    if abs(hurst_true - 0.5) < 1e-12:
        return rng.standard_normal(n)
    key = (n, round(float(hurst_true), 10))
    if key not in _fgn_cholesky_cache:
        k = np.arange(n, dtype=np.float64)
        gamma = 0.5 * (np.abs(k + 1) ** (2 * hurst_true) - 2 * np.abs(k) ** (2 * hurst_true) + np.abs(k - 1) ** (2 * hurst_true))
        idx = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :])
        cov = gamma[idx]
        _fgn_cholesky_cache[key] = np.linalg.cholesky(cov + 1e-10 * np.eye(n))
    return _fgn_cholesky_cache[key] @ rng.standard_normal(n)


def white_noise_null_bands(
    windows: Sequence[int] = (126, 252, 504),
    n_sims: int = 400,
    corrected: bool = True,
    quantiles: Sequence[float] = (0.05, 0.5, 0.95),
    seed: int = 42,
) -> pd.DataFrame:
    """Distribution of estimated H when returns are pure white noise.

    Use these bands instead of a hard-coded 0.5 line: H above the 95th
    percentile is more persistent than a random walk would typically look
    at that window size; below the 5th, more anti-persistent. Takes a few
    seconds with numba, under a minute without.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for w in windows:
        w = int(w)
        scales = make_scales(w)
        ests = np.array([hurst_rs(rng.standard_normal(w), scales=scales, corrected=corrected) for _ in range(n_sims)])
        ests = ests[np.isfinite(ests)]
        row = {"window": w, "mean": float(np.mean(ests)), "std": float(np.std(ests))}
        for q in quantiles:
            row[f"q{int(q * 100):02d}"] = float(np.quantile(ests, q))
        rows.append(row)
    return pd.DataFrame(rows)
