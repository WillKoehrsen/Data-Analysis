# Rolling Hurst Exponent — S&P 500 DuckDB

Rolling **126 / 252 / 504-trading-day** Hurst exponent (classical rescaled-range
R/S) for every ticker in a standard OHLCV table stored in a local **DuckDB**
file. Notebook-only output: your database is opened **read-only** and is never
modified.

## Files

| File | Purpose |
|---|---|
| `rolling_hurst_sp500.ipynb` | Main notebook — executed end-to-end on a synthetic demo DB so you can preview every chart before running anything |
| `rolling_hurst.py` | Tested core library imported by the notebook (schema auto-detection, DuckDB loading, R/S estimator, rolling engine) |
| `make_demo_duckdb.py` | Builds a fake `demo_sp500.duckdb` (30 tickers incl. three with known true Hurst) so the notebook runs before you point it at real data |
| `test_rolling_hurst.py` | 12 pytest tests: estimator calibration on known-H series, no-look-ahead alignment, messy-column DuckDB round-trip, etc. |
| `requirements.txt` | Minimal deps for this project (use a fresh venv — see below) |

## Run it on your laptop

Use a **fresh virtual environment**. The `requirements.txt` in the repo root
pins 2019-era packages for the older notebooks; do not reuse that environment.

```bash
# macOS / Linux                     # Windows (PowerShell): py -m venv .venv ; .venv\Scripts\Activate.ps1
cd hurst_sp500
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt     # duckdb, pandas, numpy, matplotlib, jupyter, numba
python -m pytest test_rolling_hurst.py -q   # optional: 12 tests, ~2 s
jupyter lab rolling_hurst_sp500.ipynb
```

Then edit **one line** in the notebook's config cell (§0):

```python
DB_PATH = Path(r"C:\data\sp500.duckdb")                    # Windows example
DB_PATH = Path("~/data/sp500.duckdb").expanduser()        # Mac / Linux example
```

and *Run All*. If that file doesn't exist, the notebook builds and runs
against a synthetic demo database instead (clearly banner-labelled), so a
first *Run All* always works.

### Expected database shape

Any standard OHLCV layout works. Table and column names are auto-detected
case/underscore-insensitively, e.g. table `prices` / `ohlcv` / `daily_bars`
with columns like

```
ticker | date | open | high | low | close | adj_close | volume
```

(`symbol`/`trade_date`/`Adj Close` style names are also recognised.)
**Adjusted close is used when present**, otherwise raw close — splits and
dividends in a raw close otherwise masquerade as violent mean reversion.
If auto-detection picks the wrong table/column, set `TABLE`, `TICKER_COL`,
`DATE_COL` or `PRICE_COL` in the config cell. §1 of the notebook prints
exactly what was detected.

## Method in 30 seconds

For each rolling window of daily **log returns**, split the window into blocks
of length *n* (8 log-spaced scales from 10 up to window/2), compute the
rescaled range *R/S* per block, average per scale, and take *H* as the slope
of log(R/S) vs log(n). *H* ≈ 0.5 random walk, > 0.5 persistent/trending,
< 0.5 anti-persistent/mean-reverting.

**Finite-sample correction (on by default).** Textbook R/S is biased upward
on short windows — pure noise reads ≈ 0.58–0.61 at these window sizes, so a
naive "H > 0.55 = trending" rule flags nearly everything. The estimator here
divides each scale by its expected R/S under iid Gaussian noise
(Anis & Lloyd, 1976) before fitting. Measured on simulated series with known
Hurst (notebook §3 reproduces this live):

| true H | 126d | 252d | 504d | textbook R/S, 252d |
|---|---|---|---|---|
| 0.30 | 0.34 | 0.34 | 0.34 | 0.43 |
| 0.50 | 0.50 | 0.49 | 0.49 | 0.59 |
| 0.80 | 0.68 | 0.69 | 0.71 | 0.78 |

Estimates are correctly ordered and centred on 0.5 for a random walk, but
compressed toward 0.5 at the extremes. The notebook also simulates a
**random-walk band per window** (e.g. 252d ≈ 0.40–0.60) — judge tickers
against their band, not against a hard 0.5 line. Set `CORRECTED = False` in
the config to get textbook values comparable to other R/S libraries.

## Performance

Windows are evaluated every `STEP = 5` trading days (weekly Hurst points on a
date grid shared by all three windows). With `numba` installed, 30 tickers ×
10 years × 3 windows takes < 1 s of rolling compute (≈ 15 s extrapolated to
500 tickers); daily points (`STEP = 1`) are also fine. Without numba the
NumPy fallback is roughly two orders of magnitude slower — keep `STEP = 5`
or 21 there.

## Troubleshooting

* **`IO Error: Could not set lock on file`** — DuckDB allows one read-write
  process per file. Close any DBeaver/CLI/Python session that opened the
  database read-write, or copy the `.duckdb` file and point `DB_PATH` at the
  copy. The notebook itself always connects `read_only=True`.
* **Wrong table/columns detected** — set the overrides in §0 (§1 prints the
  detected schema, all candidate tables and columns).
* **`ModuleNotFoundError: numba`** — harmless; the notebook falls back to
  NumPy. Install numba for speed.
* **A ticker is missing from the 504d results** — it has ≤ 504 prices in the
  DB. §2 prints how many tickers qualify per window.

## Caveats

Hurst here is descriptive of the window it covers, not a forecast. Rolling
values overlap almost completely day to day (hence smooth-looking paths),
a present-day S&P 500 constituent list backfilled into history flatters
long-window persistence (survivorship bias), and ranking ~500 tickers
guarantees extreme-looking tails even under a pure random walk. Notebook §9
spells these out — read it [file info] before trading on any of this.
