"""Tests for rolling_hurst. Run:  python -m pytest test_rolling_hurst.py -v"""

import numpy as np
import pandas as pd
import pytest

import rolling_hurst as rh


@pytest.fixture(scope="module")
def fgn_replicates():
    """Return series with known true Hurst (exact fGn simulation)."""
    rng = np.random.default_rng(7)
    return {h: [rh.simulate_fgn(1024, h, rng) for _ in range(20)] for h in (0.3, 0.5, 0.8)}


@pytest.mark.parametrize("window", [126, 252, 504])
def test_corrected_estimator_is_centred_on_random_walk(fgn_replicates, window):
    scales = rh.make_scales(window)
    ests = [rh.hurst_rs(x[:window], scales=scales, corrected=True) for x in fgn_replicates[0.5]]
    assert abs(np.mean(ests) - 0.5) < 0.04, f"white-noise mean H={np.mean(ests):.3f}"


def test_corrected_estimator_orders_regimes_correctly(fgn_replicates):
    scales = rh.make_scales(504)
    means = {
        h: np.mean([rh.hurst_rs(x[:504], scales=scales, corrected=True) for x in reps])
        for h, reps in fgn_replicates.items()
    }
    assert means[0.3] < 0.45 < means[0.5] < 0.60 < means[0.8]
    # compressed toward 0.5 is expected for R/S - just assert separation
    assert means[0.8] - means[0.3] > 0.25


def test_classic_estimator_is_biased_upward_on_noise(fgn_replicates):
    scales = rh.make_scales(252)
    classic = np.mean([rh.hurst_rs(x[:252], scales=scales, corrected=False) for x in fgn_replicates[0.5]])
    corrected = np.mean([rh.hurst_rs(x[:252], scales=scales, corrected=True) for x in fgn_replicates[0.5]])
    assert classic > corrected + 0.04  # documents why the correction is the default


def test_expected_rs_matches_monte_carlo():
    rng = np.random.default_rng(0)
    for n in (20, 50, 100):
        sims = []
        for _ in range(1500):
            x = rng.standard_normal(n)
            z = np.cumsum(x - x.mean())
            sims.append((z.max() - z.min()) / x.std(ddof=1))
        assert rh.expected_rs_anis_lloyd(n) == pytest.approx(np.mean(sims), rel=0.03)


def test_scales_are_valid():
    for w in (126, 252, 504):
        s = rh.make_scales(w)
        assert s.min() >= 10 and s.max() <= w // 2 and np.all(np.diff(s) > 0)
    with pytest.raises(ValueError):
        rh.make_scales(30)


def test_constant_and_short_series_return_nan_or_empty():
    assert np.isnan(rh.hurst_rs(np.zeros(300)))
    pos, vals = rh.rolling_hurst(np.random.default_rng(1).standard_normal(100), window=252)
    assert len(pos) == len(vals) == 0


def test_rolling_alignment_no_lookahead():
    rng = np.random.default_rng(3)
    dates = pd.bdate_range("2020-01-01", periods=600)
    prices = pd.DataFrame(
        {"ticker": "TEST", "date": dates, "price": 100 * np.exp(np.cumsum(rng.standard_normal(600) * 0.01))}
    )
    out = rh.compute_rolling_hurst(prices, windows=(252,), step=1, verbose=False)
    # first possible 252-return window ends on price-row 252 (0-based): 253rd date
    assert out["date"].iloc[0] == dates[252]
    changing_future = prices.copy()
    changing_future.loc[400:, "price"] *= 5.0  # corrupt the future...
    out2 = rh.compute_rolling_hurst(changing_future, windows=(252,), step=1, verbose=False)
    before = out[out["date"] <= dates[399]]["hurst_252"]
    after = out2[out2["date"] <= dates[399]]["hurst_252"]
    pd.testing.assert_series_equal(before.reset_index(drop=True), after.reset_index(drop=True))


def test_windows_share_a_common_date_grid():
    """Regression test: 126/252/504 grids must align on the same dates.

    Anchoring each grid at its own window (126, 252, 504 mod step all
    differ) produced disjoint dates per column, an outer-merge full of
    NaNs, and time-series plots that silently stopped after year one.
    """
    rng = np.random.default_rng(11)
    dates = pd.bdate_range("2018-01-01", periods=1300)
    prices = pd.DataFrame(
        {"ticker": "TEST", "date": dates, "price": 100 * np.exp(np.cumsum(rng.standard_normal(1300) * 0.01))}
    )
    out = rh.compute_rolling_hurst(prices, windows=(126, 252, 504), step=5, verbose=False)
    # the 126 grid is the earliest-starting superset of dates
    assert len(out) == out["hurst_126"].notna().sum()
    # once the 504 window exists, every row carries all three windows
    later = out[out["hurst_504"].notna()]
    assert len(later) > 0 and later[["hurst_126", "hurst_252"]].notna().all().all()


def test_numba_and_numpy_paths_agree():
    if not rh.HAS_NUMBA:
        pytest.skip("numba not installed")
    rng = np.random.default_rng(5)
    x = rng.standard_normal(800)
    scales = rh.make_scales(252)
    pos, fast = rh.rolling_hurst(x, window=252, step=25)
    slow = np.array([rh.hurst_rs(x[i - 251 : i + 1], scales=scales) for i in pos])
    np.testing.assert_allclose(fast, slow, rtol=1e-6, atol=1e-8)


def test_duckdb_roundtrip_with_messy_columns(tmp_path):
    duckdb = pytest.importorskip("duckdb")
    rng = np.random.default_rng(9)
    dates = pd.bdate_range("2021-01-01", periods=600)
    rows = []
    for t in ("AAA", "ZZZ"):
        close = 50 * np.exp(np.cumsum(rng.standard_normal(600) * 0.01))
        rows.append(pd.DataFrame({"Symbol": t, "Date": dates, "Adj Close": close, "Close": close * 1.01}))
    df = pd.concat(rows)
    db = tmp_path / "messy.duckdb"
    con = duckdb.connect(str(db))
    con.register("tmp_df", df)
    con.execute('CREATE TABLE daily_bars AS SELECT Symbol, CAST(Date AS DATE) AS Date, "Adj Close", "Close" FROM tmp_df')
    con.close()

    schema = rh.discover_price_schema(db)
    assert schema.table == "daily_bars" and schema.ticker_col == "Symbol"
    assert schema.price_col == "Adj Close"  # adjusted close must win over raw close

    prices = rh.load_prices(db)
    assert list(prices.columns) == ["ticker", "date", "price"]
    assert set(prices["ticker"]) == {"AAA", "ZZZ"}
    out = rh.compute_rolling_hurst(prices, windows=(126, 252), step=21, verbose=False)
    assert {"ticker", "date", "hurst_126", "hurst_252"} <= set(out.columns)
    assert out["hurst_252"].dropna().between(0, 1.2).all()
