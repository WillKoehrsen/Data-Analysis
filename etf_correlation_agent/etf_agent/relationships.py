"""Búsqueda de relaciones entre ETFs más allá de la correlación lineal.

Incluye:
  * lead-lag (correlación cruzada con retardos)
  * causalidad de Granger
  * cointegración (Engle-Granger) + vida media + z-score del spread
  * betas / exposición factorial (regresión vs factores)
  * componentes principales (factores latentes)
  * clustering jerárquico de ETFs
  * información mutua (dependencia no lineal)
  * correlación en regímenes de estrés (tail) y en drawdowns
  * estadísticos de volatilidad
"""

from __future__ import annotations

import contextlib
import io
from contextlib import contextmanager
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import AgglomerativeClustering

from .correlation import (
    _dist_array,
    clustered_order,
    correlation_matrix,
    log_returns,
)


# ---------------------------------------------------------------------------
# Utilidad: periodo común (evita que un ETF de corta vida recorte el análisis)
# ---------------------------------------------------------------------------
def datos_comunes(returns: pd.DataFrame, nombre: str = "análisis") -> pd.DataFrame:
    """Restringe el DataFrame a las filas donde **todos** los tickers cotizan.

    Los ETFs con poco histórico (p. ej. DRAM) recortarían en silencio el periodo
    de los análisis multivariante (PCA, betas, Granger, cointegración...).
    Aquí se avisa cuando el recorte es relevante.
    """
    completo = returns.dropna()
    if 0 < len(completo) < len(returns):
        perdidas = len(returns) - len(completo)
        pct = perdidas / len(returns) * 100
        if pct > 1.0:
            print(
                f"[etf_agent] AVISO ({nombre}): se usan {len(completo)} de {len(returns)} "
                f"sesiones ({pct:.0f}% descartadas) porque algún ETF aún no cotizaba."
            )
    return completo


# ---------------------------------------------------------------------------
# 1. Lead-lag: ¿quién va por delante de quién?
# ---------------------------------------------------------------------------
def cross_correlation(
    returns: pd.DataFrame,
    a: str,
    b: str,
    max_lag: int = 10,
) -> pd.Series:
    """Correlación entre X_t y Y_{t+k} para k en [-max_lag, max_lag].

    Interpretación: k > 0  →  ``a`` va POR DELANTE de ``b`` (lidera).
                    k < 0  →  ``b`` va por delante de ``a``.
                    k = 0  →  movimientos contemporáneos.
    """
    out = {}
    for k in range(-max_lag, max_lag + 1):
        # a_t vs b_{t+k}: desplazamos b -k posiciones
        out[k] = returns[a].corr(returns[b].shift(-k))
    return pd.Series(out).rename(f"ccf({a},{b})")


def lead_lag_table(
    returns: pd.DataFrame,
    max_lag: int = 10,
    min_abs_corr: float = 0.10,
    top: Optional[int] = None,
) -> pd.DataFrame:
    """Para cada par ordenado, el rezago con mayor |correlación| (≠ 0).

    Devuelve columnas: leader, follower, best_lag, best_corr, corr_lag0, n.
    Un ``best_lag`` positivo significa que ``leader`` anticipa a ``follower``.
    """
    tickers = list(returns.columns)
    rows = []
    for i, a in enumerate(tickers):
        for b in tickers[i + 1:]:
            ccf = cross_correlation(returns, a, b, max_lag=max_lag)
            nz = ccf.drop(index=0)
            if nz.empty:
                continue
            k = int(nz.abs().idxmax())
            best = float(nz.loc[k])
            if abs(best) < min_abs_corr:
                continue
            leader, follower = (a, b) if k > 0 else (b, a)
            rows.append(
                {
                    "leader": leader,
                    "follower": follower,
                    "best_lag_days": abs(k),
                    "best_lag_signed": k,
                    "best_corr": best,
                    "corr_lag0": float(ccf.loc[0]),
                    "n_obs": int(returns[[a, b]].dropna().shape[0]),
                }
            )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.sort_values("best_corr", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)
    if top:
        df = df.head(top)
    return df


# ---------------------------------------------------------------------------
# 2. Causalidad de Granger
# ---------------------------------------------------------------------------
@contextmanager
def _suppress_stdout():
    """Silencia la salida estándar (statsmodels imprime por defecto)."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        yield


def granger_table(
    returns: pd.DataFrame,
    max_lag: int = 5,
    alpha: float = 0.05,
    tickers: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Test de causalidad de Granger sobre retornos para todos los pares ordenados.

    H0: ``cause`` NO Granger-causa ``effect`` (no aporta información predictiva).
    Devuelve el p-value mínimo entre 1..max_lag rezagos y el rezago óptimo.
    """
    from statsmodels.tsa.stattools import grangercausalitytests

    tickers = list(tickers or returns.columns)
    data = datos_comunes(returns, "Granger")
    rows = []
    for cause in tickers:
        for effect in tickers:
            if cause == effect or effect not in data or cause not in data:
                continue
            pair = data[[effect, cause]].values
            try:
                # statsmodels >= 0.15 eliminó el argumento `verbose`
                try:
                    with _suppress_stdout():
                        res = grangercausalitytests(pair, maxlag=max_lag, verbose=False)
                except TypeError:
                    with _suppress_stdout():
                        res = grangercausalitytests(pair, maxlag=max_lag)
            except Exception:
                continue
            best_lag, best_p = None, np.inf
            for lag, tests in res.items():
                p = tests[0]["ssr_ftest"][1]
                if p < best_p:
                    best_p, best_lag = p, lag
            rows.append(
                {
                    "cause": cause,
                    "effect": effect,
                    "best_lag": int(best_lag),
                    "p_value": float(best_p),
                    "significativo": bool(best_p < alpha),
                }
            )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values("p_value").reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3. Cointegración (Engle-Granger)
# ---------------------------------------------------------------------------
def _half_life(spread: pd.Series) -> float:
    """Vida media de reversión a la media vía regresión AR(1)."""
    s = spread.dropna()
    if len(s) < 20:
        return np.nan
    ds = s.diff().dropna()
    lag = s.shift(1).dropna()
    lag = lag.loc[ds.index]
    if lag.empty:
        return np.nan
    b = np.polyfit(lag.values, ds.values, 1)[0]
    if b >= 0:
        return np.nan  # no revierte
    return float(-np.log(2) / b)


def cointegration_table(
    prices: pd.DataFrame,
    alpha: float = 0.05,
    tickers: Optional[Sequence[str]] = None,
    zscore_window: Optional[int] = None,
) -> pd.DataFrame:
    """Test de cointegración de Engle-Granger sobre log-precios, para cada par.

    Columnas: hedge_ratio (beta de la regresión de cointegración),
    adf_pvalue (p-value del test de estacionariedad del spread),
    cointegrada (bool), half_life_days, zscore (del spread, al final),
    corr_logprices.
    """
    import warnings

    from statsmodels.tsa.stattools import coint, adfuller

    tickers = list(tickers or prices.columns)
    logp = datos_comunes(np.log(prices[tickers].astype(float)), "cointegración")
    rows = []
    for i, a in enumerate(tickers):
        for b in tickers[i + 1:]:
            ya, yb = logp[a], logp[b]
            if len(logp) < 60:
                continue
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", FutureWarning)
                    _, pval, _ = coint(ya, yb)
            except Exception:
                continue
            # hedge ratio: log(a) = c + h * log(b)
            h = np.polyfit(yb.values, ya.values, 1)[0]
            spread = ya - h * yb
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", FutureWarning)
                    adf_p = float(adfuller(spread, autolag="AIC")[1])
            except Exception:
                adf_p = np.nan
            win = zscore_window or len(spread)
            recent = spread.iloc[-win:]
            z = float((spread.iloc[-1] - recent.mean()) / recent.std()) if recent.std() > 0 else np.nan
            rows.append(
                {
                    "A": a,
                    "B": b,
                    "hedge_ratio": float(h),
                    "coint_pvalue": float(pval),
                    "cointegrada": bool(pval < alpha),
                    "adf_pvalue_spread": adf_p,
                    "half_life_days": _half_life(spread),
                    "spread_zscore": z,
                    "corr_logprecios": float(logp[a].corr(logp[b])),
                }
            )
    return pd.DataFrame(rows).sort_values("coint_pvalue").reset_index(drop=True)


def spread_series(prices: pd.DataFrame, a: str, b: str, hedge_ratio: Optional[float] = None) -> pd.Series:
    """Spread log(a) - h*log(b); h se estima por OLS si no se indica."""
    logp = np.log(prices[[a, b]].astype(float)).dropna()
    h = hedge_ratio
    if h is None:
        h = float(np.polyfit(logp[b].values, logp[a].values, 1)[0])
    return (logp[a] - h * logp[b]).rename(f"spread({a},{b})")


# ---------------------------------------------------------------------------
# 4. Betas / exposición factorial
# ---------------------------------------------------------------------------
def factor_betas(
    returns: pd.DataFrame,
    factors: Sequence[str] = ("SPY", "TLT", "GLD"),
    tickers: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Regresión OLS de cada ETF sobre los factores indicados.

    Devuelve alpha anualizado, beta por factor, R² y R² ajustado.
    """
    tickers = [t for t in (tickers or returns.columns) if t not in factors]
    data = datos_comunes(returns, "betas factoriales")
    rows = []
    for t in tickers:
        y = data[t].values
        X = np.column_stack([np.ones(len(data))] + [data[f].values for f in factors])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        yhat = X @ beta
        ss_res = float(((y - yhat) ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum())
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        n, k = X.shape
        r2_adj = 1 - (1 - r2) * (n - 1) / max(n - k, 1)
        row = {
            "etf": t,
            "alpha_diario": float(beta[0]),
            "alpha_anualizado_pct": float(beta[0] * 252 * 100),
            "r2": r2,
            "r2_adj": r2_adj,
        }
        for i, f in enumerate(factors):
            row[f"beta_{f}"] = float(beta[i + 1])
        rows.append(row)
    return pd.DataFrame(rows)


def rolling_beta(
    returns: pd.DataFrame,
    asset: str,
    benchmark: str = "SPY",
    window: int = 63,
) -> pd.Series:
    """Beta móvil de un ETF respecto a un benchmark."""
    cov = returns[asset].rolling(window).cov(returns[benchmark])
    var = returns[benchmark].rolling(window).var()
    return (cov / var).dropna().rename(f"beta_{asset}_vs_{benchmark}_{window}d")


# ---------------------------------------------------------------------------
# 5. Componentes principales (factores latentes)
# ---------------------------------------------------------------------------
def pca_factors(returns: pd.DataFrame, n_components: int = 3) -> Dict[str, object]:
    """PCA sobre los retornos estandarizados.

    Devuelve diccionario con:
      explained_variance_ratio, loadings (DataFrame componente x ETF),
      scores (serie temporal de los componentes), n_components.
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    data = datos_comunes(returns, "PCA")
    n_components = int(min(n_components, data.shape[1], data.shape[0]))
    X = StandardScaler().fit_transform(data.values)
    model = PCA(n_components=n_components, random_state=0)
    scores = model.fit_transform(X)
    loadings = pd.DataFrame(
        model.components_.T,
        index=data.columns,
        columns=[f"PC{i + 1}" for i in range(n_components)],
    )
    return {
        "explained_variance_ratio": pd.Series(
            model.explained_variance_ratio_, index=[f"PC{i + 1}" for i in range(n_components)]
        ),
        "loadings": loadings,
        "scores": pd.DataFrame(
            scores, index=data.index, columns=[f"PC{i + 1}" for i in range(n_components)]
        ),
        "n_components": n_components,
    }


# ---------------------------------------------------------------------------
# 6. Clustering de ETFs
# ---------------------------------------------------------------------------
def cluster_etfs(
    corr: pd.DataFrame,
    n_clusters: int = 4,
) -> pd.DataFrame:
    """Clustering jerárquico (distancia = 1 - |corr|) sobre una matriz de correlación."""
    valid = corr.dropna(how="all")
    valid = valid.loc[:, valid.notna().all()]
    if valid.shape[0] < 2:
        return pd.DataFrame({"etf": list(corr.columns), "cluster": 1})
    cols, arr = _dist_array(valid)
    k = int(min(n_clusters, len(cols)))
    labels = AgglomerativeClustering(
        n_clusters=k, metric="precomputed", linkage="average"
    ).fit_predict(arr)
    df = pd.DataFrame({"etf": list(valid.columns), "cluster": labels + 1})
    missing = [c for c in corr.columns if c not in set(df["etf"])]
    if missing:
        df = pd.concat([df, pd.DataFrame({"etf": missing, "cluster": k + 1})], ignore_index=True)
    return df.sort_values(["cluster", "etf"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 7. Información mutua (dependencia no lineal)
# ---------------------------------------------------------------------------
def mutual_information_matrix(
    returns: pd.DataFrame,
    bins: int = 8,
    tickers: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Matriz de información mutua (en nats) entre retornos discretizados.

    A diferencia de la correlación, detecta dependencias NO lineales.
    """
    from sklearn.metrics import mutual_info_score

    tickers = list(tickers or returns.columns)
    data = datos_comunes(returns[tickers], "información mutua")
    disc = data.apply(lambda s: pd.qcut(s.rank(method="first"), bins, labels=False))
    mi = pd.DataFrame(0.0, index=tickers, columns=tickers)
    for i, a in enumerate(tickers):
        for j, b in enumerate(tickers):
            if i > j:
                mi.loc[a, b] = mi.loc[b, a]
                continue
            mi.loc[a, b] = float(mutual_info_score(disc[a], disc[b]))
    return mi


# ---------------------------------------------------------------------------
# 8. Regímenes de estrés / colas
# ---------------------------------------------------------------------------
def regime_correlation(
    returns: pd.DataFrame,
    benchmark: str = "SPY",
    quantile: float = 0.10,
    tickers: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Compara la correlación de cada ETF con el benchmark en:

      * días 'malos' del benchmark (peor cuantil)
      * días 'buenos' del benchmark (mejor cuantil)
      * toda la muestra

    Útil para detectar asimetrías (suben más las correlaciones en estrés).
    """
    tickers = [t for t in (tickers or returns.columns)]
    data = datos_comunes(returns, "regímenes de estrés")
    if benchmark not in data.columns:
        raise ValueError(f"El benchmark {benchmark} no está en los datos.")
    thr_lo = data[benchmark].quantile(quantile)
    thr_hi = data[benchmark].quantile(1 - quantile)
    bad = data[data[benchmark] <= thr_lo]
    good = data[data[benchmark] >= thr_hi]
    rows = []
    for t in tickers:
        rows.append(
            {
                "etf": t,
                "corr_all": float(data[t].corr(data[benchmark])),
                "corr_bad_days": float(bad[t].corr(bad[benchmark])) if len(bad) > 2 else np.nan,
                "corr_good_days": float(good[t].corr(good[benchmark])) if len(good) > 2 else np.nan,
                "ret_bad_days_pct": float(bad[t].mean() * 100) if len(bad) else np.nan,
                "ret_good_days_pct": float(good[t].mean() * 100) if len(good) else np.nan,
            }
        )
    df = pd.DataFrame(rows)
    df["asimetria"] = df["corr_bad_days"] - df["corr_good_days"]
    return df.sort_values("asimetria", ascending=False).reset_index(drop=True)


def drawdown_episodes(prices: pd.DataFrame, benchmark: str = "SPY", threshold: float = -0.10) -> pd.DataFrame:
    """Episodios de drawdown del benchmark por debajo de ``threshold``."""
    s = prices[benchmark].dropna()
    dd = s / s.cummax() - 1
    episodes, start = [], None
    for date, val in dd.items():
        if val <= threshold and start is None:
            start = date
        elif val > threshold and start is not None:
            episodes.append((start, date, float(dd.loc[start:date].min())))
            start = None
    if start is not None:
        episodes.append((start, dd.index[-1], float(dd.loc[start:].min())))
    return pd.DataFrame(episodes, columns=["inicio", "fin", "drawdown_max"])


# ---------------------------------------------------------------------------
# 9. Estadísticos de volatilidad / riesgo
# ---------------------------------------------------------------------------
def volatility_table(
    returns: pd.DataFrame,
    periods_per_year: int = 252,
    tickers: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Volatilidad anualizada, asimetría, curtosis y máx drawdown (sobre precios)."""
    tickers = list(tickers or returns.columns)
    data = returns[tickers]
    rows = []
    for t in tickers:
        r = data[t].dropna()
        if r.empty:
            continue
        rows.append(
            {
                "etf": t,
                "ret_medio_diario_pct": float(r.mean() * 100),
                "vol_anualizada_pct": float(r.std(ddof=1) * np.sqrt(periods_per_year) * 100),
                "skew": float(stats.skew(r)),
                "kurtosis": float(stats.kurtosis(r)),
                "sharpe_aproximado": float(r.mean() / r.std(ddof=1) * np.sqrt(periods_per_year)) if r.std(ddof=1) > 0 else np.nan,
                "n_obs": int(len(r)),
            }
        )
    return pd.DataFrame(rows)


def max_drawdown(prices: pd.DataFrame, tickers: Optional[Sequence[str]] = None) -> pd.Series:
    """Máximo drawdown por ETF (en %)."""
    tickers = list(tickers or prices.columns)
    out = {}
    for t in tickers:
        s = prices[t].dropna()
        if s.empty:
            out[t] = np.nan
            continue
        dd = (s / s.cummax() - 1).min()
        out[t] = float(dd * 100)
    return pd.Series(out).rename("max_drawdown_pct")
