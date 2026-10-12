"""Cálculo de correlaciones: matrices por ventana, rolling y red de correlación."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

from .config import Window, resolve_windows


# ---------------------------------------------------------------------------
# Retornos
# ---------------------------------------------------------------------------
def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Retornos logarítmicos diarios (más estables para análisis estadístico)."""
    return np.log(prices.astype(float)).diff().dropna(how="all")


def simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Retornos simples diarios."""
    return prices.astype(float).pct_change().dropna(how="all")


# ---------------------------------------------------------------------------
# Ventanas
# ---------------------------------------------------------------------------
def window_slice(index: pd.DatetimeIndex, window, asof=None) -> pd.DatetimeIndex:
    """Devuelve el sub-índice de fechas que cae dentro de la ventana."""
    if not isinstance(window, Window):
        window = resolve_windows([window])[str(window).lower()]
    start, end = window.bounds(index, asof)
    return index[(index >= start) & (index <= end)]


# ---------------------------------------------------------------------------
# Matrices de correlación
# ---------------------------------------------------------------------------
def correlation_matrix(
    returns: pd.DataFrame,
    method: str = "pearson",
    min_periods: int = 10,
) -> pd.DataFrame:
    """Matriz de correlación de retornos.

    method: 'pearson' (lineal) o 'spearman' (rangos, robusta a outliers).
    """
    if method not in ("pearson", "spearman"):
        raise ValueError("method debe ser 'pearson' o 'spearman'")
    return returns.corr(method=method, min_periods=min_periods)


def correlation_matrices(
    prices: pd.DataFrame,
    windows: Optional[Iterable] = None,
    methods: Iterable[str] = ("pearson", "spearman"),
    asof=None,
) -> Dict[Tuple[str, str], pd.DataFrame]:
    """Calcula las matrices de correlación para cada ventana y método.

    Returns
    -------
    dict {(ventana, metodo): DataFrame}
    """
    windows = resolve_windows(windows)
    rets = log_returns(prices)
    out: Dict[Tuple[str, str], pd.DataFrame] = {}
    for label, win in windows.items():
        idx = window_slice(prices.index, win, asof)
        sub = rets.loc[rets.index.isin(idx)]
        nan = pd.DataFrame(np.nan, index=rets.columns, columns=rets.columns)
        if sub.dropna(how="all").shape[0] < 5:
            for m in methods:  # datos insuficientes
                out[(label, m)] = nan.copy()
            continue
        for m in methods:
            out[(label, m)] = correlation_matrix(sub, method=m)
    return out


# ---------------------------------------------------------------------------
# Pares más/menos correlacionados
# ---------------------------------------------------------------------------
def top_pairs(
    corr: pd.DataFrame,
    n: int = 10,
) -> pd.DataFrame:
    """Ranking de pares ordenados por |correlación| descendente."""
    cols = corr.columns
    rows = []
    for i, a in enumerate(cols):
        for j, b in enumerate(cols):
            if i >= j:
                continue
            val = corr.iloc[i, j]
            if pd.isna(val):
                continue
            rows.append({"A": a, "B": b, "corr": float(val), "abs_corr": abs(float(val))})
    df = pd.DataFrame(rows).sort_values("abs_corr", ascending=False).reset_index(drop=True)
    return df.head(n)


# ---------------------------------------------------------------------------
# Correlación rolling
# ---------------------------------------------------------------------------
def rolling_correlation(
    returns: pd.DataFrame,
    a: str,
    b: str,
    window: int = 63,
) -> pd.Series:
    """Serie de correlación móvil (por defecto 63 días ≈ 3 meses)."""
    return returns[a].rolling(window).corr(returns[b]).dropna()


def average_rolling_correlation(
    returns: pd.DataFrame,
    window: int = 63,
) -> pd.Series:
    """Correlación media del universo (promedio de todos los pares) en ventana móvil."""
    roll = returns.rolling(window).corr()
    tickers = list(returns.columns)
    mask = ~np.eye(len(tickers), dtype=bool)
    avg = {}
    for date, mat in roll.groupby(level=0):
        if mat.isna().all().all():
            avg[date] = np.nan
            continue
        avg[date] = np.nanmean(mat.values[mask])
    s = pd.Series(avg).sort_index()
    s.name = f"corr_media_{window}d"
    return s


# ---------------------------------------------------------------------------
# Red de correlación
# ---------------------------------------------------------------------------
def correlation_network(
    corr: pd.DataFrame,
    threshold: float = 0.6,
    use_abs: bool = False,
) -> pd.DataFrame:
    """Aristas de la red de correlación: pares cuyo |corr| supera el umbral."""
    rows = []
    cols = corr.columns
    for i, a in enumerate(cols):
        for j, b in enumerate(cols):
            if i >= j:
                continue
            val = corr.iloc[i, j]
            if pd.isna(val):
                continue
            keep = abs(val) >= threshold if use_abs else val >= threshold
            if keep:
                rows.append({"A": a, "B": b, "corr": float(val)})
    return pd.DataFrame(rows, columns=["A", "B", "corr"])


# ---------------------------------------------------------------------------
# Orden jerárquico para heatmaps
# ---------------------------------------------------------------------------
def distance_matrix(corr: pd.DataFrame) -> pd.DataFrame:
    """Matriz de distancia 1 - |corr| (simétrica, diagonal 0).

    Devuelve siempre una copia propia: pandas puede exponer arrays de sólo
    lectura y scipy/numpy rechazan modificarlos in-place.
    """
    valid = corr.dropna(how="all")
    valid = valid.loc[:, valid.notna().all()]
    d = 1 - valid.abs()
    arr = np.array(d.to_numpy(dtype=float), copy=True)
    arr = (arr + arr.T) / 2.0
    np.fill_diagonal(arr, 0.0)
    return pd.DataFrame(arr, index=valid.index, columns=valid.columns)


def _dist_array(corr: pd.DataFrame):
    """(columnas, array de distancia) listos para scipy/sklearn."""
    dist = distance_matrix(corr)
    return list(dist.columns), np.array(dist.to_numpy(dtype=float), copy=True)


def clustered_order(corr: pd.DataFrame) -> List[str]:
    """Orden de ETFs por clustering jerárquico sobre distancia 1-|corr|."""
    valid = corr.dropna(how="all")
    valid = valid.loc[:, valid.notna().all()]
    if valid.shape[0] < 3:
        return list(corr.columns)
    cols, arr = _dist_array(valid)
    try:
        z = linkage(squareform(arr, checks=False), method="average")
        order = [cols[i] for i in leaves_list(z)]
    except Exception:
        order = list(cols)
    order += [c for c in corr.columns if c not in order]
    return order


# ---------------------------------------------------------------------------
# Significancia estadística
# ---------------------------------------------------------------------------
def pair_pvalues(returns: pd.DataFrame) -> pd.DataFrame:
    """p-values de la correlación de Pearson para cada par (test bivariante)."""
    cols = list(returns.columns)
    pv = pd.DataFrame(np.nan, index=cols, columns=cols)
    data = returns.dropna()
    if len(data) < 3:
        return pv
    for i, a in enumerate(cols):
        for j, b in enumerate(cols):
            if i >= j:
                continue
            _, p = stats.pearsonr(data[a], data[b])
            pv.loc[a, b] = pv.loc[b, a] = p
    np.fill_diagonal(pv.values, 0.0)
    return pv
