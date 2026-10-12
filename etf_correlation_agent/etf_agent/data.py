"""Descarga, cache y validación de precios históricos de ETFs."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

# Directorio de cache por defecto (relativo al cwd)
DEFAULT_CACHE_DIR = Path("data") / "etf_cache"


# ---------------------------------------------------------------------------
# Utilidades de normalización
# ---------------------------------------------------------------------------
# Campos de precio que yfinance puede devolver (se buscan en minúsculas)
PRICE_FIELDS = {
    "open", "high", "low", "close", "adj close", "volume",
    "capital gains", "dividends", "stock splits",
}


def _pick_price_field(fields_lower: List[str]) -> Optional[str]:
    """Elige el campo de precio preferido: 'Adj Close' si existe, si no 'Close'."""
    for candidate in ("adj close", "close"):
        if candidate in fields_lower:
            return candidate
    return None


def _normalize_columns(raw: pd.DataFrame, tickers: List[str]) -> pd.DataFrame:
    """Convierte la salida de yfinance (cualquier versión/formato) en un DataFrame
    con índice=fecha, columnas=ticker y valores=precio de cierre ajustado.

    Formatos soportados:
      * MultiIndex (Ticker, Price)   -> yfinance nuevo con group_by='ticker'
      * MultiIndex (Price, Ticker)   -> yfinance con group_by='column'
      * Columnas planas (un solo ticker)
    """
    if raw is None or getattr(raw, "empty", True):
        raise ValueError("La descarga no devolvió datos.")

    if isinstance(raw.columns, pd.MultiIndex):
        lvl0 = {str(c).lower() for c in raw.columns.get_level_values(0).unique()}
        lvl1 = {str(c).lower() for c in raw.columns.get_level_values(1).unique()}
        tiene0, tiene1 = lvl0 & PRICE_FIELDS, lvl1 & PRICE_FIELDS
        if tiene1 and not tiene0:
            field_level, ticker_level = 1, 0      # (Ticker, Price)
        elif tiene0 and not tiene1:
            field_level, ticker_level = 0, 1      # (Price, Ticker)
        else:
            raise ValueError(f"Columnas inesperadas en la descarga: {raw.columns[:5]}")
        fields = [str(c).lower() for c in raw.columns.get_level_values(field_level)]
        field = _pick_price_field(fields)
        if field is None:
            raise ValueError(f"Columnas inesperadas en la descarga: {raw.columns[:5]}")
        mask = [f == field for f in fields]
        sub = raw.loc[:, mask]
        sub.columns = [str(c) for c in sub.columns.get_level_values(ticker_level)]
        out = sub.copy()
    else:
        cols = {str(c).lower(): c for c in raw.columns}
        field = _pick_price_field(list(cols))
        if field is None:
            raise ValueError(f"Columnas inesperadas en la descarga: {list(raw.columns)[:5]}")
        out = raw[[cols[field]]].copy()
        out.columns = [str(tickers[0]).upper()]

    out.index = pd.to_datetime(out.index)
    out = out[~out.index.duplicated(keep="last")].sort_index()
    out = out.apply(pd.to_numeric, errors="coerce")
    out = out.dropna(how="all")
    # nombres de columna en mayúsculas para uniformar
    out.columns = [str(c).upper() for c in out.columns]
    return out


def _download_yfinance(tickers: List[str], start=None, end=None, retries: int = 2) -> pd.DataFrame:
    """Descarga con yfinance.

    1) Intenta la descarga conjunta (normalizando el formato que devuelva la versión).
    2) Si falla, descarga ticker a ticker con ``.history()`` (muy robusto).
    """
    import yfinance as yf

    last_err: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            raw = yf.download(
                tickers=" ".join(tickers),
                start=start,
                end=end,
                auto_adjust=True,
                progress=False,
                group_by="ticker",
                threads=True,
            )
            return _normalize_columns(raw, tickers)
        except Exception as err:  # depende de red / versión de yfinance
            last_err = err
            time.sleep(1.5 * (attempt + 1))

    # Fallback: un ticker cada vez con la API .history()
    if retries >= 0:
        print(f"[etf_agent] Reintentando ticker a ticker ({len(tickers)} ETFs)...")
    frames: Dict[str, pd.Series] = {}
    for t in tickers:
        for attempt in range(retries + 1):
            try:
                if start is None:
                    hist = yf.Ticker(t).history(period="max", end=end, auto_adjust=True)
                else:
                    hist = yf.Ticker(t).history(start=start, end=end, auto_adjust=True)
                if hist is not None and not hist.empty and "Close" in hist.columns:
                    frames[t] = pd.to_numeric(hist["Close"], errors="coerce").rename(t)
                break
            except Exception as err:
                last_err = err
                time.sleep(1.0 * (attempt + 1))
    if frames:
        out = pd.concat(frames.values(), axis=1)
        out.index = pd.to_datetime(out.index)
        out = out[~out.index.duplicated(keep="last")].sort_index().dropna(how="all")
        out.columns = [str(c).upper() for c in out.columns]
        return out
    raise RuntimeError(f"No se pudo descargar de Yahoo Finance: {last_err}")


def _download_stooq(tickers: List[str], start=None, end=None) -> pd.DataFrame:
    """Fallback: descarga desde Stooq vía pandas-datareader."""
    try:
        import pandas_datareader.data as web
    except Exception as err:  # pragma: no cover
        raise RuntimeError(f"pandas-datareader no disponible: {err}")

    if start is None:
        start = "1993-01-01"  # Stooq necesita fecha inicial
    frames = []
    for t in tickers:
        for attempt in range(2):
            try:
                df = web.DataReader(t.lower() + ".us", "stooq", start=start, end=end)
                if df is not None and not df.empty and "Close" in df.columns:
                    frames.append(df[["Close"]].rename(columns={"Close": t}))
                break
            except Exception:
                time.sleep(1.0)
    if not frames:
        raise RuntimeError("La descarga desde Stooq tampoco devolvió datos.")
    out = pd.concat(frames, axis=1)
    out.index = pd.to_datetime(out.index)
    out = out[~out.index.duplicated(keep="last")].sort_index().dropna(how="all")
    out.columns = [str(c).upper() for c in out.columns]
    return out


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------
def fetch_prices(
    tickers: Optional[Iterable[str]] = None,
    start=None,
    end=None,
    cache_dir=DEFAULT_CACHE_DIR,
    use_cache: bool = True,
    source: str = "yfinance",
    verbose: bool = True,
) -> pd.DataFrame:
    """Descarga precios ajustados de cierre para los tickers indicados.

    Parameters
    ----------
    tickers : lista de símbolos (por defecto DEFAULT_TICKERS).
    start, end : fechas (str 'YYYY-MM-DD' o datetime). Si start es None se
        descarga el máximo histórico disponible.
    cache_dir : carpeta donde se guarda un CSV por ticker.
    use_cache : si True, reutiliza CSVs previos y solo descarga lo que falta.
    source : 'yfinance' (por defecto) o 'stooq'.
    verbose : imprime progreso.

    Returns
    -------
    DataFrame índice=fecha, columnas=ticker, valores=precio ajustado.
    """
    tickers = [t.strip().upper() for t in (tickers or [])]
    if not tickers:
        from .config import DEFAULT_TICKERS

        tickers = list(DEFAULT_TICKERS)

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    cached: Dict[str, pd.Series] = {}
    missing: List[str] = []
    if use_cache:
        for t in tickers:
            path = cache_dir / f"{t}.csv"
            if path.exists():
                df = pd.read_csv(path, index_col=0, parse_dates=True)
                if not df.empty:
                    cached[t] = df.iloc[:, 0].rename(t)
                    continue
            missing.append(t)
    else:
        missing = list(tickers)

    fresh = pd.DataFrame()
    if missing:
        if verbose:
            print(f"[etf_agent] Descargando {len(missing)} tickers desde {source}: {', '.join(missing)}")
        if source == "stooq":
            fresh = _download_stooq(missing, start=start, end=end)
        else:
            try:
                fresh = _download_yfinance(missing, start=start, end=end)
            except Exception as err:
                if verbose:
                    print(f"[etf_agent] Fallo Yahoo Finance ({err}); probando Stooq...")
                fresh = _download_stooq(missing, start=start, end=end)
        # Guardar en cache
        for t in fresh.columns:
            try:
                fresh[[t]].to_csv(cache_dir / f"{t}.csv")
            except Exception:
                pass

    parts = [s for s in cached.values()]
    if not fresh.empty:
        parts.extend([fresh[c].rename(c) for c in fresh.columns])
    if not parts:
        raise RuntimeError("No se obtuvieron datos para ningún ticker.")

    prices = pd.concat(parts, axis=1)
    prices = prices.loc[:, ~prices.columns.duplicated()]
    prices = prices.sort_index()
    if start is not None:
        prices = prices.loc[prices.index >= pd.Timestamp(start)]
    if end is not None:
        prices = prices.loc[prices.index <= pd.Timestamp(end)]
    prices = prices.dropna(axis=1, how="all")

    failed = [t for t in tickers if t not in prices.columns]
    if failed and verbose:
        print(f"[etf_agent] AVISO: sin datos para {failed} (se excluyen del análisis).")
    if prices.empty:
        raise RuntimeError("Tras combinar cache y descargas no quedaron datos.")
    return prices


def load_cached_prices(tickers: Optional[Iterable[str]] = None, cache_dir=DEFAULT_CACHE_DIR) -> pd.DataFrame:
    """Carga precios solo desde el cache local (sin red)."""
    tickers = [t.strip().upper() for t in (tickers or [])] or None
    if tickers is None:
        from .config import DEFAULT_TICKERS

        tickers = list(DEFAULT_TICKERS)
    cache_dir = Path(cache_dir)
    series = []
    for t in tickers:
        path = cache_dir / f"{t}.csv"
        if path.exists():
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            if not df.empty:
                series.append(df.iloc[:, 0].rename(t))
    if not series:
        raise FileNotFoundError(f"No hay cache en {cache_dir}. Ejecuta fetch_prices() primero.")
    return pd.concat(series, axis=1).sort_index()


# ---------------------------------------------------------------------------
# Datos sintéticos (modo demo, sin necesidad de internet)
# ---------------------------------------------------------------------------
def make_demo_prices(
    tickers: Optional[Iterable[str]] = None,
    start: str = "2015-01-01",
    end: Optional[str] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Genera precios sintéticos con estructura realista para pruebas/demos.

    Se construye un modelo factorial latente (factor de mercado, factor de
    tipos, factor de materias primas y factores sectoriales) más ruido
    idiosincrático, de modo que existan correlaciones, relaciones lead-lag y
    cointegración de mentira que el agente puede descubrir.
    """
    tickers = [t.strip().upper() for t in (tickers or [])]
    if not tickers:
        from .config import DEFAULT_TICKERS

        tickers = list(DEFAULT_TICKERS)

    rng = np.random.default_rng(seed)
    end = pd.Timestamp(end or pd.Timestamp.today().normalize())
    dates = pd.bdate_range(start=start, end=end)
    n = len(dates)

    # Factores latentes diarios
    market = rng.normal(0.0004, 0.010, n)
    rates = rng.normal(0.0000, 0.005, n)
    commodity = rng.normal(0.0001, 0.012, n)
    credit = rng.normal(0.0002, 0.004, n)
    semi = rng.normal(0.0005, 0.015, n)
    cash = np.full(n, 0.00012) + rng.normal(0, 0.00005, n)
    dollar = rng.normal(0.0000, 0.005, n)

    # Cargas factoriales por ETF:
    # (market, rates, commodity, credit, semi, cash, dollar, idio)
    loadings = {
        "USO":  (0.30, -0.05, 0.85, 0.05, 0.05, 0.00, -0.15, 0.020),
        "GLD":  (-0.10, 0.20, 0.25, 0.00, 0.00, 0.00, -0.25, 0.010),
        "SPY":  (1.00, -0.10, 0.10, 0.10, 0.25, 0.00, -0.05, 0.006),
        "QQQ":  (1.05, -0.20, 0.05, 0.05, 0.55, 0.00, -0.05, 0.010),
        "TLT":  (-0.15, 1.00, 0.00, 0.05, -0.10, 0.00, -0.10, 0.007),
        "IEF":  (-0.05, 0.85, 0.00, 0.05, -0.05, 0.00, -0.05, 0.003),
        "IWM":  (1.05, -0.10, 0.08, 0.25, 0.15, 0.00, -0.05, 0.009),
        "LQD":  (0.10, 0.70, 0.00, 0.60, 0.00, 0.00, 0.00, 0.004),
        "HYG":  (0.30, 0.10, 0.10, 0.95, 0.05, 0.00, -0.05, 0.006),
        "SGOV": (0.00, 0.05, 0.00, 0.00, 0.00, 1.00, 0.00, 0.0004),
        "DRAM": (0.60, 0.00, 0.00, 0.00, 1.00, 0.00, 0.00, 0.030),
        "SOXX": (0.80, -0.10, 0.00, 0.05, 0.90, 0.00, 0.00, 0.020),
        "EFA":  (0.85, -0.10, 0.10, 0.10, 0.15, 0.00, -0.15, 0.008),
        "EEM":  (0.75, -0.05, 0.35, 0.30, 0.25, 0.00, -0.20, 0.010),
        "TIP":  (-0.05, 0.80, 0.10, 0.05, -0.05, 0.00, 0.00, 0.003),
        "UUP":  (-0.05, 0.15, -0.35, -0.05, -0.05, 0.00, 1.00, 0.005),
        "DBC":  (0.25, -0.05, 0.80, 0.05, 0.05, 0.00, -0.45, 0.012),
        "XLK":  (1.00, -0.20, 0.00, 0.05, 0.70, 0.00, 0.00, 0.010),
        "XLF":  (1.00, 0.20, 0.05, 0.30, 0.10, 0.00, 0.00, 0.010),
        "XLE":  (0.70, -0.05, 0.60, 0.15, 0.05, 0.00, -0.10, 0.015),
        "XLU":  (0.35, 0.55, -0.05, 0.05, 0.00, 0.00, 0.00, 0.008),
        "XLP":  (0.45, 0.25, 0.00, 0.05, 0.00, 0.00, 0.00, 0.007),
    }

    # SPY "verdadero" con media-reversión para crear cointegración con IWM/QQQ
    spy_log = np.cumsum(market * 1.0)

    # mercado retardado 1 sesión: crea relaciones lead-lag reales en la demo
    market_lag1 = np.concatenate([[0.0], market[:-1]])
    etfs: Dict[str, np.ndarray] = {}
    for t in tickers:
        if t in loadings:
            a, b, c, d, e, f, g, idio = loadings[t]
        else:
            a, b, c, d, e, f, g, idio = (0.8, 0.0, 0.0, 0.1, 0.1, 0.0, -0.05, 0.012)
        if t in ("IWM", "QQQ"):  # cointegradas con SPY
            r = 0.7 * market + 0.3 * (spy_log - np.cumsum(market)) + rng.normal(0, idio, n)
        elif t in ("SOXX", "DRAM"):  # anticipan al mercado 1 sesión (lead-lag)
            r = (a * market + 0.45 * market_lag1 + b * rates + c * commodity + d * credit
                 + e * semi + f * cash + g * dollar + rng.normal(0, idio, n))
        else:
            r = (a * market + b * rates + c * commodity + d * credit + e * semi
                 + f * cash + g * dollar + rng.normal(0, idio, n))
        logp = np.cumsum(r)
        start_price = {"USO": 20, "GLD": 120, "SPY": 200, "QQQ": 90, "TLT": 90, "IEF": 95,
                       "IWM": 100, "LQD": 110, "HYG": 80, "SGOV": 100, "DRAM": 30, "SOXX": 150,
                       "EFA": 70, "EEM": 40, "TIP": 100, "UUP": 28, "DBC": 20,
                       "XLK": 180, "XLF": 38, "XLE": 85, "XLU": 70, "XLP": 75}.get(t, 50)
        etfs[t] = start_price * np.exp(logp)

    return pd.DataFrame(etfs, index=dates)
