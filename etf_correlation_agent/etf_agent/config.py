"""Configuración del agente: ETFs por defecto, ventanas temporales y factores."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Dict, Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Universo de ETFs por defecto (los "más significativos")
# ---------------------------------------------------------------------------
DEFAULT_TICKERS = [
    # --- Renta variable USA (benchmarks) ---
    "SPY",   # SPDR S&P 500 — renta variable USA large cap (benchmark de mercado)
    "QQQ",   # Invesco QQQ — Nasdaq 100 / tecnológico
    "IWM",   # iShares Russell 2000 — small caps
    # --- Renta variable internacional ---
    "EFA",   # iShares MSCI EAFE — mercados desarrollados fuera de Norteamérica
    "EEM",   # iShares MSCI Emerging Markets — mercados emergentes
    # --- Renta fija / tipos de interés ---
    "TLT",   # iShares 20+ Year Treasury — bonos USA largos
    "IEF",   # iShares 7-10 Year Treasury — bonos USA medios
    "TIP",   # iShares TIPS — bonos protegidos contra la inflación
    "SGOV",  # iShares 0-3 Month Treasury — letras del tesoro (casi cash)
    # --- Crédito corporativo ---
    "LQD",   # iShares iBoxx Investment Grade — crédito corporativo IG
    "HYG",   # iShares iBoxx High Yield — crédito corporativo HY
    # --- Divisas, refugios y materias primas ---
    "UUP",   # Invesco DB US Dollar Index Bullish — fortaleza del dólar (DXY)
    "GLD",   # SPDR Gold Shares — oro
    "USO",   # United States Oil Fund — petróleo
    "DBC",   # Invesco DB Commodity Index Tracking — cesta de materias primas
    # --- Sectores (SPDR Sector Select) ---
    "XLK",   # Technology
    "XLF",   # Financials
    "XLE",   # Energy
    "XLU",   # Utilities (defensivo)
    "XLP",   # Consumer Staples (defensivo)
    # --- Semiconductores ---
    "SOXX",  # iShares Semiconductor — semiconductores
]

# ETFs disponibles pero NO incluidos por defecto.
# DRAM se excluye a propósito: su histórico es muy corto y recortaría
# (silenciosamente) el periodo común de los análisis multivariante
# (PCA, betas, Granger, cointegración, información mutua...).
OPTIONAL_TICKERS = {
    "DRAM": {"name": "ETF de memoria / DRAM", "clase": "Semiconductores (memoria) — historial corto"},
    "VWO": {"name": "Vanguard FTSE Emerging Markets", "clase": "Renta variable (emergentes)"},
    "SHY": {"name": "iShares 1-3 Year Treasury", "clase": "Renta fija (bonos cortos)"},
    "BIL": {"name": "SPDR Bloomberg 1-3 Month T-Bill", "clase": "Letras del Tesoro (cash-like)"},
    "AGG": {"name": "iShares Core US Aggregate Bond", "clase": "Renta fija (agregado USA)"},
    "VIXY": {"name": "ProShares VIX Short-Term Futures", "clase": "Volatilidad implícita"},
}

TICKER_INFO: Dict[str, Dict[str, str]] = {
    "SPY":  {"name": "SPDR S&P 500 ETF Trust",           "clase": "Renta variable (large cap USA)"},
    "QQQ":  {"name": "Invesco QQQ Trust",                "clase": "Renta variable (Nasdaq 100)"},
    "IWM":  {"name": "iShares Russell 2000 ETF",         "clase": "Renta variable (small caps USA)"},
    "EFA":  {"name": "iShares MSCI EAFE ETF",            "clase": "Renta variable (desarrollados ex-NA)"},
    "EEM":  {"name": "iShares MSCI Emerging Markets",    "clase": "Renta variable (emergentes)"},
    "TLT":  {"name": "iShares 20+ Year Treasury Bond",   "clase": "Renta fija (bonos largos USA)"},
    "IEF":  {"name": "iShares 7-10 Year Treasury Bond",  "clase": "Renta fija (bonos medios USA)"},
    "TIP":  {"name": "iShares TIPS Bond ETF",            "clase": "Renta fija (protegido de inflación)"},
    "SGOV": {"name": "iShares 0-3 Month Treasury Bond",  "clase": "Letras del Tesoro (cash-like)"},
    "LQD":  {"name": "iShares iBoxx $ IG Corp Bond",     "clase": "Crédito corporativo (IG)"},
    "HYG":  {"name": "iShares iBoxx $ High Yield Corp",  "clase": "Crédito corporativo (HY)"},
    "UUP":  {"name": "Invesco DB US Dollar Index Bull",  "clase": "Divisas (dólar DXY)"},
    "GLD":  {"name": "SPDR Gold Shares",                 "clase": "Materias primas (oro)"},
    "USO":  {"name": "United States Oil Fund",           "clase": "Materias primas (petróleo)"},
    "DBC":  {"name": "Invesco DB Commodity Index Track","clase": "Materias primas (cesta diversificada)"},
    "XLK":  {"name": "Technology Select Sector SPDR",    "clase": "Sector: Tecnología"},
    "XLF":  {"name": "Financial Select Sector SPDR",     "clase": "Sector: Financiero"},
    "XLE":  {"name": "Energy Select Sector SPDR",        "clase": "Sector: Energía"},
    "XLU":  {"name": "Utilities Select Sector SPDR",     "clase": "Sector: Utilities (defensivo)"},
    "XLP":  {"name": "Consumer Staples Select Sector",   "clase": "Sector: Consumo defensivo"},
    "SOXX": {"name": "iShares Semiconductor ETF",        "clase": "Semiconductores"},
    **OPTIONAL_TICKERS,
}

# ---------------------------------------------------------------------------
# Factores usados para el análisis de betas / regresiones
# ---------------------------------------------------------------------------
DEFAULT_FACTORS = ["SPY", "TLT", "GLD"]


# ---------------------------------------------------------------------------
# Ventanas temporales
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Window:
    """Definición de una ventana de análisis."""

    label: str
    description: str
    months: Optional[float] = None   # nº de meses hacia atrás (None = histórico completo)
    ytd: bool = False                # desde el 1 de enero del año en curso

    def bounds(self, index: pd.DatetimeIndex, asof=None) -> tuple:
        """Devuelve (inicio, fin) como Timestamps para un índice de fechas dado."""
        if len(index) == 0:
            raise ValueError("El índice de fechas está vacío.")
        end = pd.Timestamp(asof) if asof is not None else index.max()
        if self.ytd:
            start = pd.Timestamp(year=end.year, month=1, day=1)
        elif self.months is None:
            start = index.min()
        else:
            start = end - pd.DateOffset(months=int(self.months))
        return start, end


WINDOWS: Dict[str, Window] = {
    "max": Window("max", "Histórico completo", months=None),
    "ytd": Window("ytd", "Año en curso (YTD)", ytd=True),
    "3m":  Window("3m",  "Últimos 3 meses", months=3),
    "1m":  Window("1m",  "Último mes", months=1),
}

# Ventanas adicionales disponibles bajo demanda
EXTRA_WINDOWS: Dict[str, Window] = {
    "6m":  Window("6m",  "Últimos 6 meses", months=6),
    "1y":  Window("1y",  "Último año", months=12),
    "3y":  Window("3y",  "Últimos 3 años", months=36),
    "5y":  Window("5y",  "Últimos 5 años", months=60),
}

ALL_WINDOWS: Dict[str, Window] = {**WINDOWS, **EXTRA_WINDOWS}

DEFAULT_WINDOWS = ["max", "ytd", "3m", "1m"]


def resolve_windows(names) -> Dict[str, Window]:
    """Valida y devuelve las ventanas pedidas (acepta str o Window)."""
    if names is None:
        names = DEFAULT_WINDOWS
    if isinstance(names, (str, Window)):
        names = [names]
    resolved: Dict[str, Window] = {}
    for name in names:
        if isinstance(name, Window):
            resolved[name.label] = name
            continue
        key = str(name).strip().lower()
        if key not in ALL_WINDOWS:
            raise ValueError(
                f"Ventana desconocida: {name}. Disponibles: {sorted(ALL_WINDOWS)}"
            )
        resolved[key] = ALL_WINDOWS[key]
    return resolved


def window_bounds(index: pd.DatetimeIndex, window, asof=None) -> tuple:
    """Atajo: calcula (inicio, fin) para una ventana sobre un índice de fechas."""
    if not isinstance(window, Window):
        window = ALL_WINDOWS[str(window).lower()]
    return window.bounds(index, asof)


def describe_tickers(tickers) -> str:
    """Texto legible con la clase de activo de cada ticker."""
    parts = []
    for t in tickers:
        info = TICKER_INFO.get(t, {})
        parts.append(f"{t} ({info.get('clase', 'desconocido')})")
    return ", ".join(parts)


def today() -> date:
    return date.today()
