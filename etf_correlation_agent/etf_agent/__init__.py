"""
etf_agent — Agente de análisis de correlaciones y relaciones entre ETFs.

Permite descargar precios históricos de ETFs, calcular matrices de correlación
en distintas ventanas temporales (histórico, YTD, 3M, 1M, ...) y buscar
cualquier tipo de relación entre sus movimientos (lead-lag, causalidad de
Granger, cointegración, betas factoriales, PCA, clustering, información mutua,
regímenes de estrés, etc.).
"""

__version__ = "1.0.0"

from .config import DEFAULT_TICKERS, TICKER_INFO, WINDOWS, resolve_windows, window_bounds
from .data import fetch_prices, load_cached_prices, make_demo_prices
from .correlation import (
    log_returns,
    correlation_matrix,
    correlation_matrices,
    rolling_correlation,
    correlation_network,
)
from .relationships import (
    lead_lag_table,
    granger_table,
    cointegration_table,
    factor_betas,
    pca_factors,
    cluster_etfs,
    mutual_information_matrix,
    regime_correlation,
    volatility_table,
)
from .report import build_report

__all__ = [
    "DEFAULT_TICKERS",
    "TICKER_INFO",
    "WINDOWS",
    "resolve_windows",
    "fetch_prices",
    "load_cached_prices",
    "make_demo_prices",
    "log_returns",
    "correlation_matrix",
    "correlation_matrices",
    "rolling_correlation",
    "correlation_network",
    "window_bounds",
    "lead_lag_table",
    "granger_table",
    "cointegration_table",
    "factor_betas",
    "pca_factors",
    "cluster_etfs",
    "mutual_information_matrix",
    "regime_correlation",
    "volatility_table",
    "build_report",
]
