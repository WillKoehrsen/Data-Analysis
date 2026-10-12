"""Interfaz de línea de comandos del agente de correlaciones entre ETFs.

Ejemplos
--------
    # Modo demo (datos sintéticos, sin internet)
    python -m etf_agent demo

    # Análisis real con el universo por defecto
    python -m etf_agent analyze

    # Análisis de ETFs concretos y ventanas concretas
    python -m etf_agent analyze --tickers SPY,TLT,GLD,UUP --windows max,ytd,3m,1m,1y

    # Dashboard interactivo (Streamlit)
    python -m etf_agent dashboard
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .config import ALL_WINDOWS, DEFAULT_FACTORS, DEFAULT_TICKERS, DEFAULT_WINDOWS


def _parse_list(value: str | None, default):
    if not value:
        return list(default)
    return [v.strip().upper() for v in value.split(",") if v.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="etf_agent",
        description=(
            "Agente para buscar correlaciones y relaciones entre ETFs "
            "(histórico, YTD, 3M, 1M, ...)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Ventanas disponibles: " + ", ".join(sorted(ALL_WINDOWS)) + "\n"
            "Universo por defecto: " + ", ".join(DEFAULT_TICKERS)
        ),
    )
    parser.add_argument("--version", action="version", version=f"etf_agent {__version__}")
    sub = parser.add_subparsers(dest="command")

    # ---------------- analyze ----------------
    an = sub.add_parser("analyze", help="Analiza correlaciones y relaciones (datos reales).")
    an.add_argument("--tickers", "-t", type=str, default=None,
                    help="Lista separada por comas. Por defecto: el universo completo.")
    an.add_argument("--windows", "-w", type=str, default=",".join(DEFAULT_WINDOWS),
                    help="Ventanas separadas por comas (max,ytd,3m,1m,6m,1y,3y,5y).")
    an.add_argument("--factors", "-f", type=str, default=",".join(DEFAULT_FACTORS),
                    help="Factores para las betas (por defecto SPY,TLT,GLD).")
    an.add_argument("--start", type=str, default=None, help="Fecha inicial YYYY-MM-DD (opcional).")
    an.add_argument("--end", type=str, default=None, help="Fecha final YYYY-MM-DD (opcional).")
    an.add_argument("--outdir", "-o", type=str, default="reports", help="Carpeta de salida.")
    an.add_argument("--cache-dir", type=str, default="data/etf_cache",
                    help="Carpeta de cache de precios.")
    an.add_argument("--no-cache", action="store_true", help="Ignora el cache local.")
    an.add_argument("--source", type=str, default="yfinance", choices=["yfinance", "stooq"],
                    help="Fuente de datos.")
    an.add_argument("--rolling-window", type=int, default=63,
                    help="Ventana de correlación móvil en sesiones (63 ≈ 3 meses).")
    an.add_argument("--max-lag", type=int, default=10, help="Rezagos máximos para lead-lag.")
    an.add_argument("--top", type=int, default=15, help="Nº de pares en los rankings.")

    # ---------------- demo ----------------
    dm = sub.add_parser("demo", help="Ejecuta el análisis con datos sintéticos (sin internet).")
    dm.add_argument("--tickers", "-t", type=str, default=None)
    dm.add_argument("--windows", "-w", type=str, default=",".join(DEFAULT_WINDOWS))
    dm.add_argument("--factors", "-f", type=str, default=",".join(DEFAULT_FACTORS))
    dm.add_argument("--outdir", "-o", type=str, default="reports")
    dm.add_argument("--rolling-window", type=int, default=63)
    dm.add_argument("--max-lag", type=int, default=10)
    dm.add_argument("--top", type=int, default=15)
    dm.add_argument("--start", type=str, default="2015-01-01")

    # ---------------- dashboard ----------------
    db = sub.add_parser("dashboard", help="Lanza el dashboard interactivo con Streamlit.")
    db.add_argument("--port", type=int, default=8501)

    # ---------------- list ----------------
    sub.add_parser("list", help="Muestra el universo de ETFs y las ventanas disponibles.")

    return parser


def cmd_analyze(args) -> int:
    from .data import fetch_prices
    from .report import build_report

    tickers = _parse_list(args.tickers, DEFAULT_TICKERS)
    windows = [w.strip().lower() for w in args.windows.split(",") if w.strip()]
    factors = _parse_list(args.factors, DEFAULT_FACTORS)

    print(f"[etf_agent] Universo ({len(tickers)}): {', '.join(tickers)}")
    prices = fetch_prices(
        tickers=tickers,
        start=args.start,
        end=args.end,
        cache_dir=args.cache_dir,
        use_cache=not args.no_cache,
        source=args.source,
    )
    print(f"[etf_agent] Datos: {prices.shape[0]} sesiones × {prices.shape[1]} ETFs "
          f"({prices.index.min():%Y-%m-%d} → {prices.index.max():%Y-%m-%d})")
    result = build_report(
        prices,
        windows=windows,
        factors=factors,
        outdir=args.outdir,
        rolling_window=args.rolling_window,
        max_lag=args.max_lag,
        top_n=args.top,
    )
    print(f"[etf_agent] Informe Markdown: {result['markdown']}")
    print(f"[etf_agent] Informe HTML:     {result['html']}")
    return 0


def cmd_demo(args) -> int:
    from .data import make_demo_prices
    from .report import build_report

    tickers = _parse_list(args.tickers, DEFAULT_TICKERS)
    windows = [w.strip().lower() for w in args.windows.split(",") if w.strip()]
    factors = _parse_list(args.factors, DEFAULT_FACTORS)

    prices = make_demo_prices(tickers=tickers, start=args.start)
    print(f"[etf_agent] MODO DEMO: datos sintéticos {prices.shape[0]} sesiones × "
          f"{prices.shape[1]} ETFs")
    result = build_report(
        prices,
        windows=windows,
        factors=factors,
        outdir=args.outdir,
        rolling_window=args.rolling_window,
        max_lag=args.max_lag,
        top_n=args.top,
        demo=True,
    )
    print(f"[etf_agent] Informe Markdown: {result['markdown']}")
    print(f"[etf_agent] Informe HTML:     {result['html']}")
    return 0


def cmd_dashboard(args) -> int:
    import subprocess

    app = Path(__file__).parent / "dashboard.py"
    cmd = [
        sys.executable, "-m", "streamlit", "run", str(app),
        "--server.port", str(args.port),
        "--server.address", "0.0.0.0",
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false",
    ]
    print("[etf_agent] Lanzando dashboard:\n  " + " ".join(cmd))
    return subprocess.call(cmd)


def cmd_list(args) -> int:
    from .config import TICKER_INFO

    print("Universo de ETFs por defecto:\n")
    for t in DEFAULT_TICKERS:
        info = TICKER_INFO.get(t, {})
        print(f"  {t:<6} {info.get('clase', ''):<42} {info.get('name', '')}")
    print("\nOpcionales documentados:")
    for t, info in TICKER_INFO.items():
        if t not in DEFAULT_TICKERS:
            print(f"  {t:<6} {info.get('clase', ''):<42} {info.get('name', '')}")
    print("\nVentanas disponibles:")
    for k, w in sorted(ALL_WINDOWS.items()):
        print(f"  {k:<4} {w.description}")
    print("\nFactores por defecto para betas: " + ", ".join(DEFAULT_FACTORS))
    return 0


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return 1
    handlers = {
        "analyze": cmd_analyze,
        "demo": cmd_demo,
        "dashboard": cmd_dashboard,
        "list": cmd_list,
    }
    try:
        return handlers[args.command](args)
    except KeyboardInterrupt:
        print("\n[etf_agent] Interrumpido por el usuario.")
        return 130
    except Exception as err:
        print(f"[etf_agent] ERROR: {err}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
