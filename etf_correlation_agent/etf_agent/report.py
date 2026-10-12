"""Generación de informes: gráficos, CSVs, Markdown y HTML auto-contenido."""

from __future__ import annotations

import base64
import io
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")  # backend sin pantalla

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from . import relationships as rel
from .config import (
    DEFAULT_FACTORS,
    TICKER_INFO,
    Window,
    describe_tickers,
    resolve_windows,
)
from .correlation import (
    average_rolling_correlation,
    clustered_order,
    correlation_matrices,
    log_returns,
    rolling_correlation,
    top_pairs,
)

PALETTE = "RdBu_r"
FIGSIZE_BASE = 1.0


# ---------------------------------------------------------------------------
# Utilidades de plotting
# ---------------------------------------------------------------------------
def _save(fig, path: Optional[Path]):
    """Guarda la figura si se indica ruta; si no, la deja abierta para el notebook."""
    if path is None:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def _fig_size(n: int, base: float = 0.55, min_size: float = 6.0, max_size: float = 16.0) -> tuple:
    side = float(np.clip(base * n, min_size, max_size))
    return (side, side * 0.85)


def heatmap(
    corr: pd.DataFrame,
    title: str = "",
    path: Optional[Path] = None,
    order: Optional[Sequence[str]] = None,
    vmin: float = -1.0,
    vmax: float = 1.0,
    annot: bool = True,
) -> Optional[Path]:
    """Dibuja (y opcionalmente guarda) un heatmap de correlación."""
    if order:
        cols = [c for c in order if c in corr.columns]
        corr = corr.loc[cols, cols]
    n = corr.shape[0]
    fig, ax = plt.subplots(figsize=_fig_size(n))
    sns.heatmap(
        corr,
        vmin=vmin,
        vmax=vmax,
        cmap=PALETTE,
        center=0,
        annot=annot and n <= 24,
        fmt=".2f",
        square=True,
        linewidths=0.4,
        cbar_kws={"shrink": 0.75},
        ax=ax,
    )
    ax.set_title(title or "Matriz de correlación", fontsize=max(10, min(18, n * 0.6)))
    plt.xticks(rotation=45, ha="right", fontsize=max(6, 12 - n // 4))
    plt.yticks(rotation=0, fontsize=max(6, 12 - n // 4))
    fig.tight_layout()
    return _save(fig, path)


def _plot_lines(
    series_dict: Dict[str, pd.Series],
    title: str,
    ylabel: str,
    path: Optional[Path] = None,
    hlines: Optional[Dict[str, float]] = None,
    figsize: tuple = (11, 4.5),
) -> Optional[Path]:
    fig, ax = plt.subplots(figsize=figsize)
    for name, s in series_dict.items():
        ax.plot(s.index, s.values, label=name, linewidth=1.2)
    if hlines:
        for label, val in hlines.items():
            ax.axhline(val, linestyle="--", linewidth=1, alpha=0.7, label=label)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    return _save(fig, path)


def plot_normalized_prices(prices: pd.DataFrame, path: Optional[Path] = None,
                           base: float = 100.0) -> Optional[Path]:
    """Precios rebased a 100 (normalizados) para comparar la evolución."""
    norm = prices / prices.iloc[0] * base
    fig, ax = plt.subplots(figsize=(12, 5))
    for c in norm.columns:
        ax.plot(norm.index, norm[c], linewidth=1.0, label=c)
    ax.set_yscale("log")
    ax.set_title("Evolución comparada (base 100, escala logarítmica)")
    ax.set_ylabel(f"Precio (base {base})")
    ax.legend(fontsize=7, ncol=4)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return _save(fig, path)


def plot_pca_scree(pca: Dict[str, object], path: Optional[Path] = None) -> Optional[Path]:
    evr = pca["explained_variance_ratio"]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.bar(evr.index, evr.values, color="#2b6cb0")
    cum = evr.cumsum()
    ax2 = ax.twinx()
    ax2.plot(evr.index, cum.values, color="#c53030", marker="o", label="acumulado")
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("varianza explicada acumulada")
    for i, v in enumerate(evr.values):
        ax.text(i, v + 0.01, f"{v:.0%}", ha="center", fontsize=8)
    ax.set_title("Componentes principales (scree plot)")
    ax.set_ylabel("fracción de varianza explicada")
    fig.tight_layout()
    return _save(fig, path)


def plot_dendrogram(corr: pd.DataFrame, path: Optional[Path] = None,
                    order: Optional[Sequence[str]] = None) -> Optional[Path]:
    """Dendrograma simple a partir de la distancia 1-|corr|."""
    from scipy.cluster.hierarchy import dendrogram, linkage
    from scipy.spatial.distance import squareform

    if order is None:
        order = list(corr.columns)
    sub = corr.loc[list(order), list(order)]
    dist = 1 - sub.abs()
    dist = (dist + dist.T) / 2
    dist_arr = np.array(dist.to_numpy(dtype=float), copy=True)
    np.fill_diagonal(dist_arr, 0.0)
    try:
        z = linkage(squareform(dist_arr, checks=False), method="average")
    except Exception:
        return path
    fig, ax = plt.subplots(figsize=(10, 4.5))
    dendrogram(z, labels=list(sub.columns), ax=ax, leaf_rotation=45, color_threshold=0.5)
    ax.set_title("Clustering jerárquico de ETFs (distancia = 1 - |correlación|)")
    ax.set_ylabel("distancia")
    fig.tight_layout()
    return _save(fig, path)


def plot_network(corr: pd.DataFrame, path: Optional[Path] = None, threshold: float = 0.6,
                 use_abs: bool = False) -> Optional[Path]:
    """Red de correlación (layout de resortes con networkx si está disponible)."""
    try:
        import networkx as nx
    except Exception:
        return path
    G = nx.Graph()
    cols = list(corr.columns)
    G.add_nodes_from(cols)
    for i, a in enumerate(cols):
        for j, b in enumerate(cols):
            if i >= j:
                continue
            v = corr.iloc[i, j]
            if pd.isna(v):
                continue
            keep = abs(v) >= threshold if use_abs else v >= threshold
            if keep:
                G.add_edge(a, b, weight=float(abs(v)))
    fig, ax = plt.subplots(figsize=(9, 7))
    pos = nx.spring_layout(G, seed=7, weight="weight", k=0.6)
    weights = [G[u][v]["weight"] * 3 for u, v in G.edges()]
    nx.draw_networkx_nodes(G, pos, node_color="#2c5282", node_size=900, alpha=0.9, ax=ax)
    nx.draw_networkx_edges(G, pos, width=weights, alpha=0.5, edge_color="#4a5568", ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=9, font_color="white", ax=ax)
    ax.set_title(f"Red de correlaciones (umbral {threshold}{' en |·|' if use_abs else ''})")
    ax.axis("off")
    fig.tight_layout()
    return _save(fig, path)


# ---------------------------------------------------------------------------
# Tabla Markdown helper
# ---------------------------------------------------------------------------
def _df_to_md(df: pd.DataFrame, max_rows: int = 20, float_fmt: str = "{:.3f}") -> str:
    """Convierte un DataFrame a tabla Markdown legible (sin depender de tabulate)."""
    if df is None or df.empty:
        return "_Sin resultados._\n"
    d = df.head(max_rows).copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda v: "n/a" if pd.isna(v) else float_fmt.format(v))
        elif pd.api.types.is_datetime64_any_dtype(d[c]):
            d[c] = d[c].dt.strftime("%Y-%m-%d")
        else:
            d[c] = d[c].astype(str)
    header = "| " + " | ".join(str(c) for c in d.columns) + " |"
    sep = "| " + " | ".join("---" for _ in d.columns) + " |"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in d.itertuples(index=False)]
    return "\n".join([header, sep, *rows]) + "\n"


def _fig_to_b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


# ---------------------------------------------------------------------------
# Informe principal
# ---------------------------------------------------------------------------
def build_report(
    prices: pd.DataFrame,
    windows: Optional[Iterable] = None,
    factors: Sequence[str] = DEFAULT_FACTORS,
    outdir: str | Path = "reports",
    rolling_window: int = 63,
    max_lag: int = 10,
    top_n: int = 15,
    demo: bool = False,
    asof=None,
    verbose: bool = True,
) -> Dict[str, object]:
    """Ejecuta todos los análisis y escribe el informe completo en ``outdir``.

    Returns
    -------
    dict con rutas: {'dir', 'markdown', 'html', 'images', 'tables', 'summary'}
    """
    windows = resolve_windows(windows)
    tickers = list(prices.columns)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = "demo" if demo else "live"
    outdir = Path(outdir) / f"etf_correlations_{tag}_{stamp}"
    img_dir = outdir / "images"
    data_dir = outdir / "data"
    for d in (img_dir, data_dir):
        d.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, object] = {
        "generado": datetime.now().isoformat(timespec="seconds"),
        "modo": "demo (datos sintéticos)" if demo else "datos reales",
        "tickers": tickers,
        "ventanas": list(windows.keys()),
        "analisis": {},
        "errores": {},
    }

    rets = log_returns(prices)
    images: Dict[str, Path] = {}
    tables: Dict[str, Path] = {}
    md_parts: List[str] = []

    # ------------------------------------------------------------------
    # Cabecera
    # ------------------------------------------------------------------
    first, last = prices.index.min(), prices.index.max()
    md_parts.append(
        f"# Informe de correlaciones y relaciones entre ETFs\n\n"
        f"* **Modo**: {summary['modo']}\n"
        f"* **Generado**: {summary['generado']}\n"
        f"* **ETFs analizados** ({len(tickers)}): {', '.join(tickers)}\n"
        f"* **Periodo de datos**: {first:%Y-%m-%d} → {last:%Y-%m-%d} "
        f"({len(prices)} sesiones, {len(rets)} retornos)\n"
        f"* **Ventanas**: {', '.join(f'{k} ({v.description})' for k, v in windows.items())}\n\n"
        f"```\n{describe_tickers(tickers)}\n```\n"
    )

    # ------------------------------------------------------------------
    # 0. Precios normalizados
    # ------------------------------------------------------------------
    try:
        images["precios"] = plot_normalized_prices(prices, img_dir / "precios_normalizados.png")
        md_parts.append("\n## 0. Evolución comparada\n\n")
        md_parts.append(f"![precios]({images['precios'].relative_to(outdir).as_posix()})\n")
    except Exception as e:
        summary["errores"]["precios"] = str(e)

    # ------------------------------------------------------------------
    # 1. Matrices de correlación por ventana
    # ------------------------------------------------------------------
    corr_mats = correlation_matrices(prices, windows=windows, asof=asof)
    md_parts.append("\n## 1. Correlaciones por ventana temporal\n\n")
    for label, win in windows.items():
        idx = rets.index[rets.index.isin(
            prices.index[(prices.index >= win.bounds(prices.index, asof)[0])
                         & (prices.index <= win.bounds(prices.index, asof)[1])]
        )]
        n_obs = len(rets.loc[rets.index.isin(idx)].dropna(how="all"))
        md_parts.append(
            f"### Ventana `{label}` — {win.description}\n\n"
            f"Observaciones utilizadas: **{n_obs}** retornos diarios.\n\n"
        )
        for method in ("pearson", "spearman"):
            key = (label, method)
            if key not in corr_mats:
                continue
            corr = corr_mats[key]
            name = "Pearson (lineal)" if method == "pearson" else "Spearman (rangos)"
            try:
                path = heatmap(
                    corr,
                    title=f"Correlación {name} — {label}",
                    path=img_dir / f"heatmap_{label}_{method}.png",
                )
                if path:
                    images[f"heatmap_{label}_{method}"] = path
                    md_parts.append(
                        f"**{name}**\n\n![heatmap {label} {method}]"
                        f"({path.relative_to(outdir).as_posix()})\n"
                    )
            except Exception as e:
                summary["errores"][f"heatmap_{label}_{method}"] = str(e)
            csv = data_dir / f"corr_{label}_{method}.csv"
            corr.round(4).to_csv(csv)
            tables[f"corr_{label}_{method}"] = csv

        # Top pares
        tp = top_pairs(corr_mats[(label, "pearson")], n=top_n)
        tp_csv = data_dir / f"top_pairs_{label}.csv"
        tp.to_csv(tp_csv, index=False)
        tables[f"top_pairs_{label}"] = tp_csv
        md_parts.append(
            f"\n**Top {min(top_n, len(tp))} pares más correlacionados (Pearson, {label})**\n\n"
            + _df_to_md(tp, max_rows=top_n)
        )
        if len(tp):
            summary["analisis"][f"top_pairs_{label}"] = tp.head(5).to_dict("records")

    # ------------------------------------------------------------------
    # 2. Correlación rolling media del universo
    # ------------------------------------------------------------------
    try:
        avg = average_rolling_correlation(rets, window=rolling_window)
        images["rolling_avg"] = _plot_lines(
            {"correlación media": avg},
            f"Correlación media del universo (ventana móvil {rolling_window}d)",
            "correlación media",
            img_dir / "rolling_corr_media.png",
            hlines={"media histórica": float(avg.mean())},
        )
        md_parts.append(
            "\n## 2. Régimen de correlación\n\n"
            "La correlación media entre todos los pares del universo muestra cómo se "
            "**concentran los movimientos** (sube en crisis, baja en mercados tranquilos).\n\n"
            f"![rolling avg]({images['rolling_avg'].relative_to(outdir).as_posix()})\n"
        )
        avg.to_frame().to_csv(data_dir / "rolling_corr_media.csv")
        tables["rolling_corr_media"] = data_dir / "rolling_corr_media.csv"
        top_reg = avg.sort_values(ascending=False).head(10)
        summary["analisis"]["max_corr_media"] = {
            str(d.date()): round(float(v), 3) for d, v in top_reg.items()
        }
    except Exception as e:
        summary["errores"]["rolling_avg"] = str(e)

    # ------------------------------------------------------------------
    # 3. Correlación rolling de los pares más correlacionados
    # ------------------------------------------------------------------
    try:
        base_corr = corr_mats[("max", "pearson")] if ("max", "pearson") in corr_mats else corr_mats[
            (list(windows)[0], "pearson")
        ]
        pairs = top_pairs(base_corr, n=5)
        roll_fig = {}
        for _, row in pairs.iterrows():
            a, b = row["A"], row["B"]
            if a not in rets or b not in rets:
                continue
            s = rolling_correlation(rets, a, b, window=rolling_window)
            roll_fig[f"{a}-{b}"] = s
            s.to_frame(name=f"corr_{a}_{b}").to_csv(data_dir / f"rolling_corr_{a}_{b}.csv")
        if roll_fig:
            images["rolling_pairs"] = _plot_lines(
                roll_fig,
                f"Correlación móvil ({rolling_window}d) de los pares más correlacionados",
                "correlación",
                img_dir / "rolling_corr_pares.png",
                figsize=(12, 5),
            )
            md_parts.append(
                "\n## 3. Correlación móvil de los pares más relevantes\n\n"
                f"![rolling pairs]({images['rolling_pairs'].relative_to(outdir).as_posix()})\n"
            )
    except Exception as e:
        summary["errores"]["rolling_pairs"] = str(e)

    # ------------------------------------------------------------------
    # 4. Lead-lag
    # ------------------------------------------------------------------
    try:
        ll = rel.lead_lag_table(rets, max_lag=max_lag, top=top_n)
        ll.to_csv(data_dir / "lead_lag.csv", index=False)
        tables["lead_lag"] = data_dir / "lead_lag.csv"
        md_parts.append(
            "\n## 4. Relaciones lead-lag (¿quién anticipa a quién?)\n\n"
            "Para cada par se busca el rezago (hasta ±"
            f"{max_lag} sesiones) con mayor correlación. Un `best_lag_days > 0` indica que "
            "`leader` se mueve **antes** que `follower`.\n\n"
            + _df_to_md(ll, max_rows=top_n)
        )
        summary["analisis"]["lead_lag"] = ll.head(8).to_dict("records") if not ll.empty else []
    except Exception as e:
        summary["errores"]["lead_lag"] = str(e)

    # ------------------------------------------------------------------
    # 5. Granger
    # ------------------------------------------------------------------
    try:
        gc = rel.granger_table(rets, max_lag=5)
        gc.to_csv(data_dir / "granger.csv", index=False)
        tables["granger"] = data_dir / "granger.csv"
        sig = gc[gc["significativo"]] if not gc.empty else gc
        md_parts.append(
            "\n## 5. Causalidad de Granger\n\n"
            "H0: `cause` **no** ayuda a predecir `effect`. Pares con p-value < 0.05 "
            "(cuidado: con muchas series aparecerán falsos positivos; valora también la "
            "fuerza económica).\n\n"
            f"**Pares significativos: {len(sig)} de {len(gc)} tests.**\n\n"
            + _df_to_md(gc.head(15), max_rows=15)
        )
        summary["analisis"]["granger_significativos"] = int(len(sig))
        summary["analisis"]["granger_top"] = gc.head(8).to_dict("records") if not gc.empty else []
    except Exception as e:
        summary["errores"]["granger"] = str(e)

    # ------------------------------------------------------------------
    # 6. Cointegración
    # ------------------------------------------------------------------
    try:
        co = rel.cointegration_table(prices)
        co.to_csv(data_dir / "cointegration.csv", index=False)
        tables["cointegration"] = data_dir / "cointegration.csv"
        n_co = int(co["cointegrada"].sum()) if not co.empty else 0
        md_parts.append(
            "\n## 6. Cointegración (relaciones de largo plazo)\n\n"
            f"Pares cointegrados (p < 0.05): **{n_co}**. Estos pares comparten una "
            "tendencia estocástica común: su spread revierte a la media, lo que permite "
            "estrategias de pares. `half_life_days` ≈ días que tarda el spread en "
            "corregirse a la mitad; `spread_zscore` mide cuán lejos está ahora.\n\n"
            + _df_to_md(co.head(15), max_rows=15)
        )
        summary["analisis"]["cointegrados"] = n_co
        summary["analisis"]["cointegration_top"] = co.head(5).to_dict("records") if not co.empty else []
    except Exception as e:
        summary["errores"]["cointegration"] = str(e)

    # ------------------------------------------------------------------
    # 7. Betas factoriales
    # ------------------------------------------------------------------
    try:
        factors_avail = [f for f in factors if f in rets.columns]
        betas = rel.factor_betas(rets, factors=factors_avail)
        betas.to_csv(data_dir / "betas.csv", index=False)
        tables["betas"] = data_dir / "betas.csv"
        md_parts.append(
            "\n## 7. Exposición factorial (betas)\n\n"
            f"Regresión OLS de cada ETF sobre los factores {factors_avail}. "
            "El `beta` mide sensibilidad (beta 1 = se mueve igual que el factor).\n\n"
            + _df_to_md(betas, max_rows=len(betas))
        )
        summary["analisis"]["betas"] = betas.to_dict("records")
    except Exception as e:
        summary["errores"]["betas"] = str(e)

    # ------------------------------------------------------------------
    # 8. PCA
    # ------------------------------------------------------------------
    try:
        pca = rel.pca_factors(rets, n_components=min(5, len(tickers)))
        pca["loadings"].round(3).to_csv(data_dir / "pca_loadings.csv")
        pca["explained_variance_ratio"].round(4).to_frame("varianza").to_csv(
            data_dir / "pca_varianza.csv"
        )
        tables["pca_loadings"] = data_dir / "pca_loadings.csv"
        images["pca"] = plot_pca_scree(pca, img_dir / "pca_scree.png")
        md_parts.append(
            "\n## 8. Factores latentes (PCA)\n\n"
            f"Los primeros {pca['n_components']} componentes explican "
            f"**{pca['explained_variance_ratio'].sum():.1%}** de la varianza total. "
            "PC1 suele ser 'el mercado'; los siguientes capturan tipos, materias primas, "
            "dólar, etc. Cargas altas (en |·|) = ETF muy ligado a ese factor.\n\n"
            f"![pca]({images['pca'].relative_to(outdir).as_posix()})\n\n"
            "**Varianza explicada**\n\n"
            + _df_to_md(pca["explained_variance_ratio"].round(4).to_frame("fraccion").reset_index()
                        .rename(columns={"index": "componente"}), max_rows=6)
            + "\n**Cargas (loadings)**\n\n"
            + _df_to_md(pca["loadings"].reset_index().rename(columns={"index": "etf"}), max_rows=25)
        )
        summary["analisis"]["pca_varianza"] = {
            k: round(float(v), 4) for k, v in pca["explained_variance_ratio"].items()
        }
    except Exception as e:
        summary["errores"]["pca"] = str(e)

    # ------------------------------------------------------------------
    # 9. Clustering
    # ------------------------------------------------------------------
    try:
        base_corr = corr_mats[("max", "pearson")] if ("max", "pearson") in corr_mats else corr_mats[
            (list(windows)[0], "pearson")
        ]
        order = clustered_order(base_corr)
        clusters = rel.cluster_etfs(base_corr, n_clusters=min(5, len(tickers)))
        clusters.to_csv(data_dir / "clusters.csv", index=False)
        tables["clusters"] = data_dir / "clusters.csv"
        images["dendrogram"] = plot_dendrogram(base_corr, img_dir / "dendrograma.png", order)
        md_parts.append(
            "\n## 9. Agrupamiento de ETFs (clustering)\n\n"
            "ETFs con comportamientos similares quedan agrupados (distancia = 1 - |corr|):\n\n"
            + _df_to_md(clusters, max_rows=30)
            + f"\n![dendrograma]({images['dendrogram'].relative_to(outdir).as_posix()})\n"
        )
        summary["analisis"]["clusters"] = clusters.to_dict("records")
    except Exception as e:
        summary["errores"]["clusters"] = str(e)

    # ------------------------------------------------------------------
    # 10. Red de correlación
    # ------------------------------------------------------------------
    try:
        base_corr = corr_mats[("max", "pearson")] if ("max", "pearson") in corr_mats else corr_mats[
            (list(windows)[0], "pearson")
        ]
        images["red"] = plot_network(base_corr, img_dir / "red_correlacion.png", threshold=0.6)
        net = []
        cols = list(base_corr.columns)
        for i, a in enumerate(cols):
            for j, b in enumerate(cols):
                if i < j and pd.notna(base_corr.iloc[i, j]) and base_corr.iloc[i, j] >= 0.6:
                    net.append({"A": a, "B": b, "corr": round(float(base_corr.iloc[i, j]), 3)})
        pd.DataFrame(net).to_csv(data_dir / "red_correlacion.csv", index=False)
        md_parts.append(
            "\n## 10. Red de correlación (umbral 0.6)\n\n"
            f"![red]({images['red'].relative_to(outdir).as_posix()})\n"
        )
        summary["analisis"]["red_edges_060"] = len(net)
    except Exception as e:
        summary["errores"]["red"] = str(e)

    # ------------------------------------------------------------------
    # 11. Información mutua
    # ------------------------------------------------------------------
    try:
        mi = rel.mutual_information_matrix(rets)
        mi.round(4).to_csv(data_dir / "mutual_information.csv")
        tables["mutual_information"] = data_dir / "mutual_information.csv"
        md_parts.append(
            "\n## 11. Dependencia no lineal (información mutua)\n\n"
            "La información mutua (en nats) detecta relaciones **no lineales** que la "
            "correlación de Pearson no ve. Compara con la matriz de correlación: pares con "
            "MI alta pero correlación baja son candidatos a relación no lineal.\n\n"
            + _df_to_md(
                mi.stack().rename("MI").reset_index().rename(
                    columns={"level_0": "A", "level_1": "B"}
                ).query("A < B").sort_values("MI", ascending=False).head(15),
                max_rows=15,
            )
        )
    except Exception as e:
        summary["errores"]["mutual_information"] = str(e)

    # ------------------------------------------------------------------
    # 12. Regímenes de estrés
    # ------------------------------------------------------------------
    try:
        bench = "SPY" if "SPY" in rets.columns else tickers[0]
        reg = rel.regime_correlation(rets, benchmark=bench)
        reg.to_csv(data_dir / "regimenes.csv", index=False)
        tables["regimenes"] = data_dir / "regimenes.csv"
        dd = rel.drawdown_episodes(prices, benchmark=bench)
        dd.to_csv(data_dir / "drawdowns.csv", index=False)
        md_parts.append(
            "\n## 12. Correlación en regímenes de estrés\n\n"
            f"Correlación con el benchmark ({bench}) en los días malos (peor 10%) y "
            "buenos (mejor 10%). `asimetria` positiva = la correlación se dispara en las "
            "caídas (comportamiento típico de 'todo cae junto').\n\n"
            + _df_to_md(reg, max_rows=25)
            + "\n**Drawdowns del benchmark (>10%)**\n\n"
            + _df_to_md(dd, max_rows=15)
        )
        summary["analisis"]["regimenes"] = reg.head(10).to_dict("records")
    except Exception as e:
        summary["errores"]["regimenes"] = str(e)

    # ------------------------------------------------------------------
    # 13. Volatilidad
    # ------------------------------------------------------------------
    try:
        vol = rel.volatility_table(rets)
        vol.to_csv(data_dir / "volatilidad.csv", index=False)
        tables["volatilidad"] = data_dir / "volatilidad.csv"
        mdd = rel.max_drawdown(prices)
        vol["max_drawdown_pct"] = vol["etf"].map(mdd)
        vol.to_csv(data_dir / "volatilidad.csv", index=False)
        md_parts.append(
            "\n## 13. Volatilidad y riesgo\n\n"
            + _df_to_md(vol, max_rows=25)
        )
        summary["analisis"]["volatilidad"] = vol.to_dict("records")
    except Exception as e:
        summary["errores"]["volatilidad"] = str(e)

    # ------------------------------------------------------------------
    # Cierre + metodología
    # ------------------------------------------------------------------
    md_parts.append(
        "\n## Metodología y limitaciones\n\n"
        "* Retornos logarítmicos diarios; correlación de Pearson (lineal) y Spearman (rangos).\n"
        "* Ventanas: `max` = histórico completo, `ytd` = desde el 1 de enero, `3m`/`1m` = "
        "últimos 3 meses / 1 mes naturales.\n"
        "* Ventanas cortas (1m ≈ 21 sesiones) tienen mucho ruido: úsalas como diagnóstico, "
        "no como verdad estructural.\n"
        "* Granger ≠ causalidad real (puede haber variables omitidas).\n"
        "* La cointegración mide equilibrio de largo plazo; el hedge ratio y la vida media "
        "se recalibran con el tiempo.\n"
        "* Correlación no implica causalidad: dos ETFs pueden moverse juntos por un factor "
        "común (el mercado, los tipos, el dólar).\n"
    )

    # ------------------------------------------------------------------
    # Escritura de ficheros
    # ------------------------------------------------------------------
    md_path = outdir / "report.md"
    md_path.write_text("".join(md_parts), encoding="utf-8")

    # HTML auto-contenido
    html = _render_html("".join(md_parts), images, outdir, summary)
    html_path = outdir / "report.html"
    html_path.write_text(html, encoding="utf-8")

    summary_path = outdir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    if verbose:
        print(f"[etf_agent] Informe generado en: {outdir.resolve()}")
        print(f"[etf_agent]   - {md_path.name} / {html_path.name}")
        print(f"[etf_agent]   - {len(images)} gráficos, {len(tables)} tablas CSV")

    return {
        "dir": outdir,
        "markdown": md_path,
        "html": html_path,
        "summary": summary_path,
        "images": images,
        "tables": tables,
    }


# ---------------------------------------------------------------------------
# Render HTML (markdown muy simplificado + imágenes en base64)
# ---------------------------------------------------------------------------
def _render_html(md_text: str, images: Dict[str, Path], outdir: Path, summary: Dict) -> str:
    """HTML mínimo auto-contenido a partir del markdown generado."""
    import html as html_lib
    import re

    lines = md_text.splitlines()
    out: List[str] = [
        "<!DOCTYPE html><html lang='es'><head><meta charset='utf-8'>",
        "<title>Informe de correlaciones entre ETFs</title>",
        "<style>",
        "body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;"
        "max-width:1100px;margin:2rem auto;padding:0 1rem;color:#1a202c;line-height:1.55}",
        "h1{border-bottom:3px solid #2b6cb0;padding-bottom:.3rem}",
        "h2{margin-top:2rem;color:#2b6cb0;border-bottom:1px solid #cbd5e0;padding-bottom:.2rem}",
        "h3{color:#2c5282}",
        "table{border-collapse:collapse;margin:1rem 0;font-size:.85rem}",
        "th,td{border:1px solid #cbd5e0;padding:4px 8px;text-align:right}",
        "th{background:#edf2f7}",
        "img{max-width:100%;height:auto;border:1px solid #e2e8f0;border-radius:6px;margin:.5rem 0}",
        "code{background:#edf2f7;padding:1px 4px;border-radius:3px}",
        "pre{background:#edf2f7;padding:8px;border-radius:6px;overflow-x:auto}",
        "</style></head><body>",
    ]

    in_table = False
    in_code = False
    for raw in lines:
        line = raw.rstrip()
        if line.startswith("```"):
            if in_code:
                out.append("</pre>")
                in_code = False
            else:
                out.append("<pre>")
                in_code = True
            continue
        if in_code:
            out.append(html_lib.escape(line))
            continue
        if line.startswith("|") and line.endswith("|"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            if all(set(c) <= set("-: ") for c in cells):
                continue
            tag = "th" if not in_table else "td"
            if not in_table:
                out.append("<table>")
                in_table = True
                tag = "th"
            out.append("<tr>" + "".join(f"<{tag}>{html_lib.escape(c)}</{tag}>" for c in cells) + "</tr>")
            continue
        if in_table:
            out.append("</table>")
            in_table = False

        m = re.match(r"!\[.*?\]\((.*?)\)", line)
        if m:
            rel_path = Path(m.group(1))
            abs_path = (outdir / rel_path).resolve()
            if abs_path.exists():
                b64 = base64.b64encode(abs_path.read_bytes()).decode()
                out.append(f"<img src='data:image/png;base64,{b64}' alt='grafico'>")
            continue

        if line.startswith("### "):
            out.append(f"<h3>{html_lib.escape(line[4:])}</h3>")
        elif line.startswith("## "):
            out.append(f"<h2>{html_lib.escape(line[3:])}</h2>")
        elif line.startswith("# "):
            out.append(f"<h1>{html_lib.escape(line[2:])}</h1>")
        elif line.startswith("* "):
            out.append(f"<li>{html_lib.escape(line[2:])}</li>")
        elif line.strip():
            txt = html_lib.escape(line)
            txt = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", txt)
            txt = re.sub(r"`(.+?)`", r"<code>\1</code>", txt)
            out.append(f"<p>{txt}</p>")

    if in_table:
        out.append("</table>")
    out.append(
        f"<hr><p style='font-size:.8rem;color:#718096'>Generado por etf_agent "
        f"el {summary.get('generado')} · modo {summary.get('modo')}</p></body></html>"
    )
    return "".join(out)
