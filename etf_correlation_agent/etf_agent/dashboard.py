"""Dashboard interactivo (Streamlit) para explorar correlaciones entre ETFs.

Uso:
    python -m etf_agent dashboard
o directamente:
    streamlit run etf_correlation_agent/etf_agent/dashboard.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Permite ejecutar el fichero directamente (streamlit run <path>)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st


def _streamlit_version() -> tuple:
    try:
        return tuple(int(x) for x in st.__version__.split(".")[:3] if x.isdigit())
    except Exception:
        return (0,)


# streamlit >= 1.49 reemplaza use_container_width por width="stretch"
if _streamlit_version() >= (1, 49):
    W = {"width": "stretch"}
else:
    W = {"use_container_width": True}


from etf_agent import relationships as rel
from etf_agent.config import (
    DEFAULT_FACTORS,
    DEFAULT_TICKERS,
    TICKER_INFO,
    ALL_WINDOWS,
    DEFAULT_WINDOWS,
)
from etf_agent.correlation import (
    average_rolling_correlation,
    clustered_order,
    correlation_matrix,
    log_returns,
    rolling_correlation,
    top_pairs,
)
from etf_agent.data import fetch_prices, load_cached_prices, make_demo_prices

st.set_page_config(page_title="Agente de correlaciones entre ETFs", layout="wide")

# ---------------------------------------------------------------------------
# Barra lateral: configuración
# ---------------------------------------------------------------------------
st.title("🔎 Agente de correlaciones y relaciones entre ETFs")
st.markdown(
    "Explora correlaciones **históricas, YTD, 3M y 1M** y busca relaciones "
    "(lead-lag, cointegración, betas, factores latentes, regímenes de estrés...)."
)

with st.sidebar:
    st.header("Datos")
    modo = st.radio(
        "Fuente de datos",
        ["Demo (sintético)", "Cache local", "Descargar (Yahoo/Stooq)"],
        index=0,
        help="El modo demo funciona sin internet y sirve para probar el agente.",
    )

    # --- Añadir un ETF que no esté en la lista predefinida ---
    def _anadir_etf():
        """Callback: se ejecuta antes de instanciar los widgets en el rerun,
        por eso sí podemos modificar session_state aquí."""
        t = (st.session_state.get("nuevo_ticker") or "").strip().upper()
        if not t:
            return
        extras = st.session_state.setdefault("etfs_extra", [])
        if t not in extras:
            extras.append(t)
        seleccion = st.session_state.setdefault("sel_tickers", list(DEFAULT_TICKERS))
        if t not in seleccion:
            seleccion.append(t)
        st.session_state["nuevo_ticker"] = ""  # limpia el cuadro de texto

    st.markdown("**¿Falta algún ETF?** Escribe su símbolo y pulsa *Añadir*:")
    st.text_input(
        "Símbolo extra",
        key="nuevo_ticker",
        placeholder="VWO, AGG, VIXY, SMH...",
        label_visibility="collapsed",
    )
    st.button("➕ Añadir ETF", on_click=_anadir_etf)

    opciones = sorted(
        set(DEFAULT_TICKERS) | set(TICKER_INFO) | set(st.session_state.get("etfs_extra", []))
    )
    tickers = st.multiselect(
        "ETFs", options=opciones, default=list(DEFAULT_TICKERS), key="sel_tickers"
    )
    if not tickers:
        st.warning("Selecciona al menos un ETF.")
        st.stop()

    st.header("Ventanas de correlación")
    windows = st.multiselect(
        "Ventanas",
        options=sorted(ALL_WINDOWS),
        default=DEFAULT_WINDOWS,
        format_func=lambda k: f"{k} — {ALL_WINDOWS[k].description}",
    )

    st.header("Parámetros de análisis")
    rolling_window = st.slider("Ventana de correlación móvil (sesiones)", 20, 252, 63, step=5)
    max_lag = st.slider("Rezagos máximos (lead-lag / Granger)", 1, 20, 10)
    factors = st.multiselect("Factores para betas", options=tickers,
                             default=[f for f in DEFAULT_FACTORS if f in tickers])


# ---------------------------------------------------------------------------
# Carga de datos (cacheada)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Cargando precios...")
def cargar_precios(modo: str, tickers: tuple) -> pd.DataFrame:
    tickers = list(tickers)
    if modo.startswith("Demo"):
        return make_demo_prices(tickers=tickers, start="2015-01-01")
    if modo.startswith("Cache"):
        try:
            return load_cached_prices(tickers=tickers)
        except FileNotFoundError:
            st.warning("No hay cache local; usando datos sintéticos.")
            return make_demo_prices(tickers=tickers, start="2015-01-01")
    try:
        return fetch_prices(tickers=tickers, verbose=False)
    except Exception as err:
        st.warning(f"No se pudo descargar ({err}). Usando datos sintéticos.")
        return make_demo_prices(tickers=tickers, start="2015-01-01")


prices = cargar_precios(modo, tuple(tickers))
if prices.empty or prices.shape[1] < 2:
    st.error("Se necesitan al menos 2 ETFs con datos.")
    st.stop()

rets = log_returns(prices)
n_sesiones = len(prices)

st.caption(
    f"**{prices.shape[1]} ETFs** · {n_sesiones} sesiones · "
    f"{prices.index.min():%Y-%m-%d} → {prices.index.max():%Y-%m-%d} · "
    f"modo: {modo}"
)

# ---------------------------------------------------------------------------
# Pestañas
# ---------------------------------------------------------------------------
tab_resumen, tab_corr, tab_roll, tab_rel, tab_red = st.tabs(
    ["📊 Resumen", "🔥 Correlaciones por ventana", "📈 Correlación móvil",
     "🧠 Otras relaciones", "🕸️ Red de correlación"]
)

# ------------------------------- Resumen ------------------------------------
with tab_resumen:
    st.subheader("Universo analizado")
    info = pd.DataFrame(
        [
            {
                "ETF": t,
                "clase": TICKER_INFO.get(t, {}).get("clase", "—"),
                "nombre": TICKER_INFO.get(t, {}).get("name", "—"),
                "vol_anual_%": round(rets[t].std() * np.sqrt(252) * 100, 1),
                "ret_acum_%": round((prices[t].iloc[-1] / prices[t].iloc[0] - 1) * 100, 1),
            }
            for t in prices.columns
        ]
    )
    st.dataframe(info, **W, hide_index=True)

    st.subheader("Volatilidad y riesgo")
    vol = rel.volatility_table(rets)
    mdd = rel.max_drawdown(prices)
    vol["max_drawdown_%"] = vol["etf"].map(mdd).round(1)
    st.dataframe(vol.round(3), **W, hide_index=True)

    st.subheader("Evolución comparada (base 100)")
    norm = prices / prices.iloc[0] * 100
    st.line_chart(np.log(norm), **W)

# --------------------------- Correlaciones ----------------------------------
with tab_corr:
    st.subheader("Matrices de correlación por ventana")
    col1, col2 = st.columns(2)
    ventana = col1.selectbox("Ventana", options=windows, index=len(windows) - 1)
    metodo = col2.radio("Método", ["pearson", "spearman"], horizontal=True)

    from etf_agent.config import resolve_windows
    from etf_agent.correlation import window_slice

    win = resolve_windows([ventana])[ventana]
    idx = window_slice(prices.index, win)
    sub = rets.loc[rets.index.isin(idx)].dropna(how="all")
    st.info(
        f"Ventana **{ventana}** ({win.description}): {idx.min():%Y-%m-%d} → "
        f"{idx.max():%Y-%m-%d} · **{len(sub)}** retornos diarios"
    )
    if len(sub) < 5:
        st.error("Datos insuficientes en esta ventana. Elige una ventana más larga.")
        st.stop()

    corr = correlation_matrix(sub, method=metodo)
    import plotly.express as px

    fig = px.imshow(
        corr,
        text_auto=".2f" if corr.shape[0] <= 15 else False,
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        title=f"Correlación {metodo} — {ventana}",
        aspect="auto",
    )
    fig.update_layout(height=650)
    st.plotly_chart(fig, **W)

    st.subheader(f"Top pares más correlacionados ({ventana})")
    tp = top_pairs(corr, n=15)
    tp_display = tp.copy()
    tp_display["clase_A"] = tp_display["A"].map(lambda t: TICKER_INFO.get(t, {}).get("clase", "—"))
    tp_display["clase_B"] = tp_display["B"].map(lambda t: TICKER_INFO.get(t, {}).get("clase", "—"))
    st.dataframe(tp_display.round(3), **W, hide_index=True)

    st.subheader("Comparativa entre ventanas (mismo par)")
    pares = [f"{a}-{b}" for a, b in zip(tp["A"], tp["B"])][:8]
    if pares:
        par = st.selectbox("Par", options=pares)
        a, b = par.split("-")
        comp = {}
        for w in windows:
            w_idx = window_slice(prices.index, resolve_windows([w])[w])
            w_sub = rets.loc[rets.index.isin(w_idx)]
            comp[w] = w_sub[a].corr(w_sub[b])
        comp_df = pd.Series(comp, name="correlación").to_frame()
        comp_df["descripción"] = [ALL_WINDOWS[w].description for w in comp_df.index]
        st.dataframe(comp_df.round(3), **W)
        st.bar_chart(comp_df["correlación"])

# ------------------------ Correlación móvil ---------------------------------
with tab_roll:
    st.subheader(f"Correlación móvil ({rolling_window} sesiones ≈ "
                 f"{rolling_window / 21:.1f} meses)")
    opciones = [f"{a} / {b}" for i, a in enumerate(prices.columns) for b in list(prices.columns)[i + 1:]]
    sel = st.multiselect("Pares a representar", options=opciones,
                         default=opciones[:4] if len(opciones) >= 4 else opciones)
    chart_data = {}
    for par in sel:
        a, b = [x.strip() for x in par.split("/")]
        s = rolling_correlation(rets, a, b, window=rolling_window)
        chart_data[f"{a}-{b}"] = s
    if chart_data:
        df = pd.DataFrame(chart_data)
        st.line_chart(df, **W)
        st.caption("Correlación 1 = se mueven igual; -1 = en espejo; 0 = independientes.")

    st.subheader("Correlación media del universo")
    avg = average_rolling_correlation(rets, window=rolling_window)
    st.line_chart(avg, **W)
    st.caption("Picos = episodios de 'todo cae junto' ( crisis / estrés).")

# ------------------------- Otras relaciones --------------------------------
with tab_rel:
    st.subheader("⏪ Lead-lag: ¿quién anticipa a quién?")
    ll = rel.lead_lag_table(rets, max_lag=max_lag)
    if ll.empty:
        st.info("No se han encontrado relaciones lead-lag por encima del umbral.")
    else:
        st.dataframe(ll.head(20).round(3), **W, hide_index=True)
        st.caption("`best_lag_days` > 0 → `leader` se mueve antes que `follower`.")

    st.subheader("🔀 Causalidad de Granger")
    gc = rel.granger_table(rets, max_lag=min(max_lag, 10))
    sig = gc[gc["significativo"]] if not gc.empty else gc
    st.write(f"Pares con p-value < 0.05: **{len(sig)}** de {len(gc)} tests "
             f"(¡cuidado con los falsos positivos!)")
    st.dataframe(gc.head(20).round(4), **W, hide_index=True)

    st.subheader("🔗 Cointegración (relaciones de largo plazo)")
    co = rel.cointegration_table(prices)
    st.write(f"Pares cointegrados (p < 0.05): **{int(co['cointegrada'].sum())}**")
    st.dataframe(co.head(20).round(4), **W, hide_index=True)
    if not co.empty and bool(co.iloc[0]["cointegrada"]):
        par = st.selectbox(
            "Ver spread de un par cointegrado",
            options=[f"{r.A} / {r.B}" for r in co.head(8).itertuples()],
        )
        a, b = [x.strip() for x in par.split("/")]
        row = co[(co["A"] == a) & (co["B"] == b)].iloc[0]
        spread = rel.spread_series(prices, a, b, hedge_ratio=float(row["hedge_ratio"]))
        z = (spread - spread.mean()) / spread.std()
        st.line_chart(z.rename(f"z-score del spread {a}-{b}"), **W)
        st.caption(f"Vida media estimada: {row['half_life_days']:.1f} sesiones · "
                   f"z-score actual: {row['spread_zscore']:.2f}")

    st.subheader("📐 Exposición factorial (betas)")
    fac = [f for f in factors if f in rets.columns]
    if not fac:
        st.info("Selecciona al menos un factor en la barra lateral.")
    else:
        betas = rel.factor_betas(rets, factors=fac)
        st.dataframe(betas.round(3), **W, hide_index=True)
        beta_asset = st.selectbox("Beta móvil vs factor", options=[t for t in prices.columns])
        beta_bench = st.selectbox("Factor", options=fac)
        st.line_chart(rel.rolling_beta(rets, beta_asset, beta_bench, window=rolling_window),
                      use_container_width=True)

    st.subheader("🧩 Factores latentes (PCA)")
    n_pc = st.slider("Nº de componentes", 2, min(6, prices.shape[1]), 3)
    pca = rel.pca_factors(rets, n_components=n_pc)
    evr = pca["explained_variance_ratio"]
    st.bar_chart(evr.to_frame("fracción de varianza"))
    st.write(f"Los {n_pc} primeros componentes explican **{evr.sum():.1%}** de la varianza.")
    st.dataframe(pca["loadings"].round(3), **W)

    st.subheader("🧮 Dependencia no lineal (información mutua)")
    mi = rel.mutual_information_matrix(rets)
    fig_mi = px.imshow(mi, text_auto=".2f" if mi.shape[0] <= 15 else False,
                       color_continuous_scale="Blues", aspect="auto",
                       title="Información mutua (nats)")
    fig_mi.update_layout(height=600)
    st.plotly_chart(fig_mi, **W)
    st.caption("Pares con MI alta pero correlación baja → relación no lineal.")

    st.subheader("⚠️ Correlación en regímenes de estrés")
    bench = st.selectbox("Benchmark", options=list(prices.columns),
                         index=list(prices.columns).index("SPY") if "SPY" in prices.columns else 0)
    reg = rel.regime_correlation(rets, benchmark=bench)
    st.dataframe(reg.round(3), **W, hide_index=True)
    st.caption("`asimetria` > 0 → la correlación sube en las caídas (estrés sistémico).")

    st.subheader("📉 Drawdowns del benchmark")
    dd = rel.drawdown_episodes(prices, benchmark=bench)
    if dd.empty:
        st.info("No hay drawdowns > 10% en el periodo.")
    else:
        st.dataframe(dd, **W, hide_index=True)

    st.subheader("🌳 Agrupamiento (clustering jerárquico)")
    corr_full = correlation_matrix(rets, method="pearson")
    clusters = rel.cluster_etfs(corr_full, n_clusters=min(5, prices.shape[1]))
    st.dataframe(clusters, **W, hide_index=True)

# ---------------------------- Red de correlación ----------------------------
with tab_red:
    st.subheader("Red de correlaciones")
    umbral = st.slider("Umbral de |correlación|", 0.0, 1.0, 0.6, step=0.05)
    corr_full = correlation_matrix(rets, method="pearson")
    order = clustered_order(corr_full)
    corr_ord = corr_full.loc[order, order]

    import plotly.graph_objects as go

    n = len(order)
    fig_net = go.Figure()
    # posiciones en círculo
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pos = {t: (np.cos(a), np.sin(a)) for t, a in zip(order, ang)}
    edges_x, edges_y, edges_w = [], [], []
    for i, a in enumerate(order):
        for j, b in enumerate(order):
            if i < j:
                v = corr_full.loc[a, b]
                if pd.notna(v) and abs(v) >= umbral:
                    edges_x += [pos[a][0], pos[b][0], None]
                    edges_y += [pos[a][1], pos[b][1], None]
                    edges_w.append(abs(float(v)))
    fig_net.add_trace(go.Scatter(
        x=edges_x, y=edges_y, mode="lines",
        line=dict(width=1, color="#a0aec0"), hoverinfo="none",
    ))
    fig_net.add_trace(go.Scatter(
        x=[pos[t][0] for t in order], y=[pos[t][1] for t in order],
        mode="markers+text", text=order, textposition="top center",
        marker=dict(size=18, color="#2b6cb0"),
        hovertext=[f"{t} ({TICKER_INFO.get(t, {}).get('clase', '—')})" for t in order],
    ))
    fig_net.update_layout(
        showlegend=False, height=650,
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        title=f"Aristas con |corr| ≥ {umbral:.2f}",
    )
    st.plotly_chart(fig_net, **W)

    st.subheader("Matriz ordenada por similitud")
    fig_ord = px.imshow(corr_ord, text_auto=".2f" if n <= 15 else False,
                        color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto")
    fig_ord.update_layout(height=650)
    st.plotly_chart(fig_ord, **W)

st.divider()
st.caption("etf_agent · la correlación no implica causalidad. Ventanas cortas = mucho ruido.")
