# Agente de correlaciones y relaciones entre ETFs 🔎

Herramienta (paquete Python + CLI + dashboard) para **buscar correlaciones y cualquier
tipo de relación** entre los ETFs más significativos, en distintas ventanas temporales:
**histórico completo, YTD, 3M y 1M** (más 6M, 1Y, 3Y, 5Y opcionales).

```
etf_correlation_agent/
├── etf_agent/            # paquete Python
│   ├── config.py         # universo de ETFs, ventanas, factores
│   ├── data.py           # descarga (Yahoo/Stooq), cache y datos sintéticos
│   ├── correlation.py    # matrices de correlación, rolling, red, clustering
│   ├── relationships.py  # lead-lag, Granger, cointegración, betas, PCA, MI, regímenes
│   ├── report.py         # informe Markdown + HTML + CSV + gráficos PNG
│   ├── cli.py            # línea de comandos
│   └── dashboard.py      # dashboard interactivo (Streamlit)
├── tests/                # tests (datos sintéticos, sin internet)
├── notebooks/            # guía de uso en Jupyter
└── requirements.txt
```

---

## 1. Instalación

```bash
cd etf_correlation_agent
pip install -r requirements.txt
```

> El `requirements.txt` de este proyecto es independiente del del repositorio (que es
> antiguo). Se recomienda un entorno virtual propio.

## 2. Universo de ETFs

| Bloque | ETFs |
|---|---|
| Renta variable USA | `SPY`, `QQQ`, `IWM` |
| Renta variable internacional | `EFA` (desarrollados ex-NA), `EEM` (emergentes) |
| Renta fija / tipos | `TLT` (20+ años), `IEF` (7-10 años), `TIP` (TIPS/inflación), `SGOV` (letras, cash-like) |
| Crédito | `LQD` (investment grade), `HYG` (high yield) |
| Divisas / materias primas | `UUP` (dólar DXY), `GLD` (oro), `USO` (petróleo), `DBC` (cesta de commodities) |
| Sectores (rotación) | `XLK` (tech), `XLF` (financiero), `XLE` (energía), `XLU` (utilities), `XLP` (consumo defensivo) |
| Semiconductores | `SOXX` |

**21 ETFs por defecto.**

### ¿Y DRAM (memoria)?

`DRAM` **no está en el universo por defecto** a propósito: su histórico es muy corto y,
al forzar un periodo común más pequeño, recortaba en silencio los análisis
multivariante (PCA, betas, Granger, cointegración, información mutua), "ensuciando" el
estudio. Si quieres incluirlo, añádelo tú (sección 3b): el agente te avisará de cuántas
sesiones se descartan por ese motivo.

Opcionales documentados en `config.OPTIONAL_TICKERS`: `DRAM`, `VWO`, `SHY`, `BIL`, `AGG`, `VIXY`.
Puedes usar cualquier lista: `--tickers SPY,TLT,GLD,VWO`.

## 3. Uso rápido

### Modo demo (sin internet, datos sintéticos)

```bash
python -m etf_agent demo
```

Genera un informe completo con datos sintéticos que **sí** contienen correlaciones,
cointegración y relaciones lead-lag, para probar que todo funciona antes de conectar
datos reales.

### Análisis real

```bash
# universo completo, ventanas por defecto (max, ytd, 3m, 1m)
python -m etf_agent analyze

# ETFs y ventanas concretas
python -m etf_agent analyze --tickers SPY,TLT,GLD,UUP,USO --windows max,ytd,3m,1m,1y

# otras opciones útiles
python -m etf_agent analyze \
    --factors SPY,TLT,GLD,UUP \   # factores para las betas
    --rolling-window 63 \         # ventana de correlación móvil (sesiones)
    --max-lag 10 \                # rezagos para lead-lag/Granger
    --outdir reports
```

### Ver el universo y las ventanas disponibles

```bash
python -m etf_agent list
```

### Dashboard interactivo

```bash
python -m etf_agent dashboard          # http://localhost:8501
# o
streamlit run etf_correlation_agent/etf_agent/dashboard.py
```

Permite elegir ETFs, ventanas y umbrales, y ver heatmaps, correlación móvil, lead-lag,
Granger, cointegración, betas, PCA, agrupamientos, información mutua y regímenes de
estrés de forma interactiva.

### Desde Python / Jupyter

```python
from etf_agent import fetch_prices, correlation_matrices, log_returns, build_report

prices = fetch_prices()                      # descarga + cache en data/etf_cache/
mats = correlation_matrices(prices, windows=["max", "ytd", "3m", "1m"])
mats[("3m", "pearson")].round(2)             # matriz 3 meses

result = build_report(prices, outdir="reports")   # informe completo
print(result["html"])
```

Ver `notebooks/Guia_agente_correlaciones_ETFs.ipynb` para un recorrido completo.

## 3b. Añadir (o quitar) ETFs

Hay 4 formas, de más fácil a más flexible:

**A) En el dashboard** (lo más cómodo): en la barra lateral, debajo del selector de ETFs,
hay un cuadro de texto: escribe el símbolo (por ejemplo `VWO`, `AGG`, `VIXY`, `SMH`) y pulsa
**➕ Añadir ETF**. Se suma a la selección al instante. Para quitar uno, simplemente
desmárcalo de la lista.

**B) Editando el `.bat`**: abre `EJECUTAR_ANALISIS.bat` con el bloc de notas y cambia:

```bat
set TICKERS=SPY,TLT,GLD,UUP,VWO,AGG,SMH
set WINDOWS=max,ytd,3m,1m
```
Si dejas `set TICKERS=` vacío, se usa el universo completo de 22 ETFs.

**C) Por línea de comandos**:

```bash
python -m etf_agent analyze --tickers SPY,TLT,GLD,VWO --windows max,ytd,3m,1m
```

**D) Desde Python**: 

```python
from etf_agent import fetch_prices, build_report
prices = fetch_prices(tickers=["SPY", "TLT", "VWO", "AGG"])
build_report(prices, outdir="reports")
```

Notas:
- Cualquier símbolo que Yahoo Finance entienda vale (acciones, bonos, cripto...).
- Si un símbolo no existe o no tiene datos, el agente **avisa y lo excluye** en lugar de romperse.
- Los ETFs nuevos no aparecen en la tabla de "clase de activo" (saldá como "—"); si quieres
  que salga con descripción, añádelo a `TICKER_INFO` en `etf_agent/config.py`.
- Para añadirlos **de forma permanente** al universo por defecto, edita `DEFAULT_TICKERS`
  (y `TICKER_INFO`) en `etf_agent/config.py`.

## 4. Qué calcula exactamente

### Correlaciones (por ventana: `max`, `ytd`, `3m`, `1m`, ...)

| Análisis | Qué responde |
|---|---|
| Correlación de **Pearson** | relación lineal contemporánea |
| Correlación de **Spearman** | relación monótona robusta a outliers |
| Ranking de pares | qué pares se mueven más (y menos) juntos |
| Correlación **móvil** (63d por defecto) | cómo cambia la relación en el tiempo |
| Correlación media del universo | régimen: ¿"todo se mueve junto" ahora? |
| **Red de correlación** | qué ETFs forman grupos conectados |

### Otras relaciones (la parte "cualquier tipo de vínculo")

| Análisis | Qué responde |
|---|---|
| **Lead-lag** (correlación cruzada ±N días) | ¿qué ETF anticipa a qué otro? |
| **Causalidad de Granger** | ¿el pasado de X mejora la predicción de Y? |
| **Cointegración** (Engle-Granger) | pares con equilibrio de largo plazo: hedge ratio, vida media del spread, z-score |
| **Betas / factores** | sensibilidad de cada ETF al mercado (SPY), tipos (TLT), oro (GLD)… |
| **PCA** | factores latentes que explican el movimiento conjunto |
| **Clustering jerárquico** | grupos de ETFs con comportamiento similar |
| **Información mutua** | dependencia **no lineal** que la correlación no ve |
| **Regímenes de estrés** | correlación en los días malos vs. buenos del benchmark (asimetría) |
| **Volatilidad / drawdown** | perfil de riesgo de cada ETF |

## 5. Salida del informe

Cada ejecución crea `reports/etf_correlations_<modo>_<fecha>/` con:

```
report.md / report.html      informe completo (HTML auto-contenido con las imágenes incrustadas)
summary.json                 resumen estructulado con los hallazgos principales
images/                      heatmaps por ventana y método, precios normalizados,
                             correlación móvil, PCA, dendrograma, red de correlación
data/                        todas las tablas en CSV (matrices, lead-lag, Granger,
                             cointegración, betas, PCA, clusters, MI, regímenes, volatilidad)
```

## 6. Tests

```bash
python -m pytest tests/ -q
```

24 tests sobre datos sintéticos (no necesitan internet): configuración, cálculo de
correlaciones por ventana, todos los análisis de relaciones y generación del informe.

## 7. Notas metodológicas y limitaciones

- Retornos **logarítmicos diarios**; precios ajustados (dividendos/splits) vía `yfinance`.
- Ventanas naturales: `ytd` = desde el 1 de enero; `3m`/`1m` = últimos 3 meses / 1 mes.
- **Ventanas cortas = mucho ruido.** Con ~21 sesiones (1M) la correlación es inestable:
  úsala como diagnóstico reciente, no como verdad estructural.
- **Correlación ≠ causalidad.** Granger es predictibilidad, no causa; dos ETFs pueden
  moverse juntos por un factor común (mercado, tipos, dólar).
- La cointegración y los hedge ratios se recalibran: el spread puede romperse.
- Las comisiones, el rolado de futuros (USO, DBC, UUP) y el sesgo de supervivencia no
  están modelados.
- Datos en caché en `data/etf_cache/` (un CSV por ticker); borra la carpeta para forzar
  descarga limpia.

## 8. Fuentes de datos

1. **Yahoo Finance** vía `yfinance` (por defecto).
2. **Stooq** vía `pandas-datareader` (fallback automático si Yahoo falla, o `--source stooq`).

Si no hay internet, el modo `demo` sigue funcionando con datos sintéticos.
