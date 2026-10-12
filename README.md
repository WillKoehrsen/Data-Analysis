# Data-Analysis
Data Science Projects Using Python and a little R. 

This repository is a place to share my code and notebooks for numerous data science projects.

Many of these projects are written up on my [Towards Data Science Medium page](https://medium.com/@williamkoehrsen). 

Reach me on Twitter [@koehrsen_will](https://twitter.com/@koehrsen_will)

## ETF Correlation Agent

`etf_correlation_agent/` — agente (Python + CLI + dashboard Streamlit) para buscar
correlaciones y relaciones entre los ETFs más significativos (SPY, QQQ, TLT, GLD, USO,
EFA, EEM, UUP, sectores, semiconductores...) en ventanas de histórico, YTD, 3M y 1M,
además de lead-lag, cointegración, Granger, betas factoriales, PCA, clustering,
información mutua y regímenes de estrés. Ver su [README](etf_correlation_agent/README.md).

```bash
cd etf_correlation_agent
pip install -r requirements.txt
python -m etf_agent demo        # datos sintéticos, sin internet
python -m etf_agent analyze     # datos reales
python -m etf_agent dashboard   # dashboard interactivo
```
