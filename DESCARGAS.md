# 📥 Descargas — Agente de correlaciones entre ETFs

Enlaces directos para bajar el paquete a tu PC (Windows):

## 1) Paquete completo (recomendado) — 9,4 MB

**Enlace directo de descarga** (clic derecho → "Guardar enlace como..."):

https://raw.githubusercontent.com/pablodpt/Data-Analysis/arena/93e4c546-data-analysis/etf_correlation_agent_Windows.zip

O desde la página del fichero (botón **Download**):

https://github.com/pablodpt/Data-Analysis/blob/arena/93e4c546-data-analysis/etf_correlation_agent_Windows.zip

Incluye:
- Código del agente (`etf_agent/`), tests, notebook, `requirements.txt`
- Lanzadores `.bat` para Windows: `INSTALAR_WINDOWS.bat`, `EJECUTAR_DEMO.bat`,
  `EJECUTAR_ANALISIS.bat`, `EJECUTAR_DASHBOARD.bat`, `TESTS.bat`
- `LEEME_PRIMERO.txt` (instrucciones en español)
- **Informe de ejemplo ya generado**: `reports/…/report.html` + 14 gráficos PNG + 22 tablas CSV

## 2) Solo código — 50 KB

https://raw.githubusercontent.com/pablodpt/Data-Analysis/arena/93e4c546-data-analysis/etf_correlation_agent_codigo.zip

## 3) Todo el repositorio (opcional)

https://codeload.github.com/pablodpt/Data-Analysis/zip/refs/heads/arena/93e4c546-data-analysis

---

## Instalación en Windows (3 pasos)

1. Descomprime el zip (clic derecho → *Extraer todo*).
2. Instala **Python 3.10+** desde https://www.python.org/downloads/
   marcando la casilla **"Add Python to PATH"**.
3. Doble clic en **`INSTALAR_WINDOWS.bat`**.

## Uso

| Archivo | Qué hace |
|---|---|
| `EJECUTAR_DEMO.bat` | Análisis de prueba con datos sintéticos (**no necesita internet**) |
| `EJECUTAR_ANALISIS.bat` | Análisis con datos reales (descarga de Yahoo Finance) |
| `EJECUTAR_DASHBOARD.bat` | Dashboard interactivo en http://localhost:8501 |
| `TESTS.bat` | Comprueba que todo funciona |

Después de cada análisis, abre `reports\<carpeta con fecha>\report.html` con el navegador.

---

Documentación completa: [`etf_correlation_agent/README.md`](etf_correlation_agent/README.md)
