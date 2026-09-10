# Datos: de dónde salen y qué se puede hacer con ellos

**El código de este repositorio es MIT. Los datos no.** Cada serie conserva las
condiciones de su fuente, y el dataset publicado junta varias, así que hereda
la más restrictiva: **uso informativo, de investigación o docente, no
comercial, citando las fuentes.**

## Fuentes

| Serie | Fuente | Condiciones | Dónde se usa |
|---|---|---|---|
| Precio del mercado diario (España y Portugal) | [OMIE](https://www.omie.es) | La información de su web es pública y gratuita, y puede usarse libremente siempre que se respete su contenido original ([aviso legal](https://www.omie.es/es/aviso-legal)) | objetivo, retardos de precio, análogos |
| Previsión D-1 de demanda, eólica y solar; generación por tecnología | [e·sios, Red Eléctrica](https://www.esios.ree.es) | Publicación con fines informativos y **no comerciales**, citando a Red Eléctrica como fuente y la fecha de actualización ([aviso legal](https://www.ree.es/es/aviso-legal)) | variables D-1, climatologías, valor del agua |
| Capacidad instalada, indisponibilidad de centrales, flujos, NTC, embalses; precio, previsión y generación de Francia | [ENTSO-E Transparency Platform](https://transparency.entsoe.eu) | Datos abiertos, CC BY 4.0 ([condiciones](https://transparency.entsoe.eu/content/static_content/Static%20content/terms%20and%20conditions/terms%20and%20conditions.html)) | disponibilidad real, interconexión, dataset de Francia |
| Precio day-ahead de Alemania | [SMARD, Bundesnetzagentur](https://www.smard.de) | CC BY 4.0 | retardos de precio del dataset de Francia |
| Temperatura (ERA5) | [Open-Meteo](https://open-meteo.com), datos de Copernicus ERA5 | CC BY 4.0 | grados-día del dataset de Francia |
| TTF, EUA (proxy `CO2.L`), carbón API2, EUR/USD | Yahoo Finance | **No se redistribuyen.** Sus términos no lo permiten | coste de gas y CO₂, motor de despacho |
| Festivos | librería [`holidays`](https://pypi.org/project/holidays/) | cálculo local | calendario |

## Qué se publica y qué no

- **`previsiones/`** (las previsiones diarias y la de referencia del backtest):
  son elaboración propia. Los ficheros llevan también el precio real de OMIE
  para poder medirlas.
- **Datasets de variables** (adjuntos de cada
  [release](../../releases)): todas las variables del modelo, ya construidas y
  sin fuga (solo información disponible antes del cierre de la subasta),
  **menos las columnas de Yahoo Finance**: `ttf_eur_mwh`, `eua_eur_t`,
  `coal_eur_mwh_th` (y en Francia también `coal_usd_t` y `eur_usd`). Las
  variables que el modelo deriva de ellas (el precio simulado del motor de
  despacho, los análogos) sí van, porque son elaboración propia.
  `scripts/completar_materias_primas.py` descarga las series y las vuelve a
  unir exactamente como las usa el modelo.
- **Bases DuckDB** (`data/`): no se publican. Se construyen con los scripts
  `etl/backfill_*.py` y tus propios tokens (gratuitos, ver `.env.example`).

## Cómo citar

> Fuentes: OMIE; Red Eléctrica (e·sios); ENTSO-E Transparency Platform;
> Bundesnetzagentur | SMARD.de; Open-Meteo / Copernicus ERA5. Elaboración:
> Jorge Sánchez Cava, [prevision-precio-espana](https://github.com/jorgescava-maker/prevision-precio-espana).
