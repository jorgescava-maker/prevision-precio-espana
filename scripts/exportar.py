"""
Exporta lo que se publica del backtest:

  previsiones/walkforward/espana_referencia.csv
      La predicción de producción del walk-forward mensual (la referencia con
      la que se compara cualquier propuesta, ver scripts/comparar.py), con sus
      miembros por separado.

  dist/dataset_espana.parquet, dist/dataset_francia.parquet
      Las variables D-1 ya construidas, SIN las series de Yahoo Finance
      (TTF, EUA, carbón API2, EUR/USD), cuyos términos no permiten
      redistribuirlas. `scripts/completar_materias_primas.py` las descarga y
      las vuelve a unir exactamente igual. Estos ficheros se suben como
      adjuntos de una release de GitHub, no al historial de git.

Uso:
    python -m scripts.exportar
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pandas as pd

from etl.common import logging_config  # noqa: F401 - fuerza stdout a UTF-8

ROOT =Path(__file__).resolve().parents[1]
CONJUNTO = ROOT / "studies" / "precio_da_conjunto" / "output"
DIST = ROOT / "dist"

# Columnas que son directamente (o por conversión de unidades) una serie de
# Yahoo Finance. Las derivadas por el modelo (análogos, precio simulado del
# motor de despacho) son elaboración propia y se quedan.
COLUMNAS_YAHOO = ["ttf_eur_mwh", "eua_eur_t", "coal_eur_mwh_th", "coal_usd_t", "eur_usd"]

MIEMBROS = {
    "pred_final": "pred_produccion",
    "pred_pooled": "pred_agrupado",
    "pred_por_hora": "pred_por_hora",
    "pred_red_pooled": "pred_redes",
    "pred_ensemble3": "pred_mezcla",
    "pred_ensemble3_cal": "pred_mezcla_calibrada",
    "pred_zero": "filtro_precio_cero",
}


def main() -> int:
    # El backtest con el valor del agua causal (agua_causal.py, findings.md
    # #146), no el de ensemble_tres_miembros.py, que tiene esa fuga.
    ref = pd.read_parquet(CONJUNTO / "predicciones_agua_causal.parquet")
    ref = ref[["period_start_utc", "price_real"] + list(MIEMBROS)].rename(columns=MIEMBROS)
    ref = ref.sort_values("period_start_utc")
    destino = ROOT / "previsiones" / "walkforward"
    destino.mkdir(parents=True, exist_ok=True)
    ref.to_csv(destino / "espana_referencia.csv", index=False, float_format="%.3f")
    mae = (ref["price_real"] - ref["pred_produccion"]).abs().mean()
    print(f"Referencia: {len(ref):,} periodos, {ref['period_start_utc'].min()} → "
          f"{ref['period_start_utc'].max()} · MAE {mae:.3f}")

    DIST.mkdir(exist_ok=True)
    for nombre, ruta in (("espana", ROOT / "studies" / "precio_da_mejor_modelo" / "output" / "dataset.parquet"),
                         ("francia", ROOT / "studies" / "precio_da_francia" / "output" / "dataset.parquet")):
        df = pd.read_parquet(ruta)
        quitadas = [c for c in COLUMNAS_YAHOO if c in df.columns]
        df.drop(columns=quitadas).to_parquet(DIST / f"dataset_{nombre}.parquet", index=False)
        print(f"dataset_{nombre}: {len(df):,} filas, {df.shape[1] - len(quitadas)} columnas "
              f"(quitadas: {', '.join(quitadas)})")

    # Los datasets de cada mes de corte del walk-forward causal (findings.md
    # #146): el de arriba lleva el valor del agua de la superficie completa,
    # bueno para predecir hoy pero con fuga para un backtest. Con estos, el
    # backtest honesto se rehace sin montar las bases.
    causales = sorted((CONJUNTO / "agua_causal").glob("dataset_*.parquet"))
    with zipfile.ZipFile(DIST / "datasets_causales.zip", "w", compression=zipfile.ZIP_STORED) as z:
        for f in causales:
            buf = io.BytesIO()
            pd.read_parquet(f).drop(columns=COLUMNAS_YAHOO, errors="ignore").to_parquet(buf, index=False)
            z.writestr(f.name, buf.getvalue())
    print(f"datasets_causales.zip: {len(causales)} meses de corte")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
