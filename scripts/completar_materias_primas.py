"""
Vuelve a unir a los datasets publicados las columnas de Yahoo Finance que no
se pueden redistribuir (TTF, EUA, carbón API2 y EUR/USD), exactamente como las
construye el modelo: cierre de mercado con dos días de margen (D-2) respecto a
la hora de entrega, arrastrando como mucho 5 días el último cierre válido.

Antes hay que tener `data/commodities.duckdb`:
    python -m etl.backfill_commodity_prices
    python -m etl.backfill_fx_rates

Por defecto lee los datasets de la release en `dist/` y los deja donde los
busca el modelo, así que después se puede correr directamente
`python -m scripts.reconstruir walkforward`.

Uso:
    python -m scripts.completar_materias_primas
    python -m scripts.completar_materias_primas --comparar   # comprueba contra un dataset completo local
"""

from __future__ import annotations

import argparse
import io
import zipfile
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from etl.common import logging_config  # noqa: F401 - fuerza stdout a UTF-8
from studies.merit_order.build_dataset import COAL_MWH_PER_TONNE, _load_commodities

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
DESTINOS = {
    "espana": ROOT / "studies" / "precio_da_mejor_modelo" / "output" / "dataset.parquet",
    "francia": ROOT / "studies" / "precio_da_francia" / "output" / "dataset.parquet",
}
CAUSALES = ROOT / "studies" / "precio_da_conjunto" / "output" / "agua_causal"
COLUMNAS = {
    "espana": ["ttf_eur_mwh", "eua_eur_t", "coal_eur_mwh_th"],
    "francia": ["ttf_eur_mwh", "eua_eur_t", "coal_usd_t", "eur_usd", "coal_eur_mwh_th"],
}


def completar(df: pd.DataFrame, commodities: pd.DataFrame, columnas: list[str]) -> pd.DataFrame:
    """Mismo cruce que `merit_order/build_dataset.py` y `precio_da_francia/build_dataset.py`."""
    orden = df.index
    aux = pd.DataFrame({"fila": np.arange(len(df)),
                        "hour_utc": df["period_start_utc"].dt.floor("h")})
    aux["trade_date_cutoff"] = (aux["hour_utc"] - pd.Timedelta(days=2)).dt.normalize()
    aux = pd.merge_asof(aux.sort_values("trade_date_cutoff"), commodities.sort_values("trade_date"),
                        left_on="trade_date_cutoff", right_on="trade_date", direction="backward")
    aux["coal_eur_mwh_th"] = (aux["coal_usd_t"] / aux["eur_usd"]) / COAL_MWH_PER_TONNE
    aux = aux.sort_values("fila")
    out = df.copy()
    # Los constructores cruzan las materias primas sobre las filas de la
    # previsión D-1: donde no hay previsión (p. ej. el día sin datos de
    # ENTSO-E de Francia del 2023-04-18) el dataset original las deja vacías.
    sin_prevision = df["forecast_demanda_mw"].isna().to_numpy()
    for c in columnas:
        out[c] = np.where(sin_prevision, np.nan, aux[c].to_numpy())
    out.index = orden
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--origen", type=Path, default=DIST, help="carpeta con dataset_espana/francia.parquet")
    ap.add_argument("--comparar", action="store_true",
                    help="en vez de escribir, compara con el dataset completo que ya haya en el destino")
    args = ap.parse_args()

    con = duckdb.connect(str(ROOT / "data" / "commodities.duckdb"), read_only=True)
    try:
        commodities = _load_commodities(con)
    finally:
        con.close()

    trabajos = [(pais, pd.read_parquet(args.origen / f"dataset_{pais}.parquet"), destino, COLUMNAS[pais])
                for pais, destino in DESTINOS.items()]
    # Los datasets por mes de corte del walk-forward causal (findings.md #146).
    zip_causales = args.origen / "datasets_causales.zip"
    if zip_causales.exists():
        with zipfile.ZipFile(zip_causales) as z:
            for nombre in sorted(z.namelist()):
                trabajos.append((nombre.removesuffix(".parquet"), pd.read_parquet(io.BytesIO(z.read(nombre))),
                                 CAUSALES / nombre, COLUMNAS["espana"]))

    for nombre, publicado, destino, columnas in trabajos:
        completo = completar(publicado, commodities, columnas)
        if args.comparar:
            original = pd.read_parquet(destino).set_index("period_start_utc")
            nuevo = completo.set_index("period_start_utc").loc[original.index]
            distintas = {c: int((~np.isclose(original[c].to_numpy(float), nuevo[c].to_numpy(float),
                                             equal_nan=True, rtol=0, atol=1e-9)).sum()) for c in columnas}
            print(f"{nombre}: {len(original):,} filas · distintas por columna {distintas}")
            continue
        destino.parent.mkdir(parents=True, exist_ok=True)
        completo.to_parquet(destino, index=False)
        print(f"{nombre}: {len(completo):,} filas → {destino.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
