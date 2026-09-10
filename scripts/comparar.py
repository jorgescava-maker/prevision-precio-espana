"""
Compara una predicción candidata contra la predicción de referencia publicada
(la de producción, ya guardada: no hace falta reentrenar nada para medirse con
ella).

La candidata es un CSV o parquet con una fila por periodo de casación:

    period_start_utc   inicio del periodo en UTC (sin zona), como en la referencia
    <columna>          la predicción, en EUR/MWh (por defecto se llama `pred`)

Lo que decide es el MAE agregado sobre los MISMOS periodos. Si la candidata
no cubre todos los de la referencia, el script lo dice: un MAE sobre menos
filas no es comparable con el publicado, porque puede que sean las fáciles.

El desglose mensual, el MAE de la media diaria y el test de Diebold-Mariano
están para ENTENDER el resultado, no son requisitos que haya que pasar.

Uso:
    python -m scripts.comparar mi_prediccion.csv
    python -m scripts.comparar mi_prediccion.parquet --columna pred_mia --por-mes
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from etl.common import logging_config  # noqa: F401 - fuerza stdout a UTF-8

ROOT =Path(__file__).resolve().parents[1]
REFERENCIA = ROOT / "previsiones" / "walkforward" / "espana_referencia.csv"
TZ = "Europe/Madrid"


def _leer(ruta: Path) -> pd.DataFrame:
    df = pd.read_parquet(ruta) if ruta.suffix == ".parquet" else pd.read_csv(ruta)
    df["period_start_utc"] = pd.to_datetime(df["period_start_utc"])
    if getattr(df["period_start_utc"].dt, "tz", None) is not None:
        df["period_start_utc"] = df["period_start_utc"].dt.tz_convert("UTC").dt.tz_localize(None)
    return df


def diebold_mariano(dif_diaria: np.ndarray, lags: int = 7) -> tuple[float, float]:
    """DM sobre la diferencia diaria de error absoluto medio (referencia −
    candidata), con varianza de Newey-West. Positivo = la candidata tiene
    menos error. Devuelve (estadístico, p-valor bilateral, aproximación normal)."""
    d = dif_diaria[~np.isnan(dif_diaria)]
    n = len(d)
    if n < 10:
        return float("nan"), float("nan")
    u = d - d.mean()
    lrv = u @ u / n
    for k in range(1, min(lags, n - 1) + 1):
        lrv += 2 * (1 - k / (lags + 1)) * (u[k:] @ u[:-k]) / n
    if lrv <= 0:
        return float("nan"), float("nan")
    stat = d.mean() / math.sqrt(lrv / n)
    p = math.erfc(abs(stat) / math.sqrt(2))
    return stat, p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("candidata", type=Path)
    ap.add_argument("--columna", default="pred", help="columna de la predicción candidata")
    ap.add_argument("--referencia", type=Path, default=REFERENCIA)
    ap.add_argument("--por-mes", action="store_true", help="desglose mensual")
    args = ap.parse_args()

    ref = _leer(args.referencia)
    cand = _leer(args.candidata)
    if args.columna not in cand.columns:
        raise SystemExit(f"La candidata no tiene la columna '{args.columna}'. Columnas: {list(cand.columns)}")
    if cand["period_start_utc"].duplicated().any():
        raise SystemExit("La candidata tiene periodos repetidos.")

    d = ref.merge(cand[["period_start_utc", args.columna]].rename(columns={args.columna: "pred_candidata"}),
                  on="period_start_utc", how="left")
    faltan = int(d["pred_candidata"].isna().sum())
    d = d.dropna(subset=["pred_candidata"])
    if d.empty:
        raise SystemExit("Ningún periodo en común con la referencia.")

    e_ref = (d["price_real"] - d["pred_produccion"]).abs()
    e_can = (d["price_real"] - d["pred_candidata"]).abs()
    mae_ref, mae_can = e_ref.mean(), e_can.mean()

    print(f"Referencia: {args.referencia.relative_to(ROOT) if args.referencia.is_relative_to(ROOT) else args.referencia}")
    print(f"Periodos de la referencia: {len(ref):,} · en común: {len(d):,} · la candidata no cubre: {faltan:,}")
    if faltan:
        print("  AVISO: la candidata no cubre todos los periodos. El MAE de abajo se calcula sobre los")
        print("  comunes y NO es comparable con el MAE publicado de la referencia.")
    print()
    print(f"  MAE referencia (producción): {mae_ref:8.3f} EUR/MWh")
    print(f"  MAE candidata:               {mae_can:8.3f} EUR/MWh")
    print(f"  diferencia:                  {mae_can - mae_ref:+8.3f} EUR/MWh ({(mae_can / mae_ref - 1) * 100:+.2f} %)")

    d["dia"] = d["period_start_utc"].dt.tz_localize("UTC").dt.tz_convert(TZ).dt.date
    diario = d.groupby("dia").agg(real=("price_real", "mean"), ref=("pred_produccion", "mean"),
                                   can=("pred_candidata", "mean"), n=("price_real", "size"))
    # Misma definición que la cifra publicada: días locales COMPLETOS de la era
    # de 15 minutos (desde el 2025-10-01; 92, 96 o 100 periodos según el cambio
    # de hora). Los días de la era horaria y los incompletos no entran.
    completos = diario[diario["n"] >= 92]
    print(f"\n  MAE de la media diaria — referencia {(completos['real'] - completos['ref']).abs().mean():.3f} · "
          f"candidata {(completos['real'] - completos['can']).abs().mean():.3f}  "
          f"({len(completos)} días completos de 15 min)")

    d["dif"] = e_ref - e_can
    stat, p = diebold_mariano(d.groupby("dia")["dif"].mean().to_numpy())
    print(f"  Diebold-Mariano (diario, Newey-West 7): estadístico {stat:+.2f} · p {p:.3f}  "
          f"(positivo = la candidata falla menos)")

    if args.por_mes:
        d["mes"] = d["period_start_utc"].dt.to_period("M")
        m = d.groupby("mes").apply(lambda g: pd.Series({
            "n": len(g),
            "ref": (g["price_real"] - g["pred_produccion"]).abs().mean(),
            "cand": (g["price_real"] - g["pred_candidata"]).abs().mean(),
        }), include_groups=False)
        m["dif"] = m["cand"] - m["ref"]
        print("\n" + m.to_string(float_format=lambda x: f"{x:.2f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
