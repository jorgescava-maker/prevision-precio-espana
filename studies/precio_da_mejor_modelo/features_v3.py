"""Las diez variables de la revision de 2026-09-13, sobre el dataset de Espana.

**Que son.** Una auditoria del dataset contra las variables que consume el
modelo, mas una revision de la literatura de prevision de precio electrico
(EPF) centrada en CONJUNTOS DE VARIABLES DE ENTRADA y no en metodos, dejaron
diez variables que bajan el MAE de forma clara. **Ninguna anade un solo dato
nuevo**: todas son combinaciones o transformaciones de series que el modelo ya
tenia delante.

    reserve_margin_mw        ya se construye en `build_dataset.py` y nunca se
    ratio_renovable_periodo  habian conectado a SPAIN_ONLY. Coste cero.

    tension_fr               carga residual de Francia menos su nuclear
                             disponible: cuanto va a tirar el vecino.

    ltsc_90d  ltsc_365d      componente estacional de largo plazo (marco SCAR
    ltsc_pendiente           de Uniejewski y Weron). El modelo tenia EMAs de 7
    desvio_ltsc              y 28 dias, que son suavizados CORTOS; le faltaba
                             la deriva de meses.

    prev_dem_var24h          cuanto ha cambiado la prevision respecto a ayer.
    prev_eol_var24h          El modelo tenia el precio de ayer y el viento de
    prev_sol_var24h          HOY, pero no el de ayer: no podia calcular el
                             cambio. Es la mas fuerte de las diez.

**Por que funcionan, y es el mismo argumento en los tres bloques.** Un GBDT
necesita muchisimos cortes para aproximar una suma-resta de nueve variables
continuas, o una diferencia entre dos columnas. Dandoselas hechas se le ahorra
un trabajo que hacia mal. No es informacion nueva: es estructura.

**Causalidad.** Todo sale de precios y previsiones hasta D-1 inclusive, que es
lo que el modelo ya usa (`price_lag_24h` existe). Las de Francia salen de
`entsoe_load_forecast` y `entsoe_generation_forecast`, **las mismas dos tablas
que el modelo frances de este repositorio usa para su propia prediccion del dia
D**, asi que heredan su estatus causal y no anaden riesgo.

**Comprobacion de fuga, porque una ganancia asi obliga a sospechar.** El riesgo
era que `forecast_eolica_mw` se sobrescribiera con revisiones publicadas durante
D-1, posteriores al cierre de la subasta. Descartado: REE separa la prevision
DIARIA (indicadores 1775/1777/1779) de la INTRADIARIA (1776/1778/1780), y medido
sobre 6.113 horas de 2026 coinciden solo en el 0,2% de las horas y la diaria es
claramente PEOR contra el real (MAE 1.393 MW frente a 1.201) — exactamente lo
que debe pasar con una prevision D-1 genuina.

Uso:
    python -m studies.precio_da_mejor_modelo.features_v3
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from etl.common import logging_config  # noqa: F401

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
DATA_DIR = ROOT / "data"
TZ = "Europe/Madrid"

# Las dos que ya existian en el dataset y no estaban conectadas.
YA_EXISTEN = ["reserve_margin_mw", "ratio_renovable_periodo"]
# Las ocho que construye este modulo.
NUEVAS = ["tension_fr", "ltsc_90d", "ltsc_365d", "ltsc_pendiente", "desvio_ltsc",
          "prev_dem_var24h", "prev_eol_var24h", "prev_sol_var24h"]
TODAS = YA_EXISTEN + NUEVAS


def _local(s: pd.Series) -> pd.Series:
    """UTC -> dia de mercado local. El dia de mercado es la fecha LOCAL, no la
    UTC: calcularlo mal fue una fuga real de 0,80 EUR/MWh en su momento."""
    return s.dt.tz_localize("UTC").dt.tz_convert(TZ).dt.tz_localize(None)


def anadir(df: pd.DataFrame, log=print) -> tuple[pd.DataFrame, list[str]]:
    """Anade las ocho variables calculadas al dataset de Espana.

    Se llama desde `build_dataset.py` al final, y tambien desde la prediccion
    diaria, para que el camino en vivo y el walk-forward construyan EXACTAMENTE
    lo mismo.
    """
    df = df.sort_values("period_start_utc").reset_index(drop=True)
    # IDEMPOTENTE a proposito: la prediccion en vivo carga el historico ya
    # construido —que YA trae estas columnas— y le anade la fila de manana, asi
    # que hay que RECALCULARLAS sobre el conjunto extendido, no fallar ni
    # conservar las viejas. Sin esto, el join de abajo choca.
    df = df.drop(columns=[c for c in NUEVAS if c in df.columns])
    loc = _local(df["period_start_utc"])
    dia_local = loc.dt.normalize()

    # ---- 0. las dos que ya existian, por si el camino que llama aqui no las
    # construye. `build_dataset.build()` si lo hace; la prediccion en vivo
    # monta las features a mano y puede no tenerlas. Mismas formulas que
    # `_add_reserve_margin` y la linea de `ratio_renovable_periodo`, para que
    # el camino en vivo y el walk-forward construyan EXACTAMENTE lo mismo.
    if "ratio_renovable_periodo" not in df.columns:
        df["ratio_renovable_periodo"] = ((df["forecast_eolica_mw"] + df["forecast_solar_mw"])
                                         / df["forecast_demanda_mw"])
    if "reserve_margin_mw" not in df.columns:
        from studies.precio_da_mejor_modelo.build_dataset import HYDRO_TURBINE_CAP_MW
        df["reserve_margin_mw"] = (
            df["nuclear_capacity_avail_mw"] + df["gas_capacity_avail_mw"]
            + df["coal_capacity_avail_mw"] + HYDRO_TURBINE_CAP_MW
            + df["forecast_eolica_mw"] + df["forecast_solar_mw"]
            + df["cogeneracion_resto_clim"] + df["termica_renovable_clim"]
            + df["solar_termica_clim"] - df["forecast_demanda_mw"]
        )

    # ---- 1. componente estacional de largo plazo -------------------------
    # El SCAR canonico lo estima con filtro Hodrick-Prescott o wavelets sobre
    # toda la muestra, pero eso mira al futuro. Esta es la version D-1-segura:
    # suavizados que solo usan pasado, con el desplazamiento de un dia que
    # garantiza que el dia que se predice no entra en su propia tendencia.
    serie = df.groupby(dia_local)["price_real"].mean().sort_index()
    prev = serie.shift(1)
    ltsc = pd.DataFrame({"_dia_ltsc": serie.index})
    ltsc["ltsc_90d"] = prev.ewm(span=90, min_periods=20).mean().to_numpy()
    ltsc["ltsc_365d"] = prev.ewm(span=365, min_periods=20).mean().to_numpy()
    ltsc["ltsc_pendiente"] = ltsc["ltsc_90d"] - ltsc["ltsc_365d"]
    ltsc["desvio_ltsc"] = prev.ewm(span=7, min_periods=3).mean().to_numpy() - ltsc["ltsc_365d"]
    # merge explicito sobre una columna temporal, NO `join(on=Series)`: esa
    # forma inyecta una columna `key_0` en el dataset resultante.
    df["_dia_ltsc"] = dia_local.to_numpy()
    df = df.merge(ltsc, on="_dia_ltsc", how="left").drop(columns="_dia_ltsc")

    # ---- 2. variacion de las previsiones respecto a ayer -----------------
    # El paso se define en MINUTOS, no en filas: el dataset es horario antes
    # del cambio a cuartos y cuartohorario despues, asi que `shift(24)` seria
    # 24 h en un tramo y 6 h en el otro.
    res = df["resolution_minutes"] if "resolution_minutes" in df.columns else None
    for col, corto in (("forecast_demanda_mw", "dem"), ("forecast_eolica_mw", "eol"),
                       ("forecast_solar_mw", "sol")):
        c = f"prev_{corto}_var24h"
        df[c] = np.nan
        if res is None:
            df[c] = df[col] - df[col].shift(24)
        else:
            for r in sorted(res.dropna().unique()):
                m = res == r
                df.loc[m, c] = df.loc[m, col] - df.loc[m, col].shift(int(24 * 60 / r))

    # ---- 3. tension francesa ---------------------------------------------
    con = duckdb.connect(str(DATA_DIR / "france.duckdb"), read_only=True)
    try:
        carga = con.execute("SELECT interval_start_utc AS t, load_forecast_mw AS carga "
                            "FROM entsoe_load_forecast").fetchdf()
        gen = con.execute("SELECT interval_start_utc AS t, psr_type, "
                          "generation_forecast_mw AS mw FROM entsoe_generation_forecast").fetchdf()
    finally:
        con.close()
    ren = gen.pivot_table(index="t", columns="psr_type", values="mw", aggfunc="sum")
    ren = ren.reindex(columns=["B16", "B18", "B19"]).fillna(0).sum(axis=1).rename("ren")
    fr = carga.merge(ren.reset_index(), on="t", how="inner")
    fr["res_fr"] = fr["carga"] - fr["ren"]
    # La serie francesa y la espanola no comparten resolucion en todo el
    # historico: se unen por la hora en punto, que es el minimo comun seguro.
    fr["_h"] = fr["t"].dt.floor("h")
    fr_h = fr.groupby("_h", as_index=False)["res_fr"].mean()
    df["_h"] = df["period_start_utc"].dt.floor("h")
    df = df.merge(fr_h, on="_h", how="left").drop(columns="_h")
    if "nuclear_capacity_fr_avail_mw" in df.columns:
        df["tension_fr"] = df["res_fr"] - df["nuclear_capacity_fr_avail_mw"]
    else:
        df["tension_fr"] = np.nan
    df = df.drop(columns="res_fr")

    ev = df[df["period_start_utc"] >= "2025-04-01"]
    log(f"  {len(NUEVAS)} variables nuevas · cobertura en la ventana evaluable "
        f"{ev[NUEVAS].notna().all(axis=1).mean():.1%}")
    for c in TODAS:
        if c in df.columns:
            log(f"    {c:26s} nulos {ev[c].isna().mean():6.2%}  "
                f"corr con el precio {ev[c].corr(ev['price_real']):+.3f}")
    return df, NUEVAS


if __name__ == "__main__":
    ruta = OUTPUT_DIR / "dataset.parquet"
    d = pd.read_parquet(ruta)
    print(f"{ruta.name}: {len(d):,} filas")
    faltan = [c for c in YA_EXISTEN if c not in d.columns]
    if faltan:
        raise SystemExit(f"el dataset no trae {faltan} — reconstruyelo antes")
    d, nuevas = anadir(d)
    d.to_parquet(ruta, index=False)
    print(f"Guardado {ruta.name} con las {len(TODAS)} variables conectadas")
