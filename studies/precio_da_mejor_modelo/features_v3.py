"""Las diez variables de la revision de 2026-09-13, mas margen_neto (2026-09-14),
sobre el dataset de Espana.

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

    margen_neto              reserve_margin_mw menos tension_fr: la holgura
                             propia del sistema espanol, restando cuanto va
                             a tirar Francia por la interconexion. Un GBDT
                             no aproxima bien una diagonal con splits por
                             eje; dandosela ya restada se ahorra ese trabajo
                             (mismo argumento que reserve_margin_mw). Cero
                             dato nuevo: resta de dos columnas que ya estaban
                             las dos en SPAIN_ONLY.

**Meteorologia MULTIPUNTO (2026-09-15), fuente nueva.** El unico dato de esta
revision que SI es nuevo: 26 variables de viento/radiacion/temperatura en 10
puntos de Espana elegidos por donde esta el recurso eolico/solar (A Coruna,
Burgos-Soria, Zaragoza, Navarra, Tarifa, Albacete · Badajoz, Sevilla-Cordoba,
Ciudad Real, Murcia), no en un punto unico. `weather.duckdb` en este
repositorio solo trae un punto para Espana (Madrid), y el viento en Madrid no
dice nada del que ven las turbinas en Galicia o Aragon. Se consulta en vivo la
API gratuita de Open-Meteo (`previous-runs`, `lead=2`, sin clave), la misma
receta que ya funcionaba en el modelo privado equivalente. Si se le dan al
GBDT las variables de los 10 puntos, aprende el solo cuales pesan — mas barato
que construir una ponderacion por capacidad instalada que no tenemos.

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

import time
from datetime import date, timedelta
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import requests

from etl.common import logging_config  # noqa: F401

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
DATA_DIR = ROOT / "data"
TZ = "Europe/Madrid"

# Las dos que ya existian en el dataset y no estaban conectadas.
YA_EXISTEN = ["reserve_margin_mw", "ratio_renovable_periodo"]
# Las diez que construye este modulo (las ocho del 2026-09-13 + margen_neto +
# meteorologia multipunto, que en realidad son 26 columnas).
METEO_MP_API = "https://previous-runs-api.open-meteo.com/v1/forecast"
METEO_MP_MODELO = "ecmwf_ifs025"
METEO_MP_LEAD = 2                    # el unico inequivocamente D-1 seguro
METEO_MP_DESDE = "2024-03-06"        # inicio real de la radiacion solar en la fuente
METEO_MP_PUNTOS = {
    "eo_coruna":   (43.37, -8.40),
    "eo_burgos":   (41.77, -2.47),
    "eo_zaragoza": (41.65, -0.89),
    "eo_navarra":  (42.46, -2.45),
    "eo_tarifa":   (36.14, -5.74),
    "eo_albacete": (38.99, -1.86),
    "so_badajoz":   (38.88, -6.97),
    "so_sevilla":   (37.60, -5.30),
    "so_ciudadreal": (38.99, -3.93),
    "so_murcia":    (38.00, -1.13),
}
METEO_MP_VARS = {"temperature_2m": "temp", "wind_speed_100m": "v100",
                  "shortwave_radiation": "rad", "direct_radiation": "raddir"}
NUEVAS = ["tension_fr", "ltsc_90d", "ltsc_365d", "ltsc_pendiente", "desvio_ltsc",
          "prev_dem_var24h", "prev_eol_var24h", "prev_sol_var24h", "margen_neto"]
METEO_MP_COLS = ([f"v100_{p}_cubo" for p in METEO_MP_PUNTOS if p.startswith("eo_")]
                  + [f"v100_{p}" for p in METEO_MP_PUNTOS if p.startswith("eo_")]
                  + [f"rad_{p}" for p in METEO_MP_PUNTOS if p.startswith("so_")]
                  + ["eo_cubo_medio", "eo_v100_medio", "eo_v100_disp", "eo_v100_max",
                     "eo_v100_min", "so_rad_media", "so_rad_disp", "so_directa_frac",
                     "temp_media", "temp_disp"])
NUEVAS = NUEVAS + METEO_MP_COLS
TODAS = YA_EXISTEN + NUEVAS


METEO_MP_CACHE = OUTPUT_DIR / "meteo_multipunto_cache.parquet"


def _meteo_multipunto(log=print) -> pd.DataFrame:
    """Descarga (Open-Meteo, sin clave) y construye las 26 columnas de los 10
    puntos, con cache en disco por DÍA. `anadir()` se llama muchas veces en
    la misma corrida (una por cada mes de corte del walk-forward causal, más
    la predicción en vivo) — sin cache, cada llamada repite la descarga del
    histórico entero (2024-03 -> ayer) y la API gratuita empieza a devolver
    429 (verificado en vivo: falló a los 18 de 31 meses). El cache se
    invalida solo cuando cambia el día (`hasta` avanza), así que dentro de
    una misma corrida siempre es la MISMA descarga para todos los cortes —
    coherente con que el walk-forward causal ya asume "info hasta ayer"."""
    hasta = (date.today() - timedelta(days=1)).isoformat()
    if METEO_MP_CACHE.exists():
        cache = pd.read_parquet(METEO_MP_CACHE)
        if len(cache) and str(cache["_hora"].max().date()) >= hasta:
            return cache
    suf = f"_previous_day{METEO_MP_LEAD}"
    hourly = ",".join(f"{v}{suf}" for v in METEO_MP_VARS)
    sesion = requests.Session()
    trozos = []
    for nombre, (lat, lon) in METEO_MP_PUNTOS.items():
        for intento in range(1, 4):
            try:
                r = sesion.get(METEO_MP_API, params={
                    "latitude": lat, "longitude": lon,
                    "start_date": METEO_MP_DESDE, "end_date": hasta,
                    "models": METEO_MP_MODELO, "hourly": hourly, "timezone": "UTC",
                }, timeout=120)
                r.raise_for_status()
                p = r.json()
                if p.get("error"):
                    raise ValueError(p.get("reason"))
                break
            except Exception as exc:
                if intento == 3:
                    raise
                log(f"    reintento {intento} en meteo {nombre}: {exc}")
                time.sleep(3 * intento)
        h = p["hourly"]
        d = pd.DataFrame({"_hora": pd.to_datetime(h["time"])})
        for v, corto in METEO_MP_VARS.items():
            d[f"{corto}_{nombre}"] = h[f"{v}{suf}"]
        trozos.append(d.set_index("_hora"))
        time.sleep(0.3)

    w = pd.concat(trozos, axis=1).reset_index()
    eo = [c for c in w.columns if c.startswith("v100_eo_")]
    so = [c for c in w.columns if c.startswith("rad_so_")]
    dirs = [c for c in w.columns if c.startswith("raddir_so_")]
    for c in eo:
        w[c + "_cubo"] = (w[c] / 100.0) ** 3
    w["eo_cubo_medio"] = w[[c + "_cubo" for c in eo]].mean(axis=1)
    w["eo_v100_medio"] = w[eo].mean(axis=1)
    w["eo_v100_disp"] = w[eo].std(axis=1)
    w["eo_v100_max"] = w[eo].max(axis=1)
    w["eo_v100_min"] = w[eo].min(axis=1)
    w["so_rad_media"] = w[so].mean(axis=1)
    w["so_rad_disp"] = w[so].std(axis=1)
    w["so_directa_frac"] = w[dirs].sum(axis=1) / (w[so].sum(axis=1) + 1.0)
    temps = [c for c in w.columns if c.startswith("temp_")]
    w["temp_media"] = w[temps].mean(axis=1)
    w["temp_disp"] = w[temps].std(axis=1)
    log(f"  meteorologia multipunto: {len(w):,} horas descargadas ({METEO_MP_DESDE} -> {hasta})")
    out = w[["_hora"] + METEO_MP_COLS]
    OUTPUT_DIR.mkdir(exist_ok=True)
    out.to_parquet(METEO_MP_CACHE, index=False)
    return out


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

    # ---- 4. margen_neto (2026-09-14) --------------------------------------
    # reserve_margin_mw - tension_fr: la holgura propia menos cuanto tira
    # Francia. Sin dato nuevo, resta de dos columnas que ya existian.
    df["margen_neto"] = df["reserve_margin_mw"] - df["tension_fr"]

    # ---- 5. meteorologia multipunto (2026-09-15) --------------------------
    # 10 puntos donde esta el recurso eolico/solar, no Madrid. Unico dato
    # NUEVO de todo este modulo (el resto son transformaciones de lo que ya
    # habia). Se une por la hora en punto, igual que la tension francesa.
    meteo = _meteo_multipunto(log)
    df["_h"] = df["period_start_utc"].dt.floor("h")
    df = df.merge(meteo.rename(columns={"_hora": "_h"}), on="_h", how="left").drop(columns="_h")

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
