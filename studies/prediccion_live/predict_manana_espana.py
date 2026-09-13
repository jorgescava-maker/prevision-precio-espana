"""
Primera versión de "predicción en vivo" para España — pedida por el usuario
(2026-09-01): obtener la previsión de precio DA de MAÑANA con el modelo de
producción, para comparar después contra la realidad (una vez OMIE publique
el precio real de ese día).

Hallazgo real encontrado al construir esto (documentado también en
docs/status.md): `coal_eur_mwh_th` (API2 vía Yahoo Finance, sin actualizar
desde 2025-12-26 — ver findings.md) está en `FEATURES_FULL` de
`precio_da_mejor_modelo/model.py`, y el walk-forward de producción usa
`dropna` ESTRICTO sobre todas las features — como consecuencia, el modelo de
producción NUNCA se ha evaluado (ni se puede entrenar de forma completa)
sobre NINGÚN dato de 2026 (8 meses completos, incluido el episodio de calor
jul-ago 2026 que sí motivó un fix en `bandera_extremo.py`, pero no en el
modelo principal). Para esta predicción en vivo se usa una variante sin esa
columna (`FEATURES_LIVE`), igual que `bandera_extremo.py` ya hizo por el
mismo motivo — así el modelo de producción final SÍ se entrena con
2023-2026 completo, no solo hasta enero de 2026.

ACTUALIZADO 2026-09-02 — hasta esa fecha este script usaba una arquitectura
MÁS VIEJA que la de producción: modelo entrenado solo con España y mezcla de
dos miembros. Ahora reproduce la arquitectura completa del campeón, que son
cinco capas (ver `precio_da_conjunto/RESULTS.md`):

    1. modelo agrupado entrenado con España Y Francia a la vez
    2. modelos por hora del día, mezclados con el anterior
    3. ensemble de 10 redes neuronales como tercer miembro
    4. filtro de precio<=0 (gate de hurdle)
    5. calibración de la cola alta (corrige el encogimiento hacia el centro)

Los pesos de la mezcla y la recta de calibración NO se reestiman aquí: se
toman de la última decisión causal del walk-forward de producción, que es
justo lo que estaría vigente hoy. Reestimarlos con el histórico completo sería
mirar datos que en la operación real no se tendrían.

Enfoque: en vez de tocar el pipeline batch (`build_dataset.py`, pensado para
reconstruir todo el histórico contra `omie_spot_prices`, que todavía no
tiene el precio de mañana), este script reconstruye la MISMA receta de
features pero con el hueco de "mañana" añadido a mano — el resto de fuentes
(previsión D-1 de e·sios, disponibilidad real, climatologías, análogo,
lags/EMA de precio, interconexiones) ya son D-1-seguras por diseño y
soportan añadir una fila futura sin fuga (ver docstrings de
merit_order/build_dataset.py y precio_da_mejor_modelo/build_dataset.py).

Uso:
    .venv\\Scripts\\python.exe -m studies.prediccion_live.predict_manana_espana [YYYY-MM-DD]

Sin argumento, usa la fecha de mañana (hora local del sistema).
"""

from __future__ import annotations

import sys
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd

from etl.common import logging_config  # noqa: F401

from studies.merit_order.build_dataset import (
    DATA_DIR, CLIMATOLOGY_CATEGORIES, GAS_HEAT_RATE_RANGE, COAL_HEAT_RATE_RANGE,
    HEAT_RATE_GAS, EMISSION_FACTOR_GAS, COAL_MWH_PER_TONNE, HEAT_RATE_COAL, EMISSION_FACTOR_COAL,
    _load_forecast_d1, _load_realized_hourly, _load_net_import, _load_commodities,
    _load_generation_units, _add_climatology, _build_unit_tiers, _dispatch_row,
)
from studies.precio_da_mejor_modelo.build_dataset import (
    _add_real_availability, _add_holidays, _build_analog_feature_v2,
    _build_analog_feature_v3, _build_analog_feature_v4, _build_analog_feature_v5,
    _add_price_lags, _add_interconnection_features,
)
from studies.merit_order.model import build_water_value_feature
from studies.precio_da_mejor_modelo.model import FEATURES_FULL as _FEATURES_FULL, LGB_PARAMS
from studies.precio_da_mejor_modelo.hurdle_quantile import ZERO_THRESHOLD, best_threshold_f1
from studies.precio_da_conjunto.model import COMMON_FEATURES
from studies.precio_da_conjunto.model_full import SPAIN_ONLY, FRANCE_ONLY, load_france_full
from studies.precio_da_conjunto.redes_espana import _fit_predict_ensemble
from studies.precio_da_conjunto.ensemble_tres_miembros import UMBRAL_CALIBRACION
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(__file__).resolve().parent / "output"

# Ver docstring del módulo — misma exclusión que `bandera_extremo.py`, por el
# mismo motivo (corte real de la fuente, no un hueco de carga).
FEATURES_LIVE = [f for f in _FEATURES_FULL if f != "coal_eur_mwh_th"] + ["price_simulado_final"]

TZ_MADRID = ZoneInfo("Europe/Madrid")


def dia_local_en_utc(dia: date) -> tuple[datetime, datetime]:
    """Límites [inicio, fin) en UTC sin zona del día local de entrega `dia`."""
    ini = datetime.combine(dia, time(0), tzinfo=TZ_MADRID).astimezone(timezone.utc).replace(tzinfo=None)
    fin = datetime.combine(dia + timedelta(days=1), time(0), tzinfo=TZ_MADRID).astimezone(timezone.utc).replace(tzinfo=None)
    return ini, fin


def _build_merit_order_extended(target_hours: pd.DatetimeIndex, log) -> pd.DataFrame:
    """Reconstruye el motor de despacho de `merit_order` extendido con las
    horas de `target_hours` (mañana), sin exigir que `omie_spot_prices` ya
    tenga esas horas (el build() original sí lo exige vía un INNER JOIN)."""
    con = duckdb.connect(str(DATA_DIR / "spain.duckdb"), read_only=True)
    try:
        forecast = _load_forecast_d1(con)  # ya incluye "mañana" (fetch en vivo antes de correr esto)
        realized = _load_realized_hourly(con)
        units = _load_generation_units(con)
    finally:
        con.close()

    con = duckdb.connect(str(DATA_DIR / "interconnections.duckdb"), read_only=True)
    try:
        net_import = _load_net_import(con)
    finally:
        con.close()

    con = duckdb.connect(str(DATA_DIR / "commodities.duckdb"), read_only=True)
    try:
        commodities = _load_commodities(con)
    finally:
        con.close()

    missing = [h for h in target_hours if h not in set(forecast["hour_utc"])]
    if missing:
        raise SystemExit(f"Previsión D-1 de e·sios incompleta para mañana, faltan horas: {missing}")

    realized = realized.merge(net_import, on="hour_utc", how="left")
    # Filas placeholder de "mañana" para que la climatología (shift(1) antes de
    # rolling) calcule un valor en esa posición usando SOLO semanas anteriores
    # — nunca el propio valor real de mañana, que no existe.
    placeholder = pd.DataFrame({"hour_utc": target_hours})
    for col in CLIMATOLOGY_CATEGORIES + ["net_import_mw"]:
        placeholder[col] = np.nan
    realized_ext = pd.concat([realized, placeholder], ignore_index=True)
    realized_ext = _add_climatology(realized_ext, CLIMATOLOGY_CATEGORIES + ["net_import_mw"])

    # LEFT (no INNER como en build() original) para conservar las horas de
    # mañana, que `price` (OMIE) todavía no tiene.
    df = forecast.merge(realized_ext[["hour_utc"] + [f"{c}_clim" for c in CLIMATOLOGY_CATEGORIES] + ["net_import_mw_clim"]],
                         on="hour_utc", how="left")
    df = df[df["hour_utc"].isin(set(realized_ext["hour_utc"]) | set(forecast["hour_utc"]))]

    df["trade_date_cutoff"] = (df["hour_utc"] - pd.Timedelta(days=2)).dt.normalize()
    df = df.sort_values("trade_date_cutoff")
    commodities = commodities.sort_values("trade_date")
    df = pd.merge_asof(df, commodities, left_on="trade_date_cutoff", right_on="trade_date", direction="backward")

    df["coal_eur_mwh_th"] = (df["coal_usd_t"] / df["eur_usd"]) / COAL_MWH_PER_TONNE
    df["capacity_year"] = df["hour_utc"].dt.year
    gas_tiers_by_year = _build_unit_tiers(units, "B04", GAS_HEAT_RATE_RANGE)
    coal_tiers_by_year = _build_unit_tiers(units, "B05", COAL_HEAT_RATE_RANGE)

    df["must_run_mw"] = (
        df["forecast_eolica_mw"].fillna(0) + df["forecast_solar_mw"].fillna(0)
        + df["nuclear_clim"].fillna(0) + df["hidraulica_clim"].fillna(0)
        + df["solar_termica_clim"].fillna(0) + df["termica_renovable_clim"].fillna(0)
    )
    df["demand_to_cover_mw"] = df["forecast_demanda_mw"] - df["net_import_mw_clim"].fillna(0)

    results = df.apply(
        lambda r: _dispatch_row(
            r["demand_to_cover_mw"], r["must_run_mw"], r["cogeneracion_resto_clim"],
            r["capacity_year"], r["ttf_eur_mwh"], r["eua_eur_t"], r["coal_eur_mwh_th"],
            gas_tiers_by_year, coal_tiers_by_year,
        ), axis=1, result_type="expand",
    )
    df["price_simulado"], df["tecnologia_marginal"] = results[0], results[1]
    df["must_run_final_mw"] = df["must_run_mw"]
    df = df.rename(columns={"price_simulado": "price_simulado_final", "tecnologia_marginal": "tecnologia_marginal_final"})

    # Hallazgo real 2026-09-01 (pedido explícito del usuario, "esto no tiene
    # sentido"): esta función original NO calculaba `water_value_eur_mwh` —
    # como analog_price_mean_v3/v4/v5 (v5 es la feature MÁS importante de
    # todo el modelo, ver MODELO_FINAL_AUDITORIA.md §3) usan esa columna como
    # dimensión de emparejamiento y `_build_analog_generic` salta la fila por
    # completo si cualquier dimensión propia es NaN, el análogo quedaba NaN
    # para TODAS las filas de mañana — silenciosamente apagando la feature
    # dominante del modelo justo para la predicción en vivo. Corregido
    # reutilizando `build_water_value_feature` de `merit_order.model` tal
    # cual (misma DP de C3, sin recalcular nada).
    log("  Calculando el valor del agua (C3, reutilizado) para históricos+mañana...")
    df["water_value_eur_mwh"], _ = build_water_value_feature(df, log)

    # Simplificación declarada, NO corregida hoy (SHAP de price_simulado_final
    # en solitario es bajo, 1.02, frente al 15.80 del análogo — impacto menor
    # pero real): este motor extendido reproduce el despacho SIMPLE original
    # de merit_order/build_dataset.py, no la versión Fase 1 de producción
    # (hidráulica como tramo de coste vía water_value + ajuste por
    # indisponibilidad real, corr 0,522 en solitario) que sí usa el dataset
    # histórico batch. Pendiente para una futura sesión si se quiere el motor
    # exacto también en la predicción en vivo.
    keep = [
        "hour_utc", "forecast_demanda_mw", "forecast_eolica_mw", "forecast_solar_mw",
        "ttf_eur_mwh", "eua_eur_t", "coal_eur_mwh_th", "capacity_year",
        "cogeneracion_resto_clim", "solar_termica_clim", "termica_renovable_clim", "net_import_mw_clim",
        "demand_to_cover_mw", "must_run_final_mw", "price_simulado_final", "tecnologia_marginal_final",
        "water_value_eur_mwh",
    ]
    log(f"  Motor de despacho extendido a mañana: {df['hour_utc'].max()} (target incluido: "
        f"{target_hours.isin(df['hour_utc']).all()})")
    return df[keep]


def build_live_dataset(target_date: date, log) -> pd.DataFrame:
    """Histórico completo (recién reconstruido, incluye hoy) + las filas de
    `target_date` (mañana), con exactamente las mismas columnas/orden de
    transformaciones que `precio_da_mejor_modelo/build_dataset.py`."""
    hist = pd.read_parquet(ROOT / "studies" / "precio_da_mejor_modelo" / "output" / "dataset.parquet")
    log(f"Histórico ya construido: {len(hist):,} periodos hasta {hist['period_start_utc'].max()}")

    # El día que casa OMIE es el día LOCAL de entrega (Europe/Madrid): en verano
    # va de las 22:00 UTC de la víspera a las 21:45 UTC, en invierno de las
    # 23:00 a las 22:45. Cortar por fecha UTC dejaba fuera las dos primeras
    # horas locales del día (00:00-02:00) y predecía 88 periodos en vez de 96.
    ini_utc, fin_utc = dia_local_en_utc(target_date)
    con = duckdb.connect(str(DATA_DIR / "spain.duckdb"), read_only=True)
    try:
        forecast_target = con.execute(
            "SELECT DISTINCT interval_start_utc AS hour_utc FROM esios_forecast "
            "WHERE interval_start_utc >= ? AND interval_start_utc < ? ORDER BY 1",
            [ini_utc, fin_utc],
        ).fetchdf()
    finally:
        con.close()
    if forecast_target.empty:
        raise SystemExit(
            f"e·sios todavía no publica la previsión D-1 para {target_date} — "
            f"reintentar más tarde (ver etl/sources/esios_forecast.py, run_forecast_for_range)."
        )
    target_hours = pd.to_datetime(forecast_target["hour_utc"])
    log(f"Previsión D-1 disponible para mañana: {len(target_hours)} horas ({target_hours.min()} -> {target_hours.max()})")

    merit_hist_and_target = _build_merit_order_extended(target_hours, log)
    target_hourly = merit_hist_and_target[merit_hist_and_target["hour_utc"].isin(target_hours)].copy()

    log("Disponibilidad real térmica/nuclear (asume sin nueva indisponibilidad anunciada tras la última conocida)...")
    target_hourly = _add_real_availability(target_hourly, log)

    # Resolución nativa de mañana: la misma que hoy (15 min, reforma ya
    # vigente desde 2025-10-01) — se generan 4 cuartos sintéticos por hora,
    # price_real/price_pt = NaN (el objetivo de mañana, por construcción, se
    # desconoce).
    quarters = []
    for h in target_hours:
        for q in range(4):
            quarters.append({"period_start_utc": h + pd.Timedelta(minutes=15 * q), "hour_utc": h,
                              "resolution_minutes": 15, "period_in_hour": q, "price_real": np.nan, "price_pt": np.nan})
    native_target = pd.DataFrame(quarters)
    df_target = native_target.merge(target_hourly, on="hour_utc", how="left")
    df_target["hour_of_day"] = df_target["period_start_utc"].dt.hour
    df_target["day_of_week"] = df_target["period_start_utc"].dt.dayofweek
    df_target["month"] = df_target["period_start_utc"].dt.month
    df_target["period_of_day"] = df_target["hour_of_day"] * 4 + df_target["period_in_hour"]

    combined = pd.concat([hist, df_target], ignore_index=True, sort=False)
    # Si se pide un día que YA está en el histórico (útil para validar el
    # pipeline contra un día conocido), las filas de ese día quedarían
    # duplicadas y `_add_price_lags` fallaría al indexar por fecha. Se queda la
    # fila reconstruida en vivo, que es la que el modelo va a predecir.
    combined = (combined.sort_values("period_start_utc")
                        .drop_duplicates(subset=["period_start_utc"], keep="last")
                        .reset_index(drop=True))

    log("Recalculando festivos/fin de semana sobre el conjunto extendido...")
    combined = _add_holidays(combined)

    log("Recalculando análogos v2-v5 sobre el conjunto extendido (mañana solo mira hacia atrás)...")
    combined = _build_analog_feature_v2(combined, log)
    combined = _build_analog_feature_v3(combined, log)
    combined = _build_analog_feature_v4(combined, log)
    combined = _build_analog_feature_v5(combined, log)

    log("Recalculando lags/EMA/boundary de precio (por timestamp exacto)...")
    combined = _add_price_lags(combined, log)

    log("Recalculando interconexiones ES-FR/ES-PT...")
    combined = _add_interconnection_features(combined, log)

    # Las diez de la revisión de 2026-09-13, con el MISMO módulo que usa
    # `build_dataset.build()`: el camino en vivo y el walk-forward tienen que
    # construir exactamente lo mismo, o lo que se sirve no es lo que se validó.
    from studies.precio_da_mejor_modelo.features_v3 import anadir as _anadir_v3
    combined, _ = _anadir_v3(combined, log)

    return combined


SEMILLAS_REDES = list(range(42, 47))   # las mismas que usa el walk-forward


def _pesos_produccion(log) -> tuple[float, float, float]:
    """Último peso elegido causalmente por el walk-forward de producción."""
    # Del backtest con el valor del agua causal (agua_causal.py, findings.md
    # #146): el de ensemble_tres_miembros.py usa una superficie del agua
    # estimada con toda la historia y sus pesos arrastrarían esa fuga.
    ruta = ROOT / "studies" / "precio_da_conjunto" / "output" / "agua_causal_pesos.csv"
    pesos = pd.read_csv(ruta)
    fila = pesos.dropna(subset=["w_pooled", "w_hora", "w_red"]).iloc[-1]
    log(f"  pesos de la mezcla (último mes causal, {fila['mes']}): agrupado={fila['w_pooled']:.1f} · "
        f"por-hora={fila['w_hora']:.1f} · redes={fila['w_red']:.1f}")
    return float(fila["w_pooled"]), float(fila["w_hora"]), float(fila["w_red"])


def _calibracion_produccion(log):
    """Recta a la mediana y umbral, ajustados sobre las predicciones
    walk-forward ya evaluadas (no sobre el entrenamiento): es la misma
    información que tendría la operación real."""
    ruta = ROOT / "studies" / "precio_da_conjunto" / "output" / "predicciones_agua_causal.parquet"
    h = pd.read_parquet(ruta).dropna(subset=["pred_ensemble3", "price_real"])
    umbral = float(h["pred_ensemble3"].quantile(UMBRAL_CALIBRACION))
    ajuste = sm.QuantReg(h["price_real"].to_numpy(float),
                         sm.add_constant(h["pred_ensemble3"].to_numpy(float))).fit(q=0.5)
    a, b = float(ajuste.params[0]), float(ajuste.params[1])
    log(f"  calibración de cola: y = {a:+.2f} + {b:.3f}·x  aplicada por encima de "
        f"{umbral:.1f} EUR/MWh (P{int(UMBRAL_CALIBRACION * 100)} del histórico evaluado)")
    return a, b, umbral


def main() -> None:
    lines: list[str] = []

    def log(s: str = "") -> None:
        print(s)
        lines.append(s)

    target_date = date.fromisoformat(sys.argv[1]) if len(sys.argv) > 1 else date.today() + timedelta(days=1)
    log("=" * 72)
    log(f"Predicción en vivo — España, precio DA de {target_date}")
    log("Arquitectura de producción: pooled ES+FR + por-hora + redes + gate + calibración")
    log("=" * 72)

    combined = build_live_dataset(target_date, log)
    combined["is_holiday"] = combined["is_holiday"].astype(int)
    combined["is_weekend"] = combined["is_weekend"].astype(int)
    # El modelo conjunto llama `analog_price_mean` a lo que aquí se construye como
    # `analog_price_mean_v5`; se deja con los dos nombres porque el clasificador
    # del gate sigue usando la lista FEATURES_LIVE, con el nombre original.
    combined["analog_price_mean"] = combined["analog_price_mean_v5"]
    combined["country"] = "ES"

    faltan = [f for f in COMMON_FEATURES + SPAIN_ONLY if f not in combined.columns]
    if faltan:
        raise SystemExit(f"Faltan columnas en el dataset en vivo: {faltan}")

    ini_utc, fin_utc = dia_local_en_utc(target_date)
    is_target = (combined["period_start_utc"] >= ini_utc) & (combined["period_start_utc"] < fin_utc)
    es_train = combined.loc[~is_target].copy()
    target = combined.loc[is_target].copy()

    fr = load_france_full()
    pooled = pd.concat([es_train, fr], ignore_index=True)
    pooled["is_spain"] = (pooled["country"] == "ES").astype(int)
    target["is_spain"] = 1
    todas = list(dict.fromkeys(COMMON_FEATURES + SPAIN_ONLY + FRANCE_ONLY + ["is_spain"]))
    # Las columnas propias de Francia no existen en el marco español: se crean
    # vacías, que es exactamente lo que ve el modelo en las filas de España
    # durante el entrenamiento (LightGBM las maneja de forma nativa).
    for c in todas:
        if c not in target.columns:
            target[c] = np.nan

    # Mismo criterio que el walk-forward: solo se exige que estén las features
    # COMUNES; las propias de cada país pueden faltar y LightGBM lo maneja.
    entren = pooled.dropna(subset=COMMON_FEATURES + ["price_real"]).copy()
    log(f"\nEntrenamiento: {len(entren):,} periodos "
        f"({int((entren['country'] == 'ES').sum()):,} España + {int((entren['country'] == 'FR').sum()):,} Francia)")
    log(f"Periodos a predecir: {len(target)} · variables del modelo: {len(todas)}")

    log("\n-- (1) Modelo agrupado ES+FR --")
    reg = lgb.LGBMRegressor(**LGB_PARAMS)
    reg.fit(entren[todas], entren["price_real"])
    target["pred_pooled"] = reg.predict(target[todas])

    log("-- (2) Modelos por hora del día --")
    feats_hora = [f for f in todas if f not in ("hour_of_day", "period_of_day")]
    pred_h = np.full(len(target), np.nan)
    for h, g in entren.groupby("hour_of_day"):
        if len(g) < 30:
            continue
        reg_h = lgb.LGBMRegressor(**LGB_PARAMS)
        reg_h.fit(g[feats_hora], g["price_real"])
        m = (target["hour_of_day"] == h).to_numpy()
        if m.any():
            pred_h[m] = reg_h.predict(target.loc[m, feats_hora])
    target["pred_por_hora"] = pred_h

    log("-- (3) Ensemble de redes (10 redes: 5 pooled ES+FR + 5 solo España) --")
    y = entren["price_real"].to_numpy(dtype=float)
    red_pooled = _fit_predict_ensemble(entren[todas], y, target[todas], SEMILLAS_REDES, n_jobs=5)
    solo_es = entren[entren["country"] == "ES"]
    feats_es = [f for f in todas if f not in FRANCE_ONLY + ["is_spain"]]
    red_es = _fit_predict_ensemble(solo_es[feats_es], solo_es["price_real"].to_numpy(dtype=float),
                                   target[feats_es], SEMILLAS_REDES, n_jobs=5)
    target["pred_redes"] = (red_pooled + red_es) / 2

    log("\n-- Mezcla de los tres miembros --")
    w_p, w_h, w_r = _pesos_produccion(log)
    target["pred_mezcla"] = (w_p * target["pred_pooled"] + w_h * target["pred_por_hora"]
                             + w_r * target["pred_redes"])

    log("\n-- (5) Calibración de la cola alta --")
    a, b, umbral = _calibracion_produccion(log)
    crudo = target["pred_mezcla"].to_numpy(float)
    target["pred_calibrada"] = np.where(crudo > umbral, a + b * crudo, crudo)
    n_cal = int((crudo > umbral).sum())
    log(f"  periodos corregidos: {n_cal} de {len(target)}")

    log("\n-- (4) Gate de hurdle (precio<=0) --")
    hurdle_feats = [f for f in FEATURES_LIVE]
    train_h = es_train.dropna(subset=hurdle_feats + ["price_real"]).copy()
    train_h["is_zero"] = (train_h["price_real"] <= ZERO_THRESHOLD).astype(int)
    clf = lgb.LGBMClassifier(**LGB_PARAMS)
    clf.fit(train_h[hurdle_feats], train_h["is_zero"])
    proba_train = clf.predict_proba(train_h[hurdle_feats])[:, 1]
    thr = best_threshold_f1(proba_train, train_h["is_zero"].to_numpy())
    zero_fill = train_h.loc[train_h["is_zero"] == 1, "price_real"].mean()
    proba_target = clf.predict_proba(target[hurdle_feats].fillna(train_h[hurdle_feats].median()))[:, 1]
    target["proba_precio_cero"] = proba_target
    target["pred_zero"] = proba_target >= thr
    target["pred_final"] = np.where(target["pred_zero"], zero_fill, target["pred_calibrada"])
    log(f"  umbral={thr:.2f}  relleno={zero_fill:.2f}  periodos marcados 'precio<=0': "
        f"{int(target['pred_zero'].sum())}")

    log(f"\n{'='*72}\nPREDICCIÓN FINAL — {target_date}\n{'='*72}")
    for _, r in target.sort_values("period_start_utc").iterrows():
        marca = " [hurdle->0]" if r["pred_zero"] else (" [calibrada]" if r["pred_mezcla"] > umbral else "")
        log(f"  {r['period_start_utc']}  pred={r['pred_final']:7.2f} EUR/MWh"
            f"  (agrupado={r['pred_pooled']:.1f}, por_hora={r['pred_por_hora']:.1f}, "
            f"redes={r['pred_redes']:.1f}){marca}")

    media = float(target["pred_final"].mean())
    log(f"\n  Media del día: {media:.2f} EUR/MWh · mínimo {target['pred_final'].min():.2f} · "
        f"máximo {target['pred_final'].max():.2f}")

    OUTPUT_DIR.mkdir(exist_ok=True)
    out_cols = ["period_start_utc", "hour_of_day", "period_in_hour", "pred_pooled", "pred_por_hora",
                "pred_redes", "pred_mezcla", "pred_calibrada", "proba_precio_cero", "pred_zero",
                "pred_final"]
    out_path = OUTPUT_DIR / f"prediccion_espana_{target_date}.parquet"
    target[out_cols].to_parquet(out_path, index=False)
    (OUTPUT_DIR / f"prediccion_espana_{target_date}_log.txt").write_text("\n".join(lines), encoding="utf-8")
    log(f"\nGuardado: {out_path}")
    log("Para comparar contra la realidad: cuando OMIE publique el precio real de "
        f"{target_date} en `omie_spot_prices`, unir por `period_start_utc` contra este parquet.")


if __name__ == "__main__":
    main()
