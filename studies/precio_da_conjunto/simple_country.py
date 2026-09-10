"""
Constructor D-1-seguro compartido para paises "simples" (Alemania, Paises
Bajos) -- sin eventos reales de indisponibilidad nuclear/termica ni
embalse hidraulico cargados en el proyecto (a diferencia de Espana/
Francia), asi que el conjunto de features se reduce a lo que SI existe:
prevision D-1 de demanda/eolica/solar, materias primas (TTF/EUA/carbon,
D-2), el analogo (demanda+ratio renovable+ttf), y lags/EMA/boundary de
precio -- exactamente el nucleo COMMON_FEATURES ya usado en el modelo
conjunto ES+FR (ver precio_da_conjunto/model.py).

Reutiliza sin cambios `_build_analog_generic`, `_add_price_lags` (de
`precio_da_mejor_modelo.build_dataset`) y `_load_commodities` (de
`merit_order.build_dataset`) -- mismo patron que Francia.

Uso (desde el build_dataset.py de cada pais):
    from studies.precio_da_conjunto.simple_country import build_simple_country
    df = build_simple_country('germany.duckdb', 'smard_day_ahead_prices', 'germany', log)
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from studies.merit_order.build_dataset import DATA_DIR, _load_commodities
from studies.precio_da_mejor_modelo.build_dataset import _build_analog_generic, _add_price_lags
from studies.precio_da_mejor_modelo.model import walk_forward, walk_forward_by_group, report

# findings.md #103 (2026-09-01): `coal_eur_mwh_th` sin actualizar desde
# 2025-12-26 — excluida de producción (recupera 2026 completo).
# HDD/CDD (findings.md #106, 2026-09-01): probado en vivo (correlación
# univariada real +0,23 Alemania / +0,25 Países Bajos) pero NO adoptado —
# en el walk-forward completo queda plano en Alemania (15,24->15,29) y
# EMPEORA Países Bajos (17,31->17,51): la señal ya está mayormente cubierta
# por forecast_demanda_mw/mes. La columna se sigue calculando (`_add_degree_days`,
# queda en el dataset) por si sirve de dimensión de análogo en el futuro, solo
# no entra en COMMON_FEATURES. Sí ayudó en Francia (ver
# precio_da_francia/model.py) — no se generaliza automáticamente entre países.
COMMON_FEATURES = [
    "forecast_demanda_mw", "forecast_eolica_mw", "forecast_solar_mw",
    "ttf_eur_mwh", "eua_eur_t",
    "analog_price_mean",
    "price_lag_24h", "price_lag_48h", "price_lag_168h", "price_ema_7d", "price_ema_28d",
    "price_boundary_prev_day",
    "hour_of_day", "period_in_hour", "period_of_day", "day_of_week", "month",
    "is_holiday", "is_weekend",
]

WARMUP_MONTHS_ENSEMBLE = 6
PESOS_CANDIDATOS = np.arange(0.0, 1.01, 0.1)


def run_baseline_vs_ml_vs_ensemble(df: pd.DataFrame, log, features: list[str] | None = None) -> dict:
    """Pipeline compartido: (a) baseline naive price_lag_24h, (a') análogo
    solo, (b) ML flexible pooled, (d) ML por hora, (e) ensemble pooled+
    por-hora con peso causal mes a mes (mismo patrón ya validado 3 veces:
    España Fase 17, Francia Fase 4, conjunto ES+FR).

    `features` (opcional): permite extender COMMON_FEATURES con columnas
    propias de un solo país (p.ej. `net_import_de_clim` de Alemania,
    findings.md #106) sin afectar a los demás países que reutilizan esta
    misma función con la lista por defecto."""
    feats = features if features is not None else COMMON_FEATURES
    df = df.copy()
    df["is_holiday"] = df["is_holiday"].astype(int)
    df["is_weekend"] = df["is_weekend"].astype(int)

    df["pred_pooled"] = walk_forward(df, feats, "price_real", "pooled", log)
    feats_hora = [f for f in feats if f not in ("hour_of_day", "period_of_day")]
    df["pred_por_hora"] = walk_forward_by_group(df, feats_hora, "price_real", "hour_of_day", "por hora", log)

    mask = df["pred_pooled"].notna() & df["pred_por_hora"].notna()
    d = df[mask].sort_values("period_start_utc").copy()
    d["year_month"] = d["period_start_utc"].dt.to_period("M")
    months = sorted(d["year_month"].unique())

    d["pred_ensemble"] = np.nan
    for i, m in enumerate(months):
        if i < WARMUP_MONTHS_ENSEMBLE:
            continue
        hist = d[d["year_month"].isin(months[:i])]
        maes = [
            (hist["price_real"] - (w * hist["pred_pooled"] + (1 - w) * hist["pred_por_hora"])).abs().mean()
            for w in PESOS_CANDIDATOS
        ]
        w_best = PESOS_CANDIDATOS[int(np.argmin(maes))]
        cur = d[d["year_month"] == m]
        d.loc[cur.index, "pred_ensemble"] = w_best * cur["pred_pooled"] + (1 - w_best) * cur["pred_por_hora"]

    valid = d.dropna(subset=["pred_ensemble"])
    log(f"\n-- Comparación, periodo tras calentamiento ({len(valid):,} periodos) --")
    resultados = {}
    resultados["baseline"] = report(valid, "price_lag_24h", "(a) Baseline naive: price_lag_24h", log)
    resultados["analogo"] = report(valid, "analog_price_mean", "(a') Análogo solo", log)
    resultados["pooled"] = report(valid, "pred_pooled", "(b) ML flexible pooled", log)
    resultados["por_hora"] = report(valid, "pred_por_hora", "(d) ML por hora", log)
    resultados["ensemble"] = report(valid, "pred_ensemble", "(e) Ensemble (peso causal)", log)
    return {"df": valid, "resultados": resultados}


def add_net_import_climatology(df: pd.DataFrame, area_key: str, log) -> pd.DataFrame:
    """findings.md #106 (2026-09-01): importación neta física (suma de los
    flujos en TODAS las fronteras de `area_key`, ya cargadas en
    `interconnections.duckdb`) como climatología D-1-segura (media móvil de
    4 semanas del mismo día de la semana/hora, desplazada 1 — mismo criterio
    ya validado para España/Francia en `merit_order`). Probada en vivo antes
    de integrar: correlación contemporánea (cota superior, no D-1-segura)
    con el precio +0,47 en Alemania — candidato real; +0,03 en Países Bajos
    — descartada allí, no se generaliza."""
    from studies.merit_order.build_dataset import _add_climatology, DATA_DIR as _DD

    con = duckdb.connect(str(_DD / "interconnections.duckdb"), read_only=True)
    try:
        flows = con.execute(
            "SELECT interval_start_utc AS hour_utc, from_area, to_area, flow_mw FROM cross_border_flows "
            "WHERE from_area = ? OR to_area = ?",
            [area_key, area_key],
        ).fetchdf()
    finally:
        con.close()
    flows["hour_utc"] = pd.to_datetime(flows["hour_utc"])
    flows["net_import_raw"] = np.where(flows["to_area"] == area_key, flows["flow_mw"], -flows["flow_mw"])
    net = flows.groupby("hour_utc")["net_import_raw"].sum().reset_index()

    net = _add_climatology(net, ["net_import_raw"], hour_col="hour_utc")
    col = f"net_import_{area_key}_clim"
    net = net.rename(columns={"net_import_raw_clim": col})
    df = df.merge(net[["hour_utc", col]], on="hour_utc", how="left")
    log(f"  Importación neta {area_key} (climatología): {df[col].notna().sum():,}/{len(df):,} horas con dato")
    return df, col


def add_cross_country_price_lag(df: pd.DataFrame, other_db: str, other_table: str, other_pricecol: str,
                                 other_tcol: str, prefix: str, log) -> pd.DataFrame:
    """findings.md #106 (2026-09-01): lag de precio de un país VECINO como
    feature — mismo patrón que España usa con Francia/Portugal
    (`price_lag_24h_fr`/`_pt`), nunca probado para Alemania/Países Bajos.
    Solo tiene sentido probarlo donde el vecino aporte información NUEVA,
    no redundante con el propio lag del país — comprobado en vivo antes de
    construir: Alemania-Países Bajos están tan acoplados (correlación
    contemporánea 0,958) que el lag del vecino da prácticamente la MISMA
    correlación con el precio propio que el lag propio (0,685 vs 0,685) —
    candidato de expectativa baja, se prueba de todos modos por rigor."""
    con = duckdb.connect(str(DATA_DIR / other_db), read_only=True)
    try:
        other = con.execute(f"SELECT {other_tcol} AS t, {other_pricecol} AS price FROM {other_table}").fetchdf()
    finally:
        con.close()
    other["t"] = pd.to_datetime(other["t"])
    price_by_time = other.set_index("t")["price"]

    col24 = f"price_lag_24h_{prefix}"
    col168 = f"price_lag_168h_{prefix}"
    df[col24] = (df["period_start_utc"] - pd.Timedelta(hours=24)).map(price_by_time)
    df[col168] = (df["period_start_utc"] - pd.Timedelta(hours=168)).map(price_by_time)
    log(f"  Lag de precio de {prefix} (24h/168h): nulos={df[col24].isna().sum()}/{df[col168].isna().sum()}")
    return df, [col24, col168]


def _load_forecast_d1(con: duckdb.DuckDBPyConnection, log) -> pd.DataFrame:
    demanda = con.execute(
        "SELECT interval_start_utc AS hour_utc, load_forecast_mw AS forecast_demanda_mw FROM entsoe_load_forecast"
    ).fetchdf()
    gen = con.execute(
        "SELECT interval_start_utc AS hour_utc, psr_type, generation_forecast_mw FROM entsoe_generation_forecast"
    ).fetchdf()
    gen_wide = gen.pivot_table(index="hour_utc", columns="psr_type", values="generation_forecast_mw", aggfunc="mean")
    for col in ("B16", "B18", "B19"):
        if col not in gen_wide.columns:
            gen_wide[col] = np.nan
    gen_wide = gen_wide.reset_index()
    gen_wide["forecast_eolica_mw"] = gen_wide["B19"].fillna(0) + gen_wide["B18"].fillna(0)
    gen_wide["forecast_solar_mw"] = gen_wide["B16"]
    df = demanda.merge(gen_wide[["hour_utc", "forecast_eolica_mw", "forecast_solar_mw"]], on="hour_utc", how="inner")
    df["hour_utc"] = pd.to_datetime(df["hour_utc"])
    log(f"  Previsión D-1 cargada: {len(df):,} horas, {df['hour_utc'].min()} -> {df['hour_utc'].max()}")
    return df


def _load_native_price(con: duckdb.DuckDBPyConnection, price_table: str, log) -> pd.DataFrame:
    df = con.execute(
        f"SELECT interval_start_utc AS period_start_utc, resolution_minutes, price_eur_mwh AS price_real "
        f"FROM {price_table} ORDER BY 1"
    ).fetchdf()
    df["period_start_utc"] = pd.to_datetime(df["period_start_utc"])
    df["period_in_hour"] = (df["period_start_utc"].dt.minute // 15).astype(int)
    n_hourly = (df["resolution_minutes"] == 60).sum()
    n_quarter = (df["resolution_minutes"] == 15).sum()
    log(f"  Precio nativo cargado: {n_hourly:,} horarios + {n_quarter:,} de 15 min = {len(df):,} filas")
    return df


def _expand_to_native(hourly_df: pd.DataFrame, native_price: pd.DataFrame) -> pd.DataFrame:
    native_price = native_price.copy()
    native_price["hour_key"] = native_price["period_start_utc"].dt.floor("h")
    df = native_price.merge(hourly_df, left_on="hour_key", right_on="hour_utc", how="left")
    df = df.drop(columns=["hour_key", "hour_utc"])
    df["hour_of_day"] = df["period_start_utc"].dt.hour
    df["day_of_week"] = df["period_start_utc"].dt.dayofweek
    df["month"] = df["period_start_utc"].dt.month
    df["period_of_day"] = df["hour_of_day"] * 4 + df["period_in_hour"]
    return df


def _add_holidays(df: pd.DataFrame, con: duckdb.DuckDBPyConnection, holiday_key: str) -> pd.DataFrame:
    holidays = con.execute(
        "SELECT holiday_date FROM holiday_calendar WHERE location_key = ?", [holiday_key]
    ).fetchdf()
    holiday_dates = set(pd.to_datetime(holidays["holiday_date"]).dt.date)
    df["date"] = df["period_start_utc"].dt.date
    df["is_holiday"] = df["date"].isin(holiday_dates)
    df["is_weekend"] = df["day_of_week"] >= 5
    df["is_dia_no_laborable"] = df["is_holiday"] | df["is_weekend"]
    return df


def _add_commodities(df: pd.DataFrame, log) -> pd.DataFrame:
    con = duckdb.connect(str(DATA_DIR / "commodities.duckdb"), read_only=True)
    try:
        commodities = _load_commodities(con)
    finally:
        con.close()
    df["trade_date_cutoff"] = (df["hour_utc"] - pd.Timedelta(days=2)).dt.normalize()
    df = df.sort_values("trade_date_cutoff")
    commodities = commodities.sort_values("trade_date")
    df = pd.merge_asof(df, commodities, left_on="trade_date_cutoff", right_on="trade_date", direction="backward")
    from studies.merit_order.build_dataset import COAL_MWH_PER_TONNE
    df["coal_eur_mwh_th"] = (df["coal_usd_t"] / df["eur_usd"]) / COAL_MWH_PER_TONNE
    log(f"  Materias primas (D-2): ttf nulos={df['ttf_eur_mwh'].isna().sum()}, coal nulos={df['coal_eur_mwh_th'].isna().sum()}")
    return df.sort_values("hour_utc").reset_index(drop=True)


def _add_degree_days(df: pd.DataFrame, location_key: str, log) -> pd.DataFrame:
    """HDD/CDD reales de AYER (lag 1 día) — D-1-seguro por construcción, el
    dato de ayer siempre se conoce antes de que cierre la subasta de hoy.
    findings.md #106: probado en vivo, correlación univariada real con el
    precio antes de integrar (no solo intuición)."""
    con = duckdb.connect(str(DATA_DIR / "weather.duckdb"), read_only=True)
    try:
        hdd = con.execute(
            "SELECT calendar_date AS date, hdd, cdd FROM weather_degree_days WHERE location_key = ? ORDER BY 1",
            [location_key],
        ).fetchdf()
    finally:
        con.close()
    hdd = hdd.rename(columns={"date": "_hdd_date"})
    hdd["_hdd_date"] = pd.to_datetime(hdd["_hdd_date"])
    hdd["hdd_lag1"] = hdd["hdd"].shift(1)
    hdd["cdd_lag1"] = hdd["cdd"].shift(1)
    df["_date_only"] = pd.to_datetime(df["period_start_utc"].dt.date)
    df = df.merge(hdd[["_hdd_date", "hdd_lag1", "cdd_lag1"]], left_on="_date_only", right_on="_hdd_date", how="left")
    df = df.drop(columns=["_date_only", "_hdd_date"])
    log(f"  HDD/CDD (lag 1 día): nulos hdd_lag1={df['hdd_lag1'].isna().sum()}, cdd_lag1={df['cdd_lag1'].isna().sum()}")
    return df


def build_simple_country(db_name: str, price_table: str, holiday_key: str, log) -> pd.DataFrame:
    con = duckdb.connect(str(DATA_DIR / db_name), read_only=True)
    try:
        hourly_df = _load_forecast_d1(con, log)
        hourly_df = _add_commodities(hourly_df, log)
        native_price = _load_native_price(con, price_table, log)
        df = _expand_to_native(hourly_df, native_price)
        df = _add_holidays(df, con, holiday_key)
    finally:
        con.close()

    log("Añadiendo HDD/CDD reales de ayer (findings.md #106)...")
    df = _add_degree_days(df, holiday_key, log)

    log("Construyendo el análogo (demanda + ratio renovable + ttf, borrow_parent_hour_history)...")
    df["ratio_renovable_periodo"] = (df["forecast_eolica_mw"] + df["forecast_solar_mw"]) / df["forecast_demanda_mw"]
    df = _build_analog_generic(
        df, ["forecast_demanda_mw", "ratio_renovable_periodo", "ttf_eur_mwh"], "analog_price_mean", log,
        f"{db_name} (demanda+ratio+ttf)", borrow_parent_hour_history=True,
    )

    log("Añadiendo lags de precio por timestamp exacto...")
    df = _add_price_lags(df, log)

    return df.sort_values("period_start_utc").reset_index(drop=True)
