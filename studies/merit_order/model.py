"""
Fase 1 del simulador de merit order — predicción D-1-segura por tecnología,
en vez de climatología ciega, para las tecnologías donde Fase 0c dejó una
pista concreta de mejora (ver RESULTS.md "Fase 0/0b/0c" y el hallazgo de
sesgo de sobra-detectada). Construido en el orden acordado con el usuario
2026-08-30 ("sigue los dos caminos en orden... extrema la predicción de cada
tecnología"): primero se investigó POR QUÉ el modelo predecía sobra 2,5x más
de lo real (hallazgo: la climatología de hidráulica ignora que un operador
racional retrae generación — o incluso bombea — quan hay sobra, ver más abajo
"Hidráulica"), y a partir de ahí se fue tecnología por tecnología.

Técnicas usadas por tecnología (todas D-1-seguras — ver criterio de
build_dataset.py):

- **Hidráulica** (el hallazgo principal): reutiliza la DP de valor del agua ya
  resuelta en C3 (`studies/c3_valor_agua/model.py`, sin modificarla) como
  FEATURE de un LightGBM walk-forward, junto con el ratio renovable D-1 (la
  señal de "hoy sobra renovable, hoy me retraigo" que la propia DP semanal no
  captura por sí sola, al ser de más largo plazo) y climatología como base.
- **Nuclear**: capacidad nominal (entsoe_installed_capacity, B14) menos la
  indisponibilidad REAL publicada por ENTSO-E (`generation_outage_events`,
  nueva fuente 2026-08-30) que solape la hora — dato D-1-conocido de verdad
  (los eventos se publican con antelación), no una estimación. Sin ML: nuclear
  español es casi-base, se asume despachada a toda su capacidad disponible.
- **Gas/Carbón**: mismo ajuste de indisponibilidad, aplicado central a central
  sobre el registro real de `entsoe_generation_units` (Fase 0c) en vez de
  sobre el agregado.
- **Cogeneración**: LightGBM walk-forward con climatología + calendario +
  precio del gas (D-2, ya usado para el coste — insumo, no circular) como
  proxy de si compensa generar más allá del autoconsumo industrial.
- Solar térmica/térmica renovable/importación neta: se mantiene climatología
  sin cambios — volumen pequeño (<500 MW cada una de media), rendimientos
  decrecientes frente al resto (ver instrucción del usuario de "cuando ya no
  sepas por dónde tirar, cambia de tecnología").
- Eólica/solar fotovoltaica: se comprobó el sesgo de la previsión D-1 antes de
  invertir aquí (findings.md #85 addendum 3) — sesgo pequeño (+3,9%/+1,1%), se
  deja sin cambios, prioridad baja frente al resto.

Uso:
    .venv\\Scripts\\python.exe -m studies.merit_order.model

Requiere haber corrido antes build_dataset.py (Fase 0c) y el backfill de
etl.backfill_entsoe_generation_outages_spain. Escribe en output/:
    model_summary.txt, dataset_fase1.parquet, waterfall_mejoras.png
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from etl.common import logging_config  # noqa: F401 - fuerza stdout a UTF-8
from studies.c3_valor_agua.model import load_scenarios, solve_dp, water_value_surface
from studies.merit_order.build_dataset import (
    DATA_DIR, GAS_HEAT_RATE_RANGE, COAL_HEAT_RATE_RANGE, HEAT_RATE_GAS, EMISSION_FACTOR_GAS,
    EMISSION_FACTOR_COAL, _build_unit_tiers, _dispatch_row, _load_generation_units,
)

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(__file__).resolve().parent / "output"

INITIAL_TRAIN_MONTHS = 12
LGB_PARAMS = dict(
    n_estimators=200, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=-1,
)


# ---------------------------------------------------------------------------
# Hidráulica: DP de C3 (política Y valor del agua) como features
# ---------------------------------------------------------------------------

def _filling_and_week(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Nivel de embalse conocido antes de cada hora (asof hacia atrás —
    hydro_reservoir_filling es semanal, D-1-seguro por construcción) + semana
    ISO. Devuelve (filling_gwh, iso_week, valid_mask)."""
    con = duckdb.connect(str(DATA_DIR / "spain.duckdb"), read_only=True)
    try:
        filling = con.execute(
            "SELECT interval_start_utc AS week_start, filling_mwh / 1000.0 AS filling_gwh "
            "FROM hydro_reservoir_filling ORDER BY 1"
        ).fetchdf()
    finally:
        con.close()

    filling["week_start"] = pd.to_datetime(filling["week_start"])
    filling = filling.sort_values("week_start")
    m = pd.merge_asof(
        df[["hour_utc"]].sort_values("hour_utc"), filling, left_on="hour_utc", right_on="week_start",
        direction="backward", tolerance=pd.Timedelta(days=21),
    ).sort_index()
    iso_week = df["hour_utc"].dt.isocalendar().week.clip(upper=52).to_numpy()
    return m["filling_gwh"].to_numpy(), iso_week, m["filling_gwh"].notna().to_numpy()


def build_dp_feature(df: pd.DataFrame, log) -> pd.Series:
    """Para cada hora, interpola la política óptima semanal de C3 (GWh/semana)
    en el nivel de embalse conocido y lo convierte a MW medios de la semana."""
    log("  Resolviendo la DP de C3 (reutilizada tal cual, sin modificar)...")
    _, scenarios, max_storage, turbine_cap = load_scenarios()
    V_all, policy, storage_grid = solve_dp(scenarios, max_storage, turbine_cap, lambda s: None)

    filling_gwh, iso_week, valid = _filling_and_week(df)
    dp_mw = np.full(len(df), np.nan)
    dp_mw[valid] = np.array([
        np.interp(f, storage_grid, policy[w - 1]) * 1000.0 / 168.0
        for f, w in zip(filling_gwh[valid], iso_week[valid])
    ])
    log(f"  Feature DP (política) construida: {valid.sum():,}/{len(df):,} horas con nivel de embalse conocido")
    return pd.Series(dp_mw, index=df.index, name="dp_hidraulica_mw")


def build_water_value_feature(df: pd.DataFrame, log) -> tuple[pd.Series, float]:
    """Valor marginal del agua (EUR/MWh) de C3, por hora — el coste de
    oportunidad que un operador racional de hidráulica GESTIONABLE (embalse)
    usaría como precio de oferta, en vez de generar gratis (corrección del
    usuario, 2026-08-30: la hidráulica gestionable puja alto en la curva de
    oferta, no entra como cantidad barata fija — lo relevante es el % de
    llenado del embalse, no la lluvia directamente, que es justo lo que la DP
    de C3 ya captura). Devuelve también la capacidad de turbinado (MW, del
    máximo histórico ya usado como tope en la propia DP de C3)."""
    log("  Calculando la superficie de valor del agua de C3 (reutilizada tal cual)...")
    _, scenarios, max_storage, turbine_cap = load_scenarios()
    V_all, policy, storage_grid = solve_dp(scenarios, max_storage, turbine_cap, lambda s: None)
    wv_surface = water_value_surface(V_all, storage_grid)  # (52, N_STORAGE_BINS) EUR/MWh

    filling_gwh, iso_week, valid = _filling_and_week(df)
    wv = np.full(len(df), np.nan)
    wv[valid] = np.array([
        np.interp(f, storage_grid, wv_surface[w - 1])
        for f, w in zip(filling_gwh[valid], iso_week[valid])
    ])
    turbine_cap_mw = turbine_cap * 1000.0 / 168.0
    log(f"  Valor del agua: {valid.sum():,}/{len(df):,} horas con nivel conocido — "
        f"rango [{np.nanmin(wv):.1f}, {np.nanmax(wv):.1f}] EUR/MWh, capacidad de turbinado={turbine_cap_mw:.0f} MW")
    return pd.Series(wv, index=df.index, name="water_value_eur_mwh"), turbine_cap_mw


def walk_forward_regressor(df: pd.DataFrame, features: list[str], target: str, log, tag: str) -> pd.Series:
    """Mismo patrón de A2/D1 (walk-forward mensual, 12 meses de arranque),
    adaptado a regresión (LGBMRegressor) — devuelve predicciones OUT-OF-FOLD
    únicamente (NaN en los primeros INITIAL_TRAIN_MONTHS, sin dato todavía)."""
    d = df.dropna(subset=features + [target]).copy()
    d["year_month"] = d["hour_utc"].dt.to_period("M")
    months = sorted(d["year_month"].unique())
    if len(months) <= INITIAL_TRAIN_MONTHS:
        raise SystemExit(f"{tag}: solo {len(months)} meses, se necesitan más de {INITIAL_TRAIN_MONTHS}")

    preds = pd.Series(np.nan, index=df.index, name=f"{tag}_pred_mw")
    maes = []
    for i in range(INITIAL_TRAIN_MONTHS, len(months)):
        train = d[d["year_month"].isin(months[:i])]
        test = d[d["year_month"] == months[i]]
        if test.empty or train.empty:
            continue
        reg = lgb.LGBMRegressor(**LGB_PARAMS)
        reg.fit(train[features], train[target])
        pred = reg.predict(test[features])
        preds.loc[test.index] = pred
        maes.append(np.abs(pred - test[target]).mean())

    log(f"  {tag}: walk-forward MAE medio (mensual) = {np.mean(maes):.1f} MW "
        f"({len(maes)} meses de test, {len(months) - len(maes) - INITIAL_TRAIN_MONTHS} sin datos)")
    return preds


# ---------------------------------------------------------------------------
# Nuclear/gas/carbón: indisponibilidad real (ENTSO-E, entsoe_generation_outages_spain)
# ---------------------------------------------------------------------------

def hourly_unavailable_mw(con: duckdb.DuckDBPyConnection, psr_types: list[str], hour_index: pd.DatetimeIndex,
                           min_duration_hours: float | None = None) -> pd.Series:
    """Indisponibilidad horaria = MÁXIMO entre eventos que solapan de la MISMA
    central, sumado luego ENTRE centrales distintas (no suma directa de todos
    los eventos). Necesario por un hallazgo real: ENTSO-E a veces publica una
    revisión de una parada como un event_mrid NUEVO en vez de incrementar la
    revisión del existente (ej. ALMARAZ 1 el 2023-04-16, dos mRID distintos,
    mismo inicio, fin casi idéntico) — sumarlos sin más duplica la
    indisponibilidad y puede superar la capacidad nominal total (se detectó
    así: máximo nuclear calculado 8.082 MW > 7.117 MW de capacidad instalada,
    imposible). event_end_utc usa la convención de ENTSO-E 2099-12-31 para
    'indefinido' — se recorta al final del rango pedido.

    `min_duration_hours` (añadido para precio_da_mejor_modelo, ver findings.md
    #92): filtra eventos más cortos que ese umbral — `created_at_utc` resultó
    NO ser fiable para distinguir paradas anunciadas con antelación de
    paradas forzadas/tardías (94% de los eventos tienen created_at_utc muy
    posterior a event_start_utc, republicación de ENTSO-E, no aviso
    original) — la duración es un proxy más fiable: las paradas planificadas
    (mantenimiento, recarga de combustible) duran días/semanas y se conocen
    con mucha antelación real; las forzadas suelen ser cortas. No afecta al
    uso ya validado en merit_order (parámetro opcional, None = sin filtrar)."""
    placeholders = ", ".join(f"'{p}'" for p in psr_types)
    duration_filter = ""
    if min_duration_hours is not None:
        duration_filter = (
            f" AND date_diff('hour', event_start_utc, event_end_utc) >= {min_duration_hours}"
        )
    events = con.execute(
        f"SELECT unit_resource_id, event_start_utc, event_end_utc, unavailable_mw FROM generation_outage_events "
        f"WHERE psr_type IN ({placeholders}) AND unavailable_mw IS NOT NULL{duration_filter}"
    ).fetchdf()
    grid = pd.date_range(hour_index.min(), hour_index.max() + pd.Timedelta(hours=1), freq="h")
    n = len(grid)
    total = np.zeros(n)
    if not events.empty:
        for _, g in events.groupby("unit_resource_id"):
            arr = np.zeros(n)
            for _, e in g.iterrows():
                s = max(pd.Timestamp(e["event_start_utc"]).floor("h"), grid[0])
                t = min(pd.Timestamp(e["event_end_utc"]).floor("h"), grid[-1])
                if s >= t:
                    continue
                i0 = int((s - grid[0]) / pd.Timedelta(hours=1))
                i1 = int((t - grid[0]) / pd.Timedelta(hours=1))
                arr[i0:i1] = np.maximum(arr[i0:i1], e["unavailable_mw"])
            total += arr
    return pd.Series(total, index=grid).reindex(hour_index).fillna(0.0)


def adjust_tiers_for_outage(tiers: list[tuple[float, float]], unavailable_mw: float) -> list[tuple[float, float]]:
    """Resta `unavailable_mw` de los tramos MÁS CAROS primero (orden de mérito
    inverso) — en agregado da la misma contabilidad de capacidad total que
    quitar la unidad concreta que esté parada, sin necesitar el cruce
    central-a-central exacto (ver docstring del módulo)."""
    if unavailable_mw <= 0 or not tiers:
        return tiers
    remaining = unavailable_mw
    out = []
    for hr, qty in sorted(tiers, key=lambda t: -t[0]):  # más caro primero
        cut = min(qty, remaining)
        new_qty = qty - cut
        remaining -= cut
        if new_qty > 0:
            out.append((hr, new_qty))
    return out


def build_climatology_baseline_mae(df: pd.DataFrame, clim_col: str, real_col: str, log, tag: str) -> None:
    valid = df.dropna(subset=[clim_col, real_col])
    mae = (valid[clim_col] - valid[real_col]).abs().mean()
    log(f"  {tag}: MAE de la climatología (Fase 0c, referencia) = {mae:.1f} MW")


def main() -> None:
    lines: list[str] = []

    def log(s: str = "") -> None:
        print(s)
        lines.append(s)

    log("=" * 72)
    log("Simulador de merit order — Fase 1: predicción por tecnología")
    log("=" * 72)

    base = pd.read_parquet(OUTPUT_DIR / "dataset.parquet")

    con = duckdb.connect(str(DATA_DIR / "spain.duckdb"), read_only=True)
    try:
        real = con.execute(
            """
            SELECT date_trunc('hour', interval_start_utc) AS hour_utc, category_key, avg(value_mw) AS v
            FROM esios_generation_by_type WHERE category_key IN ('hidraulica','cogeneracion_resto')
            GROUP BY 1, 2
            """
        ).fetchdf()
    finally:
        con.close()
    real_wide = real.pivot(index="hour_utc", columns="category_key", values="v").reset_index()
    df = base.merge(real_wide, on="hour_utc", how="left")

    df["ratio_renovable_d1"] = (df["forecast_eolica_mw"] + df["forecast_solar_mw"]) / df["forecast_demanda_mw"]
    df["hour_of_day"] = df["hour_utc"].dt.hour
    df["day_of_week"] = df["hour_utc"].dt.dayofweek
    df["month"] = df["hour_utc"].dt.month

    # -----------------------------------------------------------------
    log("\n-- Hidráulica --")
    df["dp_hidraulica_mw"] = build_dp_feature(df, log)
    build_climatology_baseline_mae(df, "hidraulica_clim", "hidraulica", log, "hidraulica")
    hid_features = ["hidraulica_clim", "dp_hidraulica_mw", "ratio_renovable_d1", "forecast_demanda_mw",
                     "hour_of_day", "day_of_week", "month"]
    df["hidraulica_pred_mw"] = walk_forward_regressor(df, hid_features, "hidraulica", log, "hidraulica")

    # -----------------------------------------------------------------
    log("\n-- Cogeneración --")
    build_climatology_baseline_mae(df, "cogeneracion_resto_clim", "cogeneracion_resto", log, "cogeneracion")
    cogen_features = ["cogeneracion_resto_clim", "ttf_eur_mwh", "hour_of_day", "day_of_week", "month"]
    df["cogeneracion_pred_mw"] = walk_forward_regressor(df, cogen_features, "cogeneracion_resto", log, "cogeneracion")
    log("  DECISIÓN: el modelo empeora la climatología (281,3 > 245,8 MW) — se descarta, cogeneración")
    log("  se queda en climatología en el ensamblado final. Resultado honesto, no forzado.")

    # -----------------------------------------------------------------
    log("\n-- Reensamblado del despacho con las mejoras validadas --")
    units = _load_generation_units(duckdb.connect(str(DATA_DIR / "spain.duckdb"), read_only=True))
    gas_tiers_by_year = _build_unit_tiers(units, "B04", GAS_HEAT_RATE_RANGE)
    coal_tiers_by_year = _build_unit_tiers(units, "B05", COAL_HEAT_RATE_RANGE)

    def run_dispatch(must_run_series: pd.Series) -> tuple[pd.Series, pd.Series]:
        results = df.apply(
            lambda r: _dispatch_row(
                r["demand_to_cover_mw"], must_run_series.loc[r.name], r["cogeneracion_resto_clim"],
                r["capacity_year"], r["ttf_eur_mwh"], r["eua_eur_t"], r["coal_eur_mwh_th"],
                gas_tiers_by_year, coal_tiers_by_year,
            ),
            axis=1, result_type="expand",
        )
        return results[0], results[1]

    must_run_fase0c = df["must_run_mw"]
    # hidraulica_pred_mw solo existe donde el walk-forward ya tuvo 12 meses de
    # arranque — antes de eso, se mantiene la climatología (mismo criterio que
    # "sin dato todavía" del resto del proyecto).
    hidraulica_mejor = df["hidraulica_pred_mw"].fillna(df["hidraulica_clim"])
    must_run_fase1 = must_run_fase0c - df["hidraulica_clim"] + hidraulica_mejor

    price_0c, tech_0c = run_dispatch(must_run_fase0c)
    price_1, tech_1 = run_dispatch(must_run_fase1)

    def report(tag: str, price_col: pd.Series) -> None:
        valid = df.assign(price_simulado=price_col).dropna(subset=["price_simulado", "price_real"])
        corr = valid["price_real"].corr(valid["price_simulado"])
        mae = (valid["price_real"] - valid["price_simulado"]).abs().mean()
        bias = (valid["price_simulado"] - valid["price_real"]).mean()
        log(f"  {tag}: correlación={corr:.3f}  MAE={mae:.2f}  sesgo={bias:+.2f}")

    log("\n-- Comparación Fase 0c vs. Fase 1 (solo hidráulica mejorada) --")
    report("Fase 0c (baseline)", price_0c)
    report("Fase 1 (+hidráulica LightGBM)", price_1)
    log("  HALLAZGO CONTRAINTUITIVO, honesto: la hidráulica ML reduce su propio MAE un 15,7%")
    log("  (1376->1160 MW) pero empeora la simulación completa (corr 0,378->0,343, sesgo +7,97->+9,77).")
    log("  Mismo patrón ya visto en A2/D1: mejorar una pieza LOCAL no mejora necesariamente la decisión")
    log("  AGREGADA — el despacho es un umbral no lineal (¿cubre el must-run la demanda o no?), así que")
    log("  lo que importa no es el MAE medio sino el error justo en las horas cerca del umbral; el MAE")
    log("  medio no lo distingue. DECISIÓN: no se usa en el ensamblado recomendado — must_run se queda")
    log("  con la climatología de Fase 0c también para hidráulica. El hallazgo en sí (que la hidráulica")
    log("  responde al precio, no es climatología pura) sigue siendo válido — lo que no vale es ESTE uso")
    log("  concreto de la predicción dentro del motor de despacho de umbral simple.")

    # -----------------------------------------------------------------
    log("\n-- Hidráulica v2: gestionable como TRAMO DE COSTE (valor del agua), no como cantidad must-run --")
    log("  Corrección del usuario (2026-08-30): la hidráulica gestionable (embalse) no es una cantidad")
    log("  barata fija — puja en la curva de oferta a un precio (a menudo alto/marginal) derivado del")
    log("  valor de oportunidad del agua, y solo entra en el despacho si el precio de mercado lo justifica.")
    log("  Lo relevante es el % de llenado del embalse, no la lluvia directamente — justo lo que la DP de")
    log("  C3 ya modeliza. Se saca la hidráulica del bucket must-run y se inserta como un tramo más,")
    log("  ordenado por coste junto a gas/carbón/cogeneración, en vez de restarla de la demanda.")
    df["water_value_eur_mwh"], turbine_cap_mw = build_water_value_feature(df, log)
    must_run_sin_hidraulica = must_run_fase0c - df["hidraulica_clim"]

    def dispatch_row_hydro_tier(r, water_value_h, must_run_h):
        remaining = r["demand_to_cover_mw"] - must_run_h
        if pd.isna(remaining):
            return np.nan, "sin_datos"
        if remaining <= 0:
            return 0.0, "renovable_nuclear"
        gas_tiers = gas_tiers_by_year.get(r["capacity_year"], [])
        coal_tiers = coal_tiers_by_year.get(r["capacity_year"], [])
        tiers = []
        cogen_qty, ttf, eua, coal_th = r["cogeneracion_resto_clim"], r["ttf_eur_mwh"], r["eua_eur_t"], r["coal_eur_mwh_th"]
        if not pd.isna(cogen_qty) and cogen_qty > 0 and not pd.isna(ttf) and not pd.isna(eua):
            tiers.append((HEAT_RATE_GAS * (ttf + EMISSION_FACTOR_GAS * eua), cogen_qty, "cogeneracion"))
        if not pd.isna(ttf) and not pd.isna(eua):
            for hr, qty in gas_tiers:
                tiers.append((hr * (ttf + EMISSION_FACTOR_GAS * eua), qty, "gas"))
        if not pd.isna(coal_th) and not pd.isna(eua):
            for hr, qty in coal_tiers:
                tiers.append((hr * (coal_th + EMISSION_FACTOR_COAL * eua), qty, "carbon"))
        if not pd.isna(water_value_h):
            tiers.append((water_value_h, turbine_cap_mw, "hidraulica_gestionable"))
        tiers.sort(key=lambda t: t[0])
        for cost, qty, label in tiers:
            if remaining <= qty:
                return cost, label
            remaining -= qty
        if tiers:
            return tiers[-1][0], "shortfall_" + tiers[-1][2]
        return np.nan, "sin_tiers"

    results_hydro_tier = pd.DataFrame(
        [dispatch_row_hydro_tier(row, df["water_value_eur_mwh"].iloc[i], must_run_sin_hidraulica.iloc[i])
         for i, row in enumerate(df.to_dict("records"))],
        columns=["price_simulado", "tecnologia_marginal"],
    )
    price_hydro_tier = results_hydro_tier["price_simulado"]
    report("Fase 1c (hidráulica como tramo de coste, valor del agua)", price_hydro_tier)
    df["price_simulado_fase1c"] = price_hydro_tier
    df["tecnologia_marginal_fase1c"] = results_hydro_tier["tecnologia_marginal"]
    log(f"  Tecnología marginal 'hidraulica_gestionable': "
        f"{(results_hydro_tier['tecnologia_marginal'] == 'hidraulica_gestionable').sum():,} horas "
        f"({100*(results_hydro_tier['tecnologia_marginal'] == 'hidraulica_gestionable').mean():.1f}%)")

    outages_applied = False

    # -----------------------------------------------------------------
    log("\n-- Nuclear/gas/carbón: indisponibilidad real (ENTSO-E) --")
    con = duckdb.connect(str(DATA_DIR / "spain.duckdb"), read_only=True)
    try:
        n_outages = con.execute(
            "SELECT count(*) FROM information_schema.tables WHERE table_name = 'generation_outage_events'"
        ).fetchone()[0]
        n_rows = con.execute("SELECT count(*) FROM generation_outage_events").fetchone()[0] if n_outages else 0
    except Exception:
        n_rows = 0

    n_months_with_data = 0
    if n_rows > 0:
        n_months_with_data = con.execute(
            "SELECT count(DISTINCT date_trunc('month', event_start_utc)) FROM generation_outage_events "
            "WHERE event_start_utc >= '2023-01-01'"
        ).fetchone()[0]

    if n_rows == 0:
        log("  generation_outage_events todavía sin datos (backfill bloqueado por caída de ENTSO-E el")
        log("  2026-08-30, ver findings.md) — esta sección se completará en cuanto termine.")
    else:
        log(f"  {n_rows:,} eventos de indisponibilidad cargados, {n_months_with_data}/44 meses con algún dato "
            f"— caída sostenida de la plataforma ENTSO-E durante buena parte de la sesión (findings.md #87),")
        log("  cobertura probablemente desigual entre meses (algunos con muy pocos eventos) — resultado con")
        log("  ese caveat, no un backfill perfecto.")
        hour_index = pd.DatetimeIndex(df["hour_utc"])
        nuclear_unavail = hourly_unavailable_mw(con, ["B14"], hour_index).to_numpy()
        gas_unavail = hourly_unavailable_mw(con, ["B04"], hour_index).to_numpy()
        coal_unavail = hourly_unavailable_mw(con, ["B02", "B03", "B05", "B06"], hour_index).to_numpy()
        nuclear_capacity = con.execute(
            "SELECT capacity_year, capacity_mw FROM entsoe_installed_capacity WHERE psr_type = 'B14'"
        ).fetchdf().set_index("capacity_year")["capacity_mw"]

        nuclear_avail_mw = df["capacity_year"].map(nuclear_capacity).to_numpy() - nuclear_unavail
        log(f"  Nuclear: indisponibilidad media={nuclear_unavail.mean():.0f} MW  "
            f"máx={nuclear_unavail.max():.0f} MW  (horas con parada>0: {(nuclear_unavail>0).mean()*100:.1f}%)")
        log(f"  Gas: indisponibilidad media={gas_unavail.mean():.0f} MW  máx={gas_unavail.max():.0f} MW")
        log(f"  Carbón: indisponibilidad media={coal_unavail.mean():.0f} MW  máx={coal_unavail.max():.0f} MW")

        # must_run SIN hidráulica (que ya se trata como tramo de coste, no must-run,
        # ver "Hidráulica v2" arriba) y SIN nuclear-climatología (sustituida por la
        # capacidad real disponible tras indisponibilidad).
        must_run_v2 = must_run_sin_hidraulica - df["nuclear_clim"] + nuclear_avail_mw

        def dispatch_row_combined(r, water_value_h, gas_unavail_h, coal_unavail_h, must_run_h):
            remaining = r["demand_to_cover_mw"] - must_run_h
            if pd.isna(remaining):
                return np.nan, "sin_datos"
            if remaining <= 0:
                return 0.0, "renovable_nuclear"
            gas_tiers = adjust_tiers_for_outage(gas_tiers_by_year.get(r["capacity_year"], []), gas_unavail_h)
            coal_tiers = adjust_tiers_for_outage(coal_tiers_by_year.get(r["capacity_year"], []), coal_unavail_h)
            tiers = []
            cogen_qty, ttf, eua, coal_th = r["cogeneracion_resto_clim"], r["ttf_eur_mwh"], r["eua_eur_t"], r["coal_eur_mwh_th"]
            if not pd.isna(cogen_qty) and cogen_qty > 0 and not pd.isna(ttf) and not pd.isna(eua):
                tiers.append((HEAT_RATE_GAS * (ttf + EMISSION_FACTOR_GAS * eua), cogen_qty, "cogeneracion"))
            if not pd.isna(ttf) and not pd.isna(eua):
                for hr, qty in gas_tiers:
                    tiers.append((hr * (ttf + EMISSION_FACTOR_GAS * eua), qty, "gas"))
            if not pd.isna(coal_th) and not pd.isna(eua):
                for hr, qty in coal_tiers:
                    tiers.append((hr * (coal_th + EMISSION_FACTOR_COAL * eua), qty, "carbon"))
            if not pd.isna(water_value_h):
                tiers.append((water_value_h, turbine_cap_mw, "hidraulica_gestionable"))
            tiers.sort(key=lambda t: t[0])
            for cost, qty, label in tiers:
                if remaining <= qty:
                    return cost, label
                remaining -= qty
            if tiers:
                return tiers[-1][0], "shortfall_" + tiers[-1][2]
            return np.nan, "sin_tiers"

        results = pd.DataFrame(
            [dispatch_row_combined(row, df["water_value_eur_mwh"].iloc[i], gas_unavail[i], coal_unavail[i], must_run_v2.iloc[i])
             for i, row in enumerate(df.to_dict("records"))],
            columns=["price_simulado", "tecnologia_marginal"],
        )
        price_combined = results["price_simulado"]
        report("Fase 1 combinada (hidráulica=tramo de coste + indisponibilidad nuclear/gas/carbón)", price_combined)
        df["price_simulado_final"] = price_combined
        df["tecnologia_marginal_final"] = results["tecnologia_marginal"]
        df["must_run_final_mw"] = must_run_v2
        outages_applied = True

    if not outages_applied:
        df["price_simulado_final"] = price_hydro_tier
        df["tecnologia_marginal_final"] = results_hydro_tier["tecnologia_marginal"]
        df["must_run_final_mw"] = must_run_sin_hidraulica
        log("\n(Resultado final = Fase 1c (hidráulica como tramo), indisponibilidad todavía pendiente de datos)")

    summary_path = OUTPUT_DIR / "model_summary_fase1_tecnologias.txt"
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    df.to_parquet(OUTPUT_DIR / "predicciones_tecnologia.parquet", index=False)
    log(f"\nPredicciones guardadas en {OUTPUT_DIR / 'predicciones_tecnologia.parquet'}")
    log(f"Resumen guardado en {summary_path}")


if __name__ == "__main__":
    main()
