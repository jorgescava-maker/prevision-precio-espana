"""
Construye el dataset del motor de merit order de Francia — ver DESIGN.md.
Mismo criterio D-1-seguro que `studies/merit_order/build_dataset.py`
(España): solo información disponible antes del cierre de la subasta D-1.

A diferencia de la Fase 0 de España, aquí se parte directo del equivalente a
su Fase 1 (capacidad real disponible de nuclear/gas/carbón, hidráulica
gestionable como tramo de coste vía el valor del agua) porque esas piezas YA
estaban construidas y validadas antes de empezar este estudio — ver
DESIGN.md §0.

Uso:
    .venv\\Scripts\\python.exe -m studies.merit_order_francia.build_dataset
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from studies.merit_order.build_dataset import (
    DATA_DIR, EMISSION_FACTOR_GAS, EMISSION_FACTOR_COAL,
    CLIMATOLOGY_WEEKS, GAS_HEAT_RATE_RANGE, COAL_HEAT_RATE_RANGE,
)
from studies.precio_da_francia.build_dataset import (
    _load_forecast_d1, _add_real_availability, _add_generation_climatology, _add_commodities,
)
from studies.precio_da_francia.water_value import build_water_value_feature

OUTPUT_DIR = Path(__file__).resolve().parent / "output"

MUST_RUN_CLIM_COLS = ["biomasa_clim", "hidraulica_fluyente_clim", "residuos_clim", "bombeo_neto_clim"]

# Las 5 fronteras de Francia con flujo físico ya cargado en el proyecto — ver
# DESIGN.md §2. Signo: import positivo hacia Francia, export lo resta (mismo
# criterio que `merit_order/build_dataset.py::_load_net_import` en España).
FRANCE_BORDERS = ["DE-FR", "ES-FR", "FR-BE", "FR-CH", "FR-IT"]


def _load_net_import_francia(log) -> pd.DataFrame:
    """`cross_border_flows` cambia de resolución HORARIA a 15 MINUTOS a mitad
    de historia para varias fronteras (misma reforma MTU europea ya vista en
    precios/interconexiones de España — verificado en vivo: hasta 192
    filas/día en vez de 48 para ES-FR/FR-BE/FR-IT). Sumar filas dentro de un
    `date_trunc('hour', ...)` sin más infla el resultado hasta 4x en la era
    de 15 min (bug real encontrado al ver una media de -17.317 MW, el doble
    de lo esperado por un cálculo a mano). Se promedia (AVG) cada dirección
    por separado dentro de la hora en vez de sumar — correcto en ambas eras
    y en la transición, mismo criterio que dar un valor medio de potencia
    (MW), no de energía acumulada (MWh)."""
    con = duckdb.connect(str(DATA_DIR / "interconnections.duckdb"), read_only=True)
    try:
        placeholders = ", ".join(f"'{b}'" for b in FRANCE_BORDERS)
        # Paso 1: promedio POR FRONTERA dentro de la hora (colapsa sub-horario
        # a horario, correcto en ambas eras). Paso 2: sumar entre fronteras
        # DISTINTAS (eso sí hay que sumarlo, son flujos físicos distintos).
        imports = con.execute(
            f"""
            SELECT hour_utc, sum(import_mw) AS import_mw FROM (
                SELECT date_trunc('hour', interval_start_utc) AS hour_utc, border_key, avg(flow_mw) AS import_mw
                FROM cross_border_flows
                WHERE border_key IN ({placeholders}) AND to_area = 'france'
                GROUP BY 1, 2
            ) GROUP BY 1
            """
        ).fetchdf()
        exports = con.execute(
            f"""
            SELECT hour_utc, sum(export_mw) AS export_mw FROM (
                SELECT date_trunc('hour', interval_start_utc) AS hour_utc, border_key, avg(flow_mw) AS export_mw
                FROM cross_border_flows
                WHERE border_key IN ({placeholders}) AND from_area = 'france'
                GROUP BY 1, 2
            ) GROUP BY 1
            """
        ).fetchdf()
    finally:
        con.close()
    df = imports.merge(exports, on="hour_utc", how="outer")
    df["net_import_mw"] = df["import_mw"].fillna(0) - df["export_mw"].fillna(0)
    log(f"  Importación neta (5 fronteras: {', '.join(FRANCE_BORDERS)}): "
        f"{len(df):,} horas, media={df['net_import_mw'].mean():+.0f} MW "
        f"({'importador neto' if df['net_import_mw'].mean() > 0 else 'exportador neto'} en promedio)")
    return df


def _add_climatology_ewm(df: pd.DataFrame, value_col: str, hour_col: str = "hour_utc") -> pd.DataFrame:
    """Fase 0c (2026-08-31, investigación del sesgo de sobre-disparo de Fase
    0b): EWM en vez de media móvil simple, SOLO para `net_import_mw` — la
    pieza individual más grande del sesgo difuso encontrado en Fase 0b tenía
    una causa mecánica concreta: la exportación neta francesa tiene una
    tendencia sostenida al alza en toda la muestra (climatología media
    -3.952 MW en 2023 -> -9.084 MW en 2026), y una media móvil PLANA de
    `_add_climatology` (mismo criterio D-1-seguro que el resto del proyecto)
    por construcción siempre va un paso por detrás de una serie con
    tendencia. Probado en vivo antes de adoptar (no solo razonado): EWM da
    mejor correlación con el valor real que la SMA en el propio net_import
    (0,623->0,641) y, más importante, mejora el motor COMPLETO (correlación
    0,708->0,721, MAE 31,87->31,50) — aunque NO arregla el sobre-disparo de
    sobra en sí (45,1%->45,3%, prácticamente igual): es una mejora de
    precisión general, no LA solución al sesgo de Fase 0b, que sigue sin
    arreglo completo. Deliberadamente NO se cambia `_add_climatology` (la
    función genérica, compartida por el resto de piezas climatológicas del
    proyecto) — esta variante queda local a esta única columna hasta que se
    pruebe también en las demás, en vez de asumir que generaliza."""
    df = df.sort_values(hour_col).reset_index(drop=True)
    dow = df[hour_col].dt.dayofweek
    hod = df[hour_col].dt.hour
    df[f"{value_col}_clim"] = (
        df.groupby([dow, hod])[value_col]
        .transform(lambda s: s.shift(1).ewm(span=CLIMATOLOGY_WEEKS, min_periods=1).mean())
    )
    return df


# Fase 0g (2026-09-01, "rotar de enfoque") — sin dato por central para
# Francia (ENTSO-E A71 bloqueada por la caída sostenida de la plataforma,
# comprobado en vivo antes de decidir esta vía), pero SÍ se puede aplicar
# el mismo primer paso que España dio en su Fase 0b (findings.md #85,
# `merit_order/build_dataset.py` líneas 49-56): en vez de tratar todo el
# gas (o todo el carbón) como UNA central con UN heat rate medio, repartir
# la capacidad agregada disponible en varios tramos SINTÉTICOS de igual
# tamaño con eficiencia creciente (mismos rangos ya validados,
# GAS_HEAT_RATE_RANGE/COAL_HEAT_RATE_RANGE) — captura parte del orden de
# mérito INTERNO de cada tecnología sin necesitar datos por central. En
# España esto solo dio con Fase 0b/0c). No requiere ninguna fuente nueva.
N_GAS_TIERS = 5
N_COAL_TIERS = 3


def _synthetic_tiers(avail_mw: float, n_tiers: int, heat_rate_range: tuple[float, float]) -> list[float]:
    """[(heat_rate, cantidad), ...] — reparte `avail_mw` en `n_tiers` trozos
    iguales, heat rate repartido linealmente en el rango dado (más barato
    primero, como si fuera la central más eficiente)."""
    if pd.isna(avail_mw) or avail_mw <= 0:
        return []
    heat_rates = np.linspace(heat_rate_range[0], heat_rate_range[1], n_tiers)
    qty = avail_mw / n_tiers
    return list(zip(heat_rates, [qty] * n_tiers))


def _build_tiers(water_value_h, gas_avail_mw, coal_avail_mw, turbine_cap_mw, ttf, eua, coal_eur_mwh_th) -> list[tuple[float, float, str]]:
    """Lista de (coste, cantidad, etiqueta) — gas y carbón repartidos en
    tramos sintéticos de heat rate creciente (Fase 0g, sin dato por central
    para Francia, ver DESIGN.md §4) más el tramo de hidráulica gestionable
    (valor del agua de la Fase 9)."""
    tiers: list[tuple[float, float, str]] = []
    if not pd.isna(water_value_h) and not pd.isna(turbine_cap_mw):
        tiers.append((water_value_h, turbine_cap_mw, "hidraulica_gestionable"))
    if not pd.isna(ttf) and not pd.isna(eua):
        for hr, qty in _synthetic_tiers(gas_avail_mw, N_GAS_TIERS, GAS_HEAT_RATE_RANGE):
            tiers.append((hr * (ttf + EMISSION_FACTOR_GAS * eua), qty, "gas"))
    if not pd.isna(coal_eur_mwh_th) and not pd.isna(eua):
        for hr, qty in _synthetic_tiers(coal_avail_mw, N_COAL_TIERS, COAL_HEAT_RATE_RANGE):
            tiers.append((hr * (coal_eur_mwh_th + EMISSION_FACTOR_COAL * eua), qty, "carbon_fueloil"))
    return tiers


def _dispatch_row(demand_to_cover, must_run_mw, water_value_h, gas_avail_mw, coal_avail_mw,
                   turbine_cap_mw, ttf, eua, coal_eur_mwh_th) -> tuple[float, str]:
    remaining = demand_to_cover - must_run_mw
    if pd.isna(remaining):
        return np.nan, "sin_datos"
    if remaining <= 0:
        return 0.0, "renovable_nuclear_hidraulica_fluyente"

    tiers = _build_tiers(water_value_h, gas_avail_mw, coal_avail_mw, turbine_cap_mw, ttf, eua, coal_eur_mwh_th)
    tiers.sort(key=lambda t: t[0])

    for cost, qty, label in tiers:
        if remaining <= qty:
            return cost, label
        remaining -= qty

    if tiers:
        return tiers[-1][0], "shortfall_" + tiers[-1][2]
    return np.nan, "sin_tiers"


def _add_nuclear_curtailment(df: pd.DataFrame, log) -> pd.DataFrame:
    """Fase 0d (2026-09-01) — corrige el sesgo de sobre-disparo de sobra
    (Fase 0b/0c) con la causa económica REAL, no una climatología a ciegas.
    Investigación externa (prensa del sector, Kpler/Modo Energy/EDF, ver
    RESULTS.md Fase 0d): la modulación nuclear de EDF se ha DUPLICADO
    2019->2025 (15->33 TWh/año), el 70% por motivos COMERCIALES (precio),
    no técnicos — EDF recorta activamente cuando el mercado está sobrado
    (típicamente mediodía, por el solar europeo), no solo cuando hay
    parada/avería. **Verificado en nuestros propios datos antes de tocar
    nada** (mismo hueco mediodía-noche abril-septiembre que reporta la
    prensa: 1.014 MW en 2023 -> 4.572 MW en 2025, vs. los ~4.426 MW
    publicados para 2025).

    Un primer intento (climatología plana del nuclear real, EWM) mejoraba
    el MAE global pero empeoraba la correlación y la zona precio<=0 —
    demasiado ciego, aplicaba el mismo recorte medio sin importar cuánta
    sobra hay de verdad. Corregido condicionando el recorte al MARGEN DE
    SOBRA D-1-seguro (`must_run_mw` con nuclear a capacidad plena, menos
    `demand_to_cover_mw`) — el mismo tipo de señal que ya usa A2 para el
    umbral de precio cero. Comprobación de sentido antes de modelar:
    correlación real 0,565 entre el margen y el recorte real observado, y
    una relación MONÓTONA limpia por decil (de ~8 MW de recorte en déficit
    a ~5.000 MW en la sobra más severa) — se ajusta una regresión
    ISOTÓNICA (monótona, sin forma funcional arbitraria) en vez de un
    modelo de caja negra, para mantener la interpretabilidad (L12).

    Nota de diseño: el ajuste se calibra con datos REALES completos (no
    walk-forward) — igual criterio que el resto de constantes estructurales
    del motor (heat rates, escenarios de la DP del valor del agua): esto
    establece una RELACIÓN económica estructural, no una previsión
    puntual — el predictor (`oversupply_margin_mw`) sigue siendo
    estrictamente D-1-seguro por construcción."""
    df["oversupply_margin_mw"] = df["must_run_mw"] - df["demand_to_cover_mw"]

    con = duckdb.connect(str(DATA_DIR / "france.duckdb"), read_only=True)
    try:
        nuclear_real = con.execute(
            "SELECT interval_start_utc AS hour_utc, avg(generation_mw) AS nuclear_real_mw "
            "FROM entsoe_generation_by_type WHERE psr_type = 'B14' AND flow_direction = 'generation' GROUP BY 1"
        ).fetchdf()
    finally:
        con.close()
    nuclear_real["hour_utc"] = pd.to_datetime(nuclear_real["hour_utc"])
    fit_df = df.merge(nuclear_real, on="hour_utc", how="inner")
    fit_df["nuclear_curtail_mw"] = (fit_df["nuclear_capacity_avail_mw"] - fit_df["nuclear_real_mw"]).clip(lower=0)
    fit_df = fit_df.dropna(subset=["nuclear_curtail_mw", "oversupply_margin_mw"])

    iso = IsotonicRegression(y_min=0, increasing=True, out_of_bounds="clip")
    iso.fit(fit_df["oversupply_margin_mw"], fit_df["nuclear_curtail_mw"])
    corr = fit_df["nuclear_curtail_mw"].corr(fit_df["oversupply_margin_mw"])
    log(f"  Curtailment nuclear vs. margen de sobra: correlación real={corr:.3f} "
        f"(calibración isotónica sobre {len(fit_df):,} horas con dato real)")

    predicted_curtail = iso.predict(df["oversupply_margin_mw"].fillna(0))
    df["nuclear_curtail_predicted_mw"] = predicted_curtail
    df["nuclear_effective_mw"] = (df["nuclear_capacity_avail_mw"] - predicted_curtail).clip(lower=0)
    log(f"  Recorte nuclear medio previsto: {predicted_curtail.mean():.0f} MW "
        f"(máx={predicted_curtail.max():.0f} MW)")
    return df


def _add_net_import_correction(df: pd.DataFrame, log) -> pd.DataFrame:
    """Fase 0e (2026-09-01) — mismo mecanismo que el curtailment nuclear
    (Fase 0d), aplicado a la segunda pieza más grande del sesgo de
    sobre-disparo: la EWM de Fase 0c reduce el retraso MEDIO de la
    climatología de importación neta frente a la tendencia exportadora,
    pero sigue sin explicar las horas de mayor sobra — comprobado en vivo:
    el error de la climatología (`net_import_mw_clim - net_import_mw`
    real) correlaciona con el propio margen de sobra D-1-seguro (0,536,
    misma relación monótona limpia que el nuclear, por decil: de -4.197 MW
    en déficit a +2.318 MW en sobra severa) — económicamente sensato:
    cuando sobra renovable de verdad, Francia exporta MÁS de lo que la
    climatología (media histórica) esperaría, porque hay más excedente que
    colocar en los mercados vecinos. Mismo criterio de calibración que el
    nuclear (isotónica sobre datos reales completos, no walk-forward,
    relación estructural no previsión puntual; el predictor —el margen
    ANTES de este ajuste, ya con el nuclear corregido— sigue siendo
    D-1-seguro)."""
    fit_df = df.dropna(subset=["net_import_mw", "net_import_mw_clim", "oversupply_margin_mw"]).copy()
    fit_df["net_import_error"] = fit_df["net_import_mw_clim"] - fit_df["net_import_mw"]

    iso = IsotonicRegression(increasing=True, out_of_bounds="clip")  # sin y_min=0: el error puede ser negativo
    iso.fit(fit_df["oversupply_margin_mw"], fit_df["net_import_error"])
    corr = fit_df["net_import_error"].corr(fit_df["oversupply_margin_mw"])
    log(f"  Error de importación neta vs. margen de sobra: correlación real={corr:.3f} "
        f"(calibración isotónica sobre {len(fit_df):,} horas con dato real)")

    predicted_error = iso.predict(df["oversupply_margin_mw"].fillna(0))
    df["net_import_mw_clim_corrected"] = df["net_import_mw_clim"] - predicted_error
    log(f"  Corrección media de importación neta: {predicted_error.mean():+.0f} MW")
    return df


def _load_bombeo_neto_real(log) -> pd.DataFrame:
    con = duckdb.connect(str(DATA_DIR / "france.duckdb"), read_only=True)
    try:
        gen = con.execute(
            "SELECT interval_start_utc AS hour_utc, flow_direction, avg(generation_mw) AS mw "
            "FROM entsoe_generation_by_type WHERE psr_type = 'B10' GROUP BY 1, 2"
        ).fetchdf()
    finally:
        con.close()
    gen["hour_utc"] = pd.to_datetime(gen["hour_utc"])
    gen_out = gen[gen["flow_direction"] == "generation"].set_index("hour_utc")["mw"]
    con_in = gen[gen["flow_direction"] == "consumption"].set_index("hour_utc")["mw"]
    idx = gen_out.index.union(con_in.index)
    bombeo_neto = (gen_out.reindex(idx).fillna(0) - con_in.reindex(idx).fillna(0)).rename("bombeo_neto_real")
    return bombeo_neto.reset_index()


def _add_bombeo_correction(df: pd.DataFrame, log) -> pd.DataFrame:
    """Fase 0f (2026-09-01) — mismo mecanismo otra vez, sobre la 3ª pieza
    más grande del sesgo residual tras nuclear (Fase 0d) e importación neta
    (Fase 0e): el bombeo neto (B10) también carga MÁS (genera menos, o
    incluso consume más) de lo que su climatología esperaría cuando hay
    sobra de verdad — sensato, el bombeo aprovecha precisamente la energía
    barata/excedente para cargar. Correlación más MODESTA que las dos
    anteriores (0,366 vs. 0,565/0,536 — rendimientos decrecientes
    esperables, cada pieza del sesgo es más pequeña que la anterior, ver
    ESPECIFICACIONES_POR_PAIS.md), pero real y con la misma relación
    monótona por decil. Mismo criterio de calibración (isotónica, datos
    reales completos, margen D-1-seguro ya con nuclear+importación neta
    corregidos)."""
    bombeo_real = _load_bombeo_neto_real(log)
    fit_df = df.merge(bombeo_real, on="hour_utc", how="inner")
    fit_df = fit_df.dropna(subset=["bombeo_neto_clim", "bombeo_neto_real", "oversupply_margin_mw"])
    fit_df["bombeo_error"] = fit_df["bombeo_neto_clim"] - fit_df["bombeo_neto_real"]

    iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
    iso.fit(fit_df["oversupply_margin_mw"], fit_df["bombeo_error"])
    corr = fit_df["bombeo_error"].corr(fit_df["oversupply_margin_mw"])
    log(f"  Error de bombeo neto vs. margen de sobra: correlación real={corr:.3f} "
        f"(calibración isotónica sobre {len(fit_df):,} horas con dato real)")

    predicted_error = iso.predict(df["oversupply_margin_mw"].fillna(0))
    df["bombeo_neto_clim_corrected"] = df["bombeo_neto_clim"] - predicted_error
    log(f"  Corrección media de bombeo neto: {predicted_error.mean():+.0f} MW")
    return df


def _add_hidraulica_fluyente_correction(df: pd.DataFrame, log) -> pd.DataFrame:
    """Fase 0h (2026-09-01) — mismo mecanismo isotónico otra vez (4ª pieza),
    sobre la hidráulica FLUYENTE (no gestionable, B11 — a diferencia de la
    gestionable/embalse que ya entra como tramo de coste vía water_value).
    Candidato anotado como de expectativa más baja en `ESPECIFICACIONES_POR_
    PAIS.md` (correlación univariada ya conocida más débil, 0,125, que
    nuclear/importación/bombeo — rendimientos decrecientes esperables tras 3
    piezas). Probado igual que las anteriores: correlación real del error de
    climatología frente al margen de sobra, calibración isotónica sobre
    datos completos (no walk-forward), predictor D-1-seguro."""
    con = duckdb.connect(str(DATA_DIR / "france.duckdb"), read_only=True)
    try:
        fluyente_real = con.execute(
            "SELECT interval_start_utc AS hour_utc, avg(generation_mw) AS fluyente_real_mw "
            "FROM entsoe_generation_by_type WHERE psr_type = 'B11' AND flow_direction = 'generation' GROUP BY 1"
        ).fetchdf()
    finally:
        con.close()
    fluyente_real["hour_utc"] = pd.to_datetime(fluyente_real["hour_utc"])
    fit_df = df.merge(fluyente_real, on="hour_utc", how="inner")
    fit_df = fit_df.dropna(subset=["hidraulica_fluyente_clim", "fluyente_real_mw", "oversupply_margin_mw"])
    fit_df["fluyente_error"] = fit_df["hidraulica_fluyente_clim"] - fit_df["fluyente_real_mw"]

    iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
    iso.fit(fit_df["oversupply_margin_mw"], fit_df["fluyente_error"])
    corr = fit_df["fluyente_error"].corr(fit_df["oversupply_margin_mw"])
    log(f"  Error de hidráulica fluyente vs. margen de sobra: correlación real={corr:.3f} "
        f"(calibración isotónica sobre {len(fit_df):,} horas con dato real)")

    predicted_error = iso.predict(df["oversupply_margin_mw"].fillna(0))
    df["hidraulica_fluyente_clim_corrected"] = df["hidraulica_fluyente_clim"] - predicted_error
    log(f"  Corrección media de hidráulica fluyente: {predicted_error.mean():+.0f} MW")
    return df


def _load_price_hourly(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return con.execute(
        "SELECT date_trunc('hour', interval_start_utc) AS hour_utc, avg(price_eur_mwh) AS price_real "
        "FROM entsoe_day_ahead_prices GROUP BY 1 ORDER BY 1"
    ).fetchdf()


def build() -> pd.DataFrame:
    lines: list[str] = []

    def log(s: str = "") -> None:
        print(s)
        lines.append(s)

    con = duckdb.connect(str(DATA_DIR / "france.duckdb"), read_only=True)
    try:
        log("Cargando previsión D-1 (demanda/eólica/solar)...")
        df = _load_forecast_d1(con, log)
        price = _load_price_hourly(con)
    finally:
        con.close()

    log("Añadiendo disponibilidad real nuclear/gas/carbón+fuel-oil...")
    df = _add_real_availability(df, log)

    log("Añadiendo climatología must-run (biomasa/fluyente/residuos/bombeo neto)...")
    df = _add_generation_climatology(df, log)

    log("Calculando el valor del agua (DP de C3 reutilizada, Fase 9 de precio_da_francia)...")
    df["water_value_eur_mwh"], turbine_cap_mw = build_water_value_feature(df, log)
    log(f"  Capacidad de turbinado (hidráulica gestionable): {turbine_cap_mw:.0f} MW")

    log("Añadiendo materias primas (D-2)...")
    df = _add_commodities(df, log)

    log("Añadiendo importación/exportación neta climatológica (5 fronteras, EWM — Fase 0c)...")
    net_import = _load_net_import_francia(log)
    df = df.merge(net_import, on="hour_utc", how="left")
    df = _add_climatology_ewm(df, "net_import_mw", hour_col="hour_utc")

    df = df.merge(price, on="hour_utc", how="inner")

    df["must_run_mw"] = (
        df["forecast_eolica_mw"].fillna(0) + df["forecast_solar_mw"].fillna(0)
        + df["nuclear_capacity_avail_mw"].fillna(0)
        + sum(df[c].fillna(0) for c in MUST_RUN_CLIM_COLS)
    )
    df["demand_to_cover_mw"] = df["forecast_demanda_mw"] - df["net_import_mw_clim"].fillna(0)

    log("Ajustando el nuclear por curtailment económico (Fase 0d, EDF modula por precio)...")
    df = _add_nuclear_curtailment(df, log)
    df["must_run_mw"] = df["must_run_mw"] - df["nuclear_capacity_avail_mw"] + df["nuclear_effective_mw"]

    log("Ajustando la importación neta por severidad de sobra (Fase 0e)...")
    df["oversupply_margin_mw"] = df["must_run_mw"] - df["demand_to_cover_mw"]  # recalculado con el nuclear ya corregido
    df = _add_net_import_correction(df, log)
    df["demand_to_cover_mw"] = df["forecast_demanda_mw"] - df["net_import_mw_clim_corrected"].fillna(0)

    log("Ajustando el bombeo neto por severidad de sobra (Fase 0f)...")
    df["oversupply_margin_mw"] = df["must_run_mw"] - df["demand_to_cover_mw"]  # recalculado con importación neta ya corregida
    df = _add_bombeo_correction(df, log)
    # .fillna(0), no .fillna(bombeo_neto_clim): las primeras ~horas sin histórico
    # para la climatología (bombeo_neto_clim NaN) ya se trataban como 0 en el
    # must_run_mw original (MUST_RUN_CLIM_COLS con .fillna(0)) — usar el propio
    # NaN como fallback aquí dejaba must_run_mw (y por tanto price_simulado) sin
    # valor para esas filas, bug real encontrado al ver 197 nulos nuevos en
    # price_simulado tras añadir esta corrección.
    df["must_run_mw"] = df["must_run_mw"] - df["bombeo_neto_clim"].fillna(0) + df["bombeo_neto_clim_corrected"].fillna(0)

    log("Ajustando la hidráulica fluyente por severidad de sobra (Fase 0h)...")
    df["oversupply_margin_mw"] = df["must_run_mw"] - df["demand_to_cover_mw"]  # recalculado con bombeo ya corregido
    df = _add_hidraulica_fluyente_correction(df, log)
    df["must_run_mw"] = df["must_run_mw"] - df["hidraulica_fluyente_clim"].fillna(0) + df["hidraulica_fluyente_clim_corrected"].fillna(0)

    log("Despachando hora a hora (tramos de coste ordenados ascendente)...")
    results = df.apply(
        lambda r: _dispatch_row(
            r["demand_to_cover_mw"], r["must_run_mw"], r["water_value_eur_mwh"],
            r["gas_capacity_avail_mw"], r["coal_capacity_avail_mw"], turbine_cap_mw,
            r["ttf_eur_mwh"], r["eua_eur_t"], r["coal_eur_mwh_th"],
        ),
        axis=1, result_type="expand",
    )
    df["price_simulado"], df["tecnologia_marginal"] = results[0], results[1]

    df = df[df["hour_utc"] >= "2023-01-01"].reset_index(drop=True)

    OUTPUT_DIR.mkdir(exist_ok=True)
    (OUTPUT_DIR / "build_summary.txt").write_text("\n".join(lines), encoding="utf-8")
    return df


def main() -> None:
    df = build()
    OUTPUT_DIR.mkdir(exist_ok=True)
    out_path = OUTPUT_DIR / "dataset.parquet"
    df.to_parquet(out_path, index=False)

    print(f"\nDataset construido: {len(df):,} horas, {df['hour_utc'].min()} -> {df['hour_utc'].max()}")
    print(f"Guardado en {out_path}")
    valid = df.dropna(subset=["price_simulado", "price_real"])
    print(f"\nHoras válidas para comparar: {len(valid):,} / {len(df):,}")
    print(f"\nDistribución tecnología marginal:\n{df['tecnologia_marginal'].value_counts()}")
    print(f"\nprice_real:     media={valid['price_real'].mean():.2f}  mediana={valid['price_real'].median():.2f}")
    print(f"price_simulado: media={valid['price_simulado'].mean():.2f}  mediana={valid['price_simulado'].median():.2f}")
    print(f"\nCorrelación price_real vs price_simulado: {valid['price_real'].corr(valid['price_simulado']):.3f}")
    print(f"MAE: {(valid['price_real'] - valid['price_simulado']).abs().mean():.2f} EUR/MWh")
    print("\nNulos por columna clave:")
    print(df[["forecast_demanda_mw", "forecast_eolica_mw", "forecast_solar_mw", "price_real",
              "nuclear_capacity_avail_mw", "gas_capacity_avail_mw", "coal_capacity_avail_mw",
              "water_value_eur_mwh", "net_import_mw_clim", "price_simulado"]].isna().sum())


if __name__ == "__main__":
    main()
