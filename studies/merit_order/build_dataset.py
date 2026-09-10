"""
Construye el dataset del simulador de merit order — ver DESIGN.md.

Punto crítico de diseño (revisado 2026-08-30 tras aviso del usuario): el precio
day-ahead de OMIE se fija en la subasta que cierra sobre el mediodía D-1, con la
información disponible EN ESE MOMENTO — no con lo que realmente pasó el día de
entrega. Usar generación real observada (ex-post) para "explicar" ese precio sería
mezclar información posterior al cierre del mercado, el mismo tipo de fuga temporal
que ya se cazó en otros estudios de este proyecto. Este script solo usa, para cada
hora de entrega D:
  - Previsión D-1 real de e·sios (demanda/eólica/solar) — la información oficial
    que el mercado sí tenía antes de cerrar.
  - Capacidad nominal instalada POR CENTRAL real (gas, carbón — entsoe_generation_units,
    Fase 0c) — cifra anual, fija y conocida con mucha antelación, no un dato en
    tiempo real.
  - Una "climatología" (media móvil de las últimas 4 semanas del mismo día de la
    semana y hora, desplazada para excluir el propio día D) para nuclear/hidráulica/
    solar térmica/térmica renovable/cogeneración/importación neta — tecnologías sin
    previsión D-1 publicada en las fuentes cargadas, pero de comportamiento lento y
    predecible día a día. Es una aproximación deliberada, documentada en DESIGN.md,
    NO el dato real de esa hora.
  - Precio de gas/carbón/CO2 con dos días de margen (D-2, no D-1): el cierre TTF/EUA
    de D-1 se publica al final de esa jornada, DESPUÉS de que la subasta eléctrica
    para D ya haya cerrado (mediodía D-1) — usar D-1 sería también fuga.

Uso:
    .venv\\Scripts\\python.exe -m studies.merit_order.build_dataset
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"

# Mismas constantes ya validadas en E2/C2 (studies/e2_fuel_switching/build_dataset.py)
HEAT_RATE_GAS = 1.818
EMISSION_FACTOR_GAS = 0.202
COAL_MWH_PER_TONNE = 6.978
HEAT_RATE_COAL = 2.632
EMISSION_FACTOR_COAL = 0.340

# Curva de oferta escalonada dentro de gas/carbón (añadido tras Fase 0, ver
# RESULTS.md "Fase 0b"): la Fase 0 original trataba todo el gas como UNA central
# con UN heat rate medio — colapsaba exactamente la parte de la curva de oferta
# (el orden de mérito DENTRO de una misma tecnología) que genera la variación de
# precio intradiaria real. Un primer intento con tramos SINTÉTICOS de igual
# tamaño (5 de gas, 3 de carbón, repartiendo la capacidad total a partes
# iguales) mejoró poco y se SATURÓ enseguida (probado también con 20/10 tramos:
# prácticamente ningún cambio) — la granularidad no era el cuello de botella.
# Fase 0c (2026-08-30, findings.md #85 addendum) sustituye eso por el registro
# REAL de centrales de `entsoe_generation_units` (documentType A71 de ENTSO-E,
# 51 centrales de gas / hasta 8 de carbón con su capacidad nominal real,
# 134-859 MW) — sigue sin haber una fuente pública de la eficiencia POR
# CENTRAL, así que el heat rate se asigna por rango de tamaño (más grande =
# más eficiente, un supuesto de ingeniería razonable pero no verificado central
# a central: los bloques de ciclo combinado más grandes de España se
# construyeron en el boom 2002-2012 con turbinas más modernas que las unidades
# pequeñas/antiguas) dentro del mismo rango que la versión sintética.
GAS_HEAT_RATE_RANGE = (1.65, 2.05)   # ~49%-61% de eficiencia
COAL_HEAT_RATE_RANGE = (2.4, 2.9)    # ~34%-42% de eficiencia

# Categorías sin previsión D-1 publicada, tratadas por climatología (ver docstring).
CLIMATOLOGY_CATEGORIES = [
    "nuclear",
    "hidraulica",
    "solar_termica",
    "termica_renovable",
    "cogeneracion_resto",
]
CLIMATOLOGY_WEEKS = 4


def _add_climatology(df: pd.DataFrame, value_cols: list[str], hour_col: str = "hour_utc") -> pd.DataFrame:
    """Media móvil de las últimas CLIMATOLOGY_WEEKS observaciones del mismo
    (día de la semana, hora del día), desplazada 1 para excluir el propio día —
    proxy de "lo que un participante del mercado podría estimar con datos hasta
    D-1", sin usar el valor real del propio día D."""
    df = df.sort_values(hour_col).reset_index(drop=True)
    dow = df[hour_col].dt.dayofweek
    hod = df[hour_col].dt.hour
    for col in value_cols:
        df[f"{col}_clim"] = (
            df.groupby([dow, hod])[col]
            .transform(lambda s: s.shift(1).rolling(CLIMATOLOGY_WEEKS, min_periods=1).mean())
        )
    return df


def _load_forecast_d1(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    query = """
        SELECT
            interval_start_utc AS hour_utc,
            max(CASE WHEN category_key = 'demanda' THEN forecast_mw END) AS forecast_demanda_mw,
            max(CASE WHEN category_key = 'eolica' THEN forecast_mw END) AS forecast_eolica_mw,
            max(CASE WHEN category_key = 'solar_fotovoltaica' THEN forecast_mw END) AS forecast_solar_mw
        FROM esios_forecast
        GROUP BY 1
        ORDER BY 1
    """
    return con.execute(query).fetchdf()


def _load_realized_hourly(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    cats = ", ".join(f"'{c}'" for c in CLIMATOLOGY_CATEGORIES)
    query = f"""
        SELECT
            date_trunc('hour', interval_start_utc) AS hour_utc,
            category_key,
            avg(value_mw) AS value_mw
        FROM esios_generation_by_type
        WHERE category_key IN ({cats})
        GROUP BY 1, 2
    """
    long_df = con.execute(query).fetchdf()
    wide = long_df.pivot(index="hour_utc", columns="category_key", values="value_mw").reset_index()
    return wide


def _load_price_hourly(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    query = """
        SELECT date_trunc('hour', delivery_start_utc) AS hour_utc, avg(price_eur_mwh_es) AS price_real
        FROM omie_spot_prices GROUP BY 1 ORDER BY 1
    """
    return con.execute(query).fetchdf()


def _load_net_import(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    query = """
        SELECT
            date_trunc('hour', interval_start_utc) AS hour_utc,
            sum(CASE WHEN to_area = 'spain' THEN flow_mw ELSE 0 END)
                - sum(CASE WHEN from_area = 'spain' THEN flow_mw ELSE 0 END) AS net_import_mw
        FROM cross_border_flows
        WHERE border_key IN ('ES-PT', 'ES-FR')
        GROUP BY 1
    """
    return con.execute(query).fetchdf()


def _load_commodities(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    commodities = con.execute(
        """
        SELECT trade_date,
               max(CASE WHEN commodity_key = 'ttf_gas' THEN close_price END) AS ttf_eur_mwh,
               max(CASE WHEN commodity_key = 'eua_carbon' THEN close_price END) AS eua_eur_t,
               max(CASE WHEN commodity_key = 'api2_coal' THEN close_price END) AS coal_usd_t
        FROM commodity_prices GROUP BY 1
        """
    ).fetchdf()
    fx = con.execute(
        "SELECT trade_date, close_price AS eur_usd FROM fx_rates WHERE pair_key = 'eur_usd'"
    ).fetchdf()
    df = commodities.merge(fx, on="trade_date", how="left").sort_values("trade_date").reset_index(drop=True)

    # Huecos puntuales por columna (p.ej. un día con EUA pero sin TTF, o viceversa)
    # no deben producir NaN en un merge_asof por fecha exacta más cercana — se
    # reindexa a calendario diario continuo y se arrastra el último precio válido
    # de CADA columna independientemente (límite 5 días: cubre fines de
    # semana/festivos puntuales sin volver "real" el silencio de `api2_coal`
    # tras finales de 2025, que debe seguir siendo NaN — ver DESIGN.md §2), para
    # que el asof-join de más abajo encuentre el último cierre real de cada
    # serie, no el de la fila exacta.
    full_range = pd.date_range(df["trade_date"].min(), df["trade_date"].max(), freq="D")
    df = df.set_index("trade_date").reindex(full_range).ffill(limit=5).rename_axis("trade_date").reset_index()
    return df


def _load_generation_units(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Registro real de centrales (etl/sources/entsoe_generation_units.py,
    documentType A71 de ENTSO-E) — ver comentario junto a GAS_HEAT_RATE_RANGE."""
    return con.execute(
        "SELECT capacity_year, psr_type, capacity_mw FROM entsoe_generation_units WHERE psr_type IN ('B04', 'B05')"
    ).fetchdf()


def _build_unit_tiers(units: pd.DataFrame, psr_type: str, heat_rate_range: tuple[float, float]) -> dict[int, list[tuple[float, float]]]:
    """{año: [(heat_rate, capacity_mw), ...]} — una entrada por central REAL de
    ese año y tecnología, ordenada de mayor a menor capacidad (más grande =
    heat rate más bajo = más eficiente, supuesto declarado en el comentario
    junto a GAS_HEAT_RATE_RANGE), con el heat rate repartido linealmente en el
    rango dado según esa posición."""
    tiers_by_year: dict[int, list[tuple[float, float]]] = {}
    sub = units[units["psr_type"] == psr_type]
    for year, grp in sub.groupby("capacity_year"):
        grp = grp.sort_values("capacity_mw", ascending=False)
        n = len(grp)
        heat_rates = np.linspace(heat_rate_range[0], heat_rate_range[1], n) if n else []
        tiers_by_year[int(year)] = list(zip(heat_rates, grp["capacity_mw"].tolist()))
    return tiers_by_year


def _build_tiers(cogen_qty_mw, capacity_year, ttf, eua, coal_eur_mwh_th, gas_tiers_by_year, coal_tiers_by_year):
    """Lista de (coste, cantidad, etiqueta) — cogeneración como un único tramo
    (aproximación ya declarada en DESIGN.md §3), gas y carbón repartidos por
    central real (ver GAS_HEAT_RATE_RANGE / _build_unit_tiers)."""
    tiers = []
    if not pd.isna(cogen_qty_mw) and cogen_qty_mw > 0 and not pd.isna(ttf) and not pd.isna(eua):
        cost_cogen = HEAT_RATE_GAS * (ttf + EMISSION_FACTOR_GAS * eua)
        tiers.append((cost_cogen, cogen_qty_mw, "cogeneracion"))
    if not pd.isna(ttf) and not pd.isna(eua):
        for hr, qty in gas_tiers_by_year.get(capacity_year, []):
            tiers.append((hr * (ttf + EMISSION_FACTOR_GAS * eua), qty, "gas"))
    if not pd.isna(coal_eur_mwh_th) and not pd.isna(eua):
        for hr, qty in coal_tiers_by_year.get(capacity_year, []):
            tiers.append((hr * (coal_eur_mwh_th + EMISSION_FACTOR_COAL * eua), qty, "carbon"))
    return tiers


def _dispatch_row(demand_to_cover, must_run_mw, cogen_qty_mw, capacity_year, ttf, eua,
                   coal_eur_mwh_th, gas_tiers_by_year, coal_tiers_by_year):
    remaining = demand_to_cover - must_run_mw
    if pd.isna(remaining):
        return np.nan, "sin_datos"
    if remaining <= 0:
        return 0.0, "renovable_nuclear_hidraulica"

    tiers = _build_tiers(cogen_qty_mw, capacity_year, ttf, eua, coal_eur_mwh_th, gas_tiers_by_year, coal_tiers_by_year)
    tiers.sort(key=lambda t: t[0])

    for cost, qty, label in tiers:
        if remaining <= qty:
            return cost, label
        remaining -= qty

    if tiers:
        return tiers[-1][0], "shortfall_" + tiers[-1][2]
    return np.nan, "sin_tiers"


def build() -> pd.DataFrame:
    con = duckdb.connect(str(DATA_DIR / "spain.duckdb"), read_only=True)
    try:
        forecast = _load_forecast_d1(con)
        realized = _load_realized_hourly(con)
        price = _load_price_hourly(con)
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

    realized = realized.merge(net_import, on="hour_utc", how="left")
    realized = _add_climatology(realized, CLIMATOLOGY_CATEGORIES + ["net_import_mw"])

    df = forecast.merge(price, on="hour_utc", how="inner")
    df = df.merge(
        realized[["hour_utc"] + [f"{c}_clim" for c in CLIMATOLOGY_CATEGORIES] + ["net_import_mw_clim"]],
        on="hour_utc",
        how="left",
    )

    # Coste de combustible con dos días de margen (D-2), no el cierre de D-1 —
    # ver docstring del módulo. merge_asof hacia atrás: para cada hora de entrega,
    # usa el último precio de cierre disponible EN O ANTES de (fecha_entrega - 2 días).
    df["trade_date_cutoff"] = (df["hour_utc"] - pd.Timedelta(days=2)).dt.normalize()
    df = df.sort_values("trade_date_cutoff")
    commodities = commodities.sort_values("trade_date")
    df = pd.merge_asof(
        df, commodities, left_on="trade_date_cutoff", right_on="trade_date", direction="backward"
    )

    # Coste medio informativo (ya no es lo que usa el despacho, ver _build_tiers —
    # se conserva solo para comparar/depurar contra la Fase 0 original).
    df["cost_gas_eur_mwh_e"] = HEAT_RATE_GAS * (df["ttf_eur_mwh"] + EMISSION_FACTOR_GAS * df["eua_eur_t"])
    df["coal_eur_mwh_th"] = (df["coal_usd_t"] / df["eur_usd"]) / COAL_MWH_PER_TONNE
    df["cost_coal_eur_mwh_e"] = HEAT_RATE_COAL * (df["coal_eur_mwh_th"] + EMISSION_FACTOR_COAL * df["eua_eur_t"])

    # Capacidad nominal por central real (anual, conocida con antelación).
    df["capacity_year"] = df["hour_utc"].dt.year
    gas_tiers_by_year = _build_unit_tiers(units, "B04", GAS_HEAT_RATE_RANGE)
    coal_tiers_by_year = _build_unit_tiers(units, "B05", COAL_HEAT_RATE_RANGE)

    df["must_run_mw"] = (
        df["forecast_eolica_mw"].fillna(0)
        + df["forecast_solar_mw"].fillna(0)
        + df["nuclear_clim"].fillna(0)
        + df["hidraulica_clim"].fillna(0)
        + df["solar_termica_clim"].fillna(0)
        + df["termica_renovable_clim"].fillna(0)
    )
    df["demand_to_cover_mw"] = df["forecast_demanda_mw"] - df["net_import_mw_clim"].fillna(0)

    results = df.apply(
        lambda r: _dispatch_row(
            r["demand_to_cover_mw"], r["must_run_mw"], r["cogeneracion_resto_clim"],
            r["capacity_year"], r["ttf_eur_mwh"], r["eua_eur_t"], r["coal_eur_mwh_th"],
            gas_tiers_by_year, coal_tiers_by_year,
        ),
        axis=1,
        result_type="expand",
    )
    df["price_simulado"], df["tecnologia_marginal"] = results[0], results[1]

    df = df[df["hour_utc"] >= "2023-01-01"].reset_index(drop=True)
    return df


def main() -> None:
    df = build()
    OUTPUT_DIR.mkdir(exist_ok=True)
    out_path = OUTPUT_DIR / "dataset.parquet"
    df.to_parquet(out_path, index=False)

    print(f"Dataset construido: {len(df):,} horas, {df['hour_utc'].min()} -> {df['hour_utc'].max()}")
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
              "cost_gas_eur_mwh_e", "cost_coal_eur_mwh_e", "price_simulado"]].isna().sum())


if __name__ == "__main__":
    main()
