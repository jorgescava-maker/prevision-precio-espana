"""
"El mejor modelo posible de precio DA" para Francia — ver DESIGN.md. Reutiliza
directamente la receta ya validada en `precio_da_mejor_modelo` (España):
resolución nativa por timestamp exacto, análogo v5 (`_build_analog_generic`,
importado tal cual), lags/EMA/boundary de precio (`_add_price_lags`, también
importado tal cual), disponibilidad real de nucleares (`_france_nuclear_avail_hourly`,
ya construida en el propio modelo español para su feature de interconexión —
aquí se usa como pieza central, no periférica). Tiene `water_value_eur_mwh`
propio desde Fase 9 (`water_value.py`, DP de C3 reutilizada sobre datos
franceses) — construido y probado, NO adoptado en `model.py::FEATURES_FULL`
en su momento (ver Fase 9 en RESULTS.md, resultado plano/dentro del ruido
SIN un motor de despacho al que agarrarse). Fase 10 (2026-08-31) añade el
motor de despacho propio (`studies/merit_order_francia/`, correlación 0,708
en solitario) como feature ancla `price_simulado_final` (L6, mismo patrón
que España) — reintenta el hallazgo nulo de la Fase 9 ahora que el valor
del agua sí tiene un mecanismo de despacho real.

Uso:
    .venv\\Scripts\\python.exe -m studies.precio_da_francia.build_dataset

Produce studies/precio_da_francia/output/dataset.parquet.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from studies.merit_order.build_dataset import DATA_DIR, COAL_MWH_PER_TONNE, _load_commodities, _add_climatology
from studies.precio_da_mejor_modelo.build_dataset import (
    _build_analog_generic, _add_price_lags, _france_nuclear_avail_hourly,
)
from studies.precio_da_conjunto.simple_country import _add_degree_days, add_cross_country_price_lag
from studies.precio_da_francia.water_value import build_water_value_feature

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(__file__).resolve().parent / "output"


# ---------------------------------------------------------------------------
# Features horarias (D-1-seguras)
# ---------------------------------------------------------------------------

def _load_forecast_d1(con: duckdb.DuckDBPyConnection, log) -> pd.DataFrame:
    demanda = con.execute(
        "SELECT interval_start_utc AS hour_utc, load_forecast_mw AS forecast_demanda_mw "
        "FROM entsoe_load_forecast"
    ).fetchdf()
    gen = con.execute(
        "SELECT interval_start_utc AS hour_utc, psr_type, generation_forecast_mw "
        "FROM entsoe_generation_forecast"
    ).fetchdf()
    gen_wide = gen.pivot_table(index="hour_utc", columns="psr_type", values="generation_forecast_mw", aggfunc="mean")
    for col in ("B16", "B18", "B19"):
        if col not in gen_wide.columns:
            gen_wide[col] = np.nan
    gen_wide = gen_wide.reset_index()
    # B18 (eólica marina) solo tiene capacidad desde 2023-08-06 — fillna(0) es
    # ausencia real de capacidad, no un hueco de datos (DESIGN.md).
    gen_wide["forecast_eolica_mw"] = gen_wide["B19"].fillna(0) + gen_wide["B18"].fillna(0)
    gen_wide["forecast_solar_mw"] = gen_wide["B16"]

    df = demanda.merge(gen_wide[["hour_utc", "forecast_eolica_mw", "forecast_solar_mw"]], on="hour_utc", how="inner")
    df["hour_utc"] = pd.to_datetime(df["hour_utc"])
    log(f"  Previsión D-1 cargada: {len(df):,} horas, {df['hour_utc'].min()} -> {df['hour_utc'].max()}")
    return df


def _unavailable_mw_thermal(con: duckdb.DuckDBPyConnection, psr_types: list[str], hour_index: pd.DatetimeIndex) -> pd.Series:
    """Mismo criterio que `hourly_unavailable_mw` de merit_order (máximo entre
    eventos que solapan de la MISMA central, sumado luego entre centrales
    distintas) pero sobre `thermal_outage_events` de Francia en vez de
    `generation_outage_events` de España — tabla distinta, misma lógica."""
    placeholders = ", ".join(f"'{p}'" for p in psr_types)
    events = con.execute(
        f"SELECT unit_resource_id, event_start_utc, event_end_utc, unavailable_mw FROM thermal_outage_events "
        f"WHERE psr_type IN ({placeholders}) AND unavailable_mw IS NOT NULL"
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


def _add_real_availability(df: pd.DataFrame, log) -> pd.DataFrame:
    con = duckdb.connect(str(DATA_DIR / "france.duckdb"), read_only=True)
    try:
        capacity = con.execute(
            "SELECT capacity_year, psr_type, capacity_mw FROM entsoe_installed_capacity "
            "WHERE psr_type IN ('B14','B04','B05')"
        ).fetchdf()
        hour_index = pd.DatetimeIndex(df["hour_utc"])
        gas_unavail = _unavailable_mw_thermal(con, ["B04"], hour_index).to_numpy()
        coal_unavail = _unavailable_mw_thermal(con, ["B05", "B06"], hour_index).to_numpy()
    finally:
        con.close()

    df["capacity_year"] = df["hour_utc"].dt.year
    cap_nuclear = capacity[capacity["psr_type"] == "B14"].set_index("capacity_year")["capacity_mw"]
    cap_gas = capacity[capacity["psr_type"] == "B04"].set_index("capacity_year")["capacity_mw"]
    # B05 (hulla/carbón) es pequeño en Francia (~1.800 MW) frente a B06
    # (fuel-oil, ~2.600-3.000 MW) — se agrupan capacidad Y indisponibilidad
    # juntas (mismo criterio que _unavailable_mw_thermal ya suma B05+B06),
    # si no la resta queda mal emparejada y puede dar negativo.
    cap_coal = capacity[capacity["psr_type"].isin(["B05", "B06"])].groupby("capacity_year")["capacity_mw"].sum()

    # Suelo en 0: el carbón+fuel-oil francés es una porción marginal y en
    # declive del mix (centrales casi-mothballed, ~4.400 MW nominales frente
    # a los ~30.000 MW de gas español) — la capacidad instalada oficial
    # (A68) y la capacidad nominal embebida en los eventos de indisponibilidad
    # (A80) no cuadran exactamente para estas unidades pequeñas, lo que puede
    # dar un "disponible" ligeramente negativo sin sentido físico. Se recorta
    # en vez de investigar a fondo una tecnología marginal para este mercado
    # (simplificación declarada, no aplica a nuclear/gas que sí importan).
    df["gas_capacity_avail_mw"] = (df["capacity_year"].map(cap_gas).to_numpy() - gas_unavail).clip(min=0)
    df["coal_capacity_avail_mw"] = (df["capacity_year"].map(cap_coal).to_numpy() - coal_unavail).clip(min=0)

    nuclear_fr = _france_nuclear_avail_hourly(log)
    df = df.merge(nuclear_fr, on="hour_utc", how="left")
    df = df.rename(columns={"nuclear_capacity_fr_avail_mw": "nuclear_capacity_avail_mw"})
    log(f"  Disponibilidad real: nuclear media={df['nuclear_capacity_avail_mw'].mean():.0f} MW, "
        f"gas media={df['gas_capacity_avail_mw'].mean():.0f} MW, térmica(carbón+fuel) media={df['coal_capacity_avail_mw'].mean():.0f} MW")
    return df


MERIT_ORDER_OUTPUT = ROOT / "studies" / "merit_order_francia" / "output"


def _add_merit_order_engine(df: pd.DataFrame, log) -> pd.DataFrame:
    """Fase 10 — ancla mecanicista (L6): precio simulado del motor de
    despacho de `merit_order_francia` (Fase 0, correlación 0,708 en
    solitario) — reutilizado tal cual, sin recalcular nada aquí. Requiere
    haber corrido antes `studies.merit_order_francia.build_dataset`."""
    engine = pd.read_parquet(MERIT_ORDER_OUTPUT / "dataset.parquet")[
        ["hour_utc", "price_simulado", "tecnologia_marginal"]
    ].rename(columns={"price_simulado": "price_simulado_final"})
    df = df.merge(engine, on="hour_utc", how="left")
    log(f"  Motor de despacho (Fase 10): {df['price_simulado_final'].notna().sum():,}/{len(df):,} "
        f"horas con precio simulado")
    return df


def _add_hydro_reservoir(df: pd.DataFrame, log) -> pd.DataFrame:
    """Nivel de embalse en bruto — asof hacia atrás, D-1-seguro por
    construcción (semanal, se conoce antes de cualquier hora posterior).
    El valor del agua (DP) se calcula aparte, ver `water_value_eur_mwh`
    más abajo (Fase 9)."""
    con = duckdb.connect(str(DATA_DIR / "france.duckdb"), read_only=True)
    try:
        filling = con.execute(
            "SELECT interval_start_utc AS week_start, filling_mwh / 1000.0 AS embalse_gwh "
            "FROM hydro_reservoir_filling ORDER BY 1"
        ).fetchdf()
    finally:
        con.close()
    filling["week_start"] = pd.to_datetime(filling["week_start"])
    m = pd.merge_asof(
        df[["hour_utc"]].sort_values("hour_utc"), filling.sort_values("week_start"),
        left_on="hour_utc", right_on="week_start", direction="backward", tolerance=pd.Timedelta(days=21),
    ).sort_index()
    df["embalse_gwh"] = m["embalse_gwh"].to_numpy()
    log(f"  Embalse: {df['embalse_gwh'].notna().sum():,}/{len(df):,} horas con nivel conocido")

    log("  Calculando el valor del agua propio de Francia (DP de C3 reutilizada, candidato de la auditoría)...")
    df["water_value_eur_mwh"], _ = build_water_value_feature(df, log)
    return df


CLIMATOLOGY_PSR_TYPES = {
    "B01": "biomasa_clim_raw",
    "B11": "hidraulica_fluyente_clim_raw",
    "B12": "hidraulica_embalse_clim_raw",
    "B17": "residuos_clim_raw",
}


def _add_generation_climatology(df: pd.DataFrame, log) -> pd.DataFrame:
    """Fase 7 (hallazgo de esta sesión): Francia tiene categorías reales de
    generación (`entsoe_generation_by_type`) sin equivalente de previsión D-1
    publicada — biomasa, hidráulica fluyente (no gestionable, como una
    renovable más), hidráulica de embalse (generación real, distinta del
    NIVEL de embalse ya usado) y residuos. España ya tenía este mismo hueco
    para categorías análogas y lo resolvió con climatología (media móvil de
    4 semanas del mismo día de la semana/hora, desplazada 1 para excluir el
    propio día — D-1-segura por construcción, `_add_climatology` reutilizada
    tal cual). También se añade el bombeo NETO (generación − consumo, B10)
    como climatología — el bombeo es price-driven, no debería predecirse el
    valor del propio día, pero su patrón HABITUAL sí es información D-1."""
    con = duckdb.connect(str(DATA_DIR / "france.duckdb"), read_only=True)
    try:
        gen = con.execute(
            "SELECT interval_start_utc AS hour_utc, psr_type, flow_direction, generation_mw "
            "FROM entsoe_generation_by_type WHERE psr_type IN ('B01','B10','B11','B12','B17')"
        ).fetchdf()
    finally:
        con.close()
    gen["hour_utc"] = pd.to_datetime(gen["hour_utc"])
    gen_h = gen.groupby(["hour_utc", "psr_type", "flow_direction"])["generation_mw"].mean().reset_index()

    wide = {}
    for psr, col in CLIMATOLOGY_PSR_TYPES.items():
        sub = gen_h[(gen_h["psr_type"] == psr) & (gen_h["flow_direction"] == "generation")]
        wide[col] = sub.set_index("hour_utc")["generation_mw"]
    bombeo_gen = gen_h[(gen_h["psr_type"] == "B10") & (gen_h["flow_direction"] == "generation")].set_index("hour_utc")["generation_mw"]
    bombeo_con = gen_h[(gen_h["psr_type"] == "B10") & (gen_h["flow_direction"] == "consumption")].set_index("hour_utc")["generation_mw"]
    wide["bombeo_neto_clim_raw"] = bombeo_gen.reindex(bombeo_gen.index.union(bombeo_con.index)).fillna(0) - \
        bombeo_con.reindex(bombeo_gen.index.union(bombeo_con.index)).fillna(0)

    raw = pd.DataFrame(wide)
    hourly_grid = pd.date_range(raw.index.min(), raw.index.max(), freq="h")
    raw = raw.reindex(hourly_grid).rename_axis("hour_utc").reset_index()

    raw = _add_climatology(raw, list(wide.keys()), hour_col="hour_utc")
    clim_cols = [f"{c}_clim" for c in wide.keys()]
    df = df.merge(raw[["hour_utc"] + clim_cols], on="hour_utc", how="left")
    rename = {f"{k}_clim": k.replace("_raw", "") for k in wide.keys()}
    df = df.rename(columns=rename)
    log(f"  Climatología de generación añadida: {list(rename.values())}")
    return df


def _add_commodities(df: pd.DataFrame, log) -> pd.DataFrame:
    """Coste de combustible con dos días de margen (D-2, L10) — mismo
    criterio que merit_order (España), reutilizando `_load_commodities`
    (agnóstica de país, `commodities.duckdb` es paneuropeo)."""
    con = duckdb.connect(str(DATA_DIR / "commodities.duckdb"), read_only=True)
    try:
        commodities = _load_commodities(con)
    finally:
        con.close()
    df["trade_date_cutoff"] = (df["hour_utc"] - pd.Timedelta(days=2)).dt.normalize()
    df = df.sort_values("trade_date_cutoff")
    commodities = commodities.sort_values("trade_date")
    df = pd.merge_asof(
        df, commodities, left_on="trade_date_cutoff", right_on="trade_date", direction="backward"
    )
    df["coal_eur_mwh_th"] = (df["coal_usd_t"] / df["eur_usd"]) / COAL_MWH_PER_TONNE
    log(f"  Materias primas (D-2): ttf nulos={df['ttf_eur_mwh'].isna().sum()}, "
        f"eua nulos={df['eua_eur_t'].isna().sum()}, coal nulos={df['coal_eur_mwh_th'].isna().sum()}")
    return df.sort_values("hour_utc").reset_index(drop=True)


def _build_hourly_features(log) -> pd.DataFrame:
    con = duckdb.connect(str(DATA_DIR / "france.duckdb"), read_only=True)
    try:
        df = _load_forecast_d1(con, log)
    finally:
        con.close()
    df = _add_real_availability(df, log)
    df = _add_hydro_reservoir(df, log)
    df = _add_generation_climatology(df, log)
    df = _add_commodities(df, log)
    df = _add_merit_order_engine(df, log)
    return df


# ---------------------------------------------------------------------------
# Resolución nativa — mismo patrón que precio_da_mejor_modelo Fase 3
# ---------------------------------------------------------------------------

def _load_native_price(con: duckdb.DuckDBPyConnection, log) -> pd.DataFrame:
    df = con.execute(
        "SELECT interval_start_utc AS period_start_utc, resolution_minutes, price_eur_mwh AS price_real "
        "FROM entsoe_day_ahead_prices ORDER BY 1"
    ).fetchdf()
    df["period_start_utc"] = pd.to_datetime(df["period_start_utc"])
    df["period_in_hour"] = (df["period_start_utc"].dt.minute // 15).astype(int)
    n_hourly = (df["resolution_minutes"] == 60).sum()
    n_quarter = (df["resolution_minutes"] == 15).sum()
    log(f"  Precio nativo cargado: {n_hourly:,} periodos horarios + {n_quarter:,} de 15 min = {len(df):,} filas")
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


def _add_holidays(df: pd.DataFrame) -> pd.DataFrame:
    con = duckdb.connect(str(DATA_DIR / "france.duckdb"), read_only=True)
    try:
        holidays = con.execute(
            "SELECT holiday_date FROM holiday_calendar WHERE location_key = 'france'"
        ).fetchdf()
    finally:
        con.close()
    holiday_dates = set(pd.to_datetime(holidays["holiday_date"]).dt.date)
    df["date"] = df["period_start_utc"].dt.date
    df["is_holiday"] = df["date"].isin(holiday_dates)
    df["is_weekend"] = df["day_of_week"] >= 5
    df["is_dia_no_laborable"] = df["is_holiday"] | df["is_weekend"]
    return df


def build() -> pd.DataFrame:
    lines: list[str] = []

    def log(s: str = "") -> None:
        print(s)
        lines.append(s)

    log("Cargando features horarias D-1-seguras de Francia...")
    hourly_df = _build_hourly_features(log)

    con = duckdb.connect(str(DATA_DIR / "france.duckdb"), read_only=True)
    try:
        native_price = _load_native_price(con, log)
    finally:
        con.close()

    log("Repartiendo features horarias a la resolución nativa de cada periodo...")
    df = _expand_to_native(hourly_df, native_price)

    log("Añadiendo festivos/fin de semana...")
    df = _add_holidays(df)

    log("Construyendo el análogo (demanda + ratio renovable + ttf + embalse, Fase 2)...")
    df["ratio_renovable_periodo"] = (df["forecast_eolica_mw"] + df["forecast_solar_mw"]) / df["forecast_demanda_mw"]
    df = _build_analog_generic(
        df, ["forecast_demanda_mw", "ratio_renovable_periodo", "ttf_eur_mwh", "embalse_gwh"], "analog_price_mean", log,
        "Francia (demanda+ratio+ttf+embalse, Fase 2)", borrow_parent_hour_history=True,
    )
    log("Construyendo el análogo v2 (=Fase 2 + valor del agua propio, Fase 9 — candidato de la auditoría)...")
    df = _build_analog_generic(
        df, ["forecast_demanda_mw", "ratio_renovable_periodo", "ttf_eur_mwh", "water_value_eur_mwh"],
        "analog_price_mean_wv", log, "Francia (demanda+ratio+ttf+water_value, Fase 9)",
        borrow_parent_hour_history=True,
    )

    log("Añadiendo lags de precio por timestamp exacto (reutilizado tal cual)...")
    df = _add_price_lags(df, log)

    log("Añadiendo HDD/CDD reales de ayer (findings.md #106, candidato nuevo)...")
    df = _add_degree_days(df, "france", log)

    log("Añadiendo lag de precio de Alemania (findings.md #106)...")
    df, _ = add_cross_country_price_lag(
        df, "germany.duckdb", "smard_day_ahead_prices", "price_eur_mwh", "interval_start_utc", "de", log
    )

    df = df.sort_values("period_start_utc").reset_index(drop=True)

    OUTPUT_DIR.mkdir(exist_ok=True)
    (OUTPUT_DIR / "build_summary.txt").write_text("\n".join(lines), encoding="utf-8")
    return df


def main() -> None:
    df = build()
    OUTPUT_DIR.mkdir(exist_ok=True)
    out_path = OUTPUT_DIR / "dataset.parquet"
    df.to_parquet(out_path, index=False)
    print(f"\nDataset construido: {len(df):,} periodos, {df['period_start_utc'].min()} -> {df['period_start_utc'].max()}")
    print(f"  De ellos, resolución 15 min: {(df['resolution_minutes']==15).sum():,}  "
          f"horaria: {(df['resolution_minutes']==60).sum():,}")
    print(f"Guardado en {out_path}")
    print(f"\nColumnas: {df.columns.tolist()}")
    print("\nNulos por columna:")
    print(df.isna().sum())


if __name__ == "__main__":
    main()
