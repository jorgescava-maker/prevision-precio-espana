"""
Valor del agua propio para Francia — candidato de mayor potencial sin probar
que dejó anotado `ESPECIFICACIONES_POR_PAIS.md` al cerrar la auditoría del
modelo español: Francia tiene `hydro_reservoir_filling` cargado pero nunca
se construyó la superficie de valor del agua específica (usa el nivel de
embalse en bruto, ver `build_dataset.py::_add_hydro_reservoir`). En España
esta misma pieza fue la mejora individual más grande de todo el proyecto
(merit_order Fase 1, +0,144 de correlación, findings.md #88) — no porque el
nivel de embalse no sirva, sino porque la hidráulica gestionable puja en la
curva de oferta a un PRECIO (valor de oportunidad del agua), no despacha una
cantidad fija barata.

Reutiliza el DP de C3 (`solve_dp`/`water_value_surface`, genéricas, sin
tocar su código) sobre una serie semanal propia de Francia — misma
metodología, datos franceses (`france.duckdb`): aportación por balance de
masa (delta de nivel + generación de hidráulica de embalse B12), precio
`entsoe_day_ahead_prices`.

Uso:
    .venv\\Scripts\\python.exe -m studies.precio_da_francia.water_value

Produce studies/precio_da_francia/output/weekly_series_agua.parquet.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from studies.c3_valor_agua.model import load_scenarios, solve_dp, water_value_surface

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def build_weekly_series() -> pd.DataFrame:
    """Mismo criterio que `c3_valor_agua/build_dataset.py::build()`, sobre
    Francia: aportación = delta de llenado + generación (balance de masa),
    sin fuente nueva. `psr_type='B12'` es hidráulica de EMBALSE (gestionable,
    distinta de B11 fluyente/no-gestionable — ya distinguidas en
    `build_dataset.py::CLIMATOLOGY_PSR_TYPES`)."""
    con = duckdb.connect(str(DATA_DIR / "france.duckdb"), read_only=True)
    try:
        filling = con.execute(
            "SELECT interval_start_utc::date AS week, filling_mwh FROM hydro_reservoir_filling ORDER BY 1"
        ).fetchdf()
        gen_weekly = con.execute(
            """
            SELECT date_trunc('week', interval_start_utc) AS week_start,
                   avg(generation_mw) * 24 * 7 / 1000 AS gen_hidraulica_gwh_week
            FROM entsoe_generation_by_type
            WHERE psr_type = 'B12' AND flow_direction = 'generation'
            GROUP BY 1 ORDER BY 1
            """
        ).fetchdf()
        price_weekly = con.execute(
            """
            SELECT date_trunc('week', interval_start_utc) AS week_start, avg(price_eur_mwh) AS price_avg
            FROM entsoe_day_ahead_prices GROUP BY 1 ORDER BY 1
            """
        ).fetchdf()
    finally:
        con.close()

    filling["week"] = pd.to_datetime(filling["week"])
    filling = filling.sort_values("week").reset_index(drop=True)
    filling["filling_gwh"] = filling["filling_mwh"] / 1000
    filling["delta_filling_gwh"] = filling["filling_gwh"].diff()

    def iso_key(s: pd.Series) -> pd.Series:
        iso = s.dt.isocalendar()
        return iso["year"].astype(str) + "-W" + iso["week"].astype(str)

    filling["iso_week"] = iso_key(filling["week"])
    gen_weekly["iso_week"] = iso_key(gen_weekly["week_start"])
    price_weekly["iso_week"] = iso_key(price_weekly["week_start"])

    df = filling.merge(gen_weekly[["iso_week", "gen_hidraulica_gwh_week"]], on="iso_week", how="left")
    df = df.merge(price_weekly[["iso_week", "price_avg"]], on="iso_week", how="left")
    df["inflow_gwh"] = df["delta_filling_gwh"] + df["gen_hidraulica_gwh_week"]

    return df[["week", "filling_gwh", "delta_filling_gwh", "gen_hidraulica_gwh_week", "inflow_gwh", "price_avg"]]


def build_water_value_feature(df: pd.DataFrame, log) -> tuple[pd.Series, float]:
    """Mismo patrón que `merit_order/model.py::build_water_value_feature`,
    pero resolviendo la DP sobre la serie semanal de Francia en vez de la de
    España — reutiliza `solve_dp`/`water_value_surface` de C3 sin tocarlas."""
    log("  Resolviendo la DP de C3 sobre la serie semanal de Francia (mismo código, datos propios)...")
    weekly_path = OUTPUT_DIR / "weekly_series_agua.parquet"
    if not weekly_path.exists():
        build_weekly_series().to_parquet(weekly_path, index=False)
    _, scenarios, max_storage, turbine_cap = load_scenarios(weekly_series_path=weekly_path)
    V_all, policy, storage_grid = solve_dp(scenarios, max_storage, turbine_cap, log)
    wv_surface = water_value_surface(V_all, storage_grid)

    con = duckdb.connect(str(DATA_DIR / "france.duckdb"), read_only=True)
    try:
        filling = con.execute(
            "SELECT interval_start_utc AS week_start, filling_mwh / 1000.0 AS filling_gwh "
            "FROM hydro_reservoir_filling ORDER BY 1"
        ).fetchdf()
    finally:
        con.close()
    filling["week_start"] = pd.to_datetime(filling["week_start"])
    m = pd.merge_asof(
        df[["hour_utc"]].sort_values("hour_utc"), filling.sort_values("week_start"),
        left_on="hour_utc", right_on="week_start", direction="backward", tolerance=pd.Timedelta(days=21),
    ).sort_index()
    filling_gwh = m["filling_gwh"].to_numpy()
    iso_week = df["hour_utc"].dt.isocalendar().week.clip(upper=52).to_numpy()
    valid = m["filling_gwh"].notna().to_numpy()

    wv = np.full(len(df), np.nan)
    wv[valid] = np.array([
        np.interp(f, storage_grid, wv_surface[w - 1])
        for f, w in zip(filling_gwh[valid], iso_week[valid])
    ])
    turbine_cap_mw = turbine_cap * 1000.0 / 168.0
    log(f"  Valor del agua (Francia): {valid.sum():,}/{len(df):,} horas con nivel conocido — "
        f"rango [{np.nanmin(wv):.1f}, {np.nanmax(wv):.1f}] EUR/MWh, capacidad de turbinado={turbine_cap_mw:.0f} MW")
    return pd.Series(wv, index=df.index, name="water_value_eur_mwh"), turbine_cap_mw


def main() -> None:
    def log(s: str = "") -> None:
        print(s)

    OUTPUT_DIR.mkdir(exist_ok=True)
    weekly = build_weekly_series()
    weekly.to_parquet(OUTPUT_DIR / "weekly_series_agua.parquet", index=False)
    print(f"Serie semanal de Francia: {len(weekly):,} semanas, {weekly['week'].min().date()} -> {weekly['week'].max().date()}")

    _, scenarios, max_storage, turbine_cap = load_scenarios(weekly_series_path=OUTPUT_DIR / "weekly_series_agua.parquet")
    V_all, policy, storage_grid = solve_dp(scenarios, max_storage, turbine_cap, log)
    wv = water_value_surface(V_all, storage_grid)
    print(f"\nValor del agua (Francia): rango [{wv.min():.2f}, {wv.max():.2f}] EUR/MWh")
    print(f"  Media en niveles BAJOS de embalse (<25%): {wv[:, storage_grid < 0.25*max_storage].mean():.2f} EUR/MWh")
    print(f"  Media en niveles ALTOS de embalse (>75%): {wv[:, storage_grid > 0.75*max_storage].mean():.2f} EUR/MWh")
    print("  (el valor del agua debería ser mayor con embalse bajo -- comprobación de sentido antes de usarla)")


if __name__ == "__main__":
    main()
