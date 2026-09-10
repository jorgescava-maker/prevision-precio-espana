"""
Construye la serie semanal del estudio C3 (valor del agua) — ver DESIGN.md §2.
Deriva la aportación hidrológica por balance de masa/energía, sin fuente nueva.

Uso:
    .venv\\Scripts\\python.exe -m studies.c3_valor_agua.build_dataset
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def build() -> pd.DataFrame:
    con = duckdb.connect(str(DATA_DIR / "spain.duckdb"), read_only=True)
    try:
        filling = con.execute(
            "SELECT interval_start_utc::date AS week, filling_mwh FROM hydro_reservoir_filling ORDER BY 1"
        ).fetchdf()
        gen_weekly = con.execute(
            """
            SELECT date_trunc('week', interval_start_utc) AS week_start,
                   avg(value_mw) * 24 * 7 / 1000 AS gen_hidraulica_gwh_week
            FROM esios_generation_by_type WHERE category_key = 'hidraulica'
            GROUP BY 1 ORDER BY 1
            """
        ).fetchdf()
        price_weekly = con.execute(
            """
            SELECT date_trunc('week', delivery_start_utc) AS week_start, avg(price_eur_mwh_es) AS price_avg
            FROM omie_spot_prices GROUP BY 1 ORDER BY 1
            """
        ).fetchdf()
    finally:
        con.close()

    filling["week"] = pd.to_datetime(filling["week"])
    filling = filling.sort_values("week").reset_index(drop=True)
    filling["filling_gwh"] = filling["filling_mwh"] / 1000
    filling["delta_filling_gwh"] = filling["filling_gwh"].diff()

    # `hydro_reservoir_filling` publica en domingo; DuckDB date_trunc('week', ...) usa
    # lunes (ISO) — no coinciden en fecha exacta, así que se empareja por (año, semana
    # ISO) en vez de por igualdad de fecha, o el join sale vacío.
    def iso_key(s: pd.Series) -> pd.Series:
        iso = s.dt.isocalendar()
        return iso["year"].astype(str) + "-W" + iso["week"].astype(str)

    filling["iso_week"] = iso_key(filling["week"])
    gen_weekly["iso_week"] = iso_key(gen_weekly["week_start"])
    price_weekly["iso_week"] = iso_key(price_weekly["week_start"])

    df = filling.merge(gen_weekly[["iso_week", "gen_hidraulica_gwh_week"]], on="iso_week", how="left")
    df = df.merge(price_weekly[["iso_week", "price_avg"]], on="iso_week", how="left")

    df["inflow_gwh"] = df["delta_filling_gwh"] + df["gen_hidraulica_gwh_week"]
    df["price_next_4w_avg"] = df["price_avg"].rolling(4).mean().shift(-4)  # precio medio de las 4 semanas SIGUIENTES

    return df[["week", "filling_gwh", "delta_filling_gwh", "gen_hidraulica_gwh_week",
               "inflow_gwh", "price_avg", "price_next_4w_avg"]]


def main() -> None:
    df = build()
    OUTPUT_DIR.mkdir(exist_ok=True)
    out_path = OUTPUT_DIR / "weekly_series.parquet"
    df.to_parquet(out_path, index=False)
    print(f"Serie semanal construida: {len(df):,} semanas, {df['week'].min()} -> {df['week'].max()}")
    print(f"Guardado en {out_path}")


if __name__ == "__main__":
    main()
