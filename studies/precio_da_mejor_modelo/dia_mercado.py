"""El dia de mercado es LOCAL, no UTC. Corrige una fuga del 93,4% de las filas.

**El bug.** `build_dataset.py` agrupa por `df["date"] = period_start_utc.dt.date`,
la fecha **UTC**. El dia de mercado espanol es CET/CEST, asi que hay un desfase
de una o dos horas y la consecuencia es directa:

- `price_boundary_prev_day` entrega, en el **93,4%** de las filas, un precio del
  PROPIO dia de mercado — fijado en la misma subasta que se esta prediciendo.
  Medido sobre las 57.502 filas del dataset: el valor asignado al 18 de
  noviembre sale de las 23:45 UTC del 17, que en hora local **ya es el 18**.
- `price_ema_7d` y `price_ema_28d` arrastran periodos del dia siguiente por la
  misma via.
- `is_weekend`, `is_holiday` y la hora del dia son UTC, no locales.

**Cuanto vale.** En el modelo privado equivalente se midio: **+0,736 EUR/MWh**
la fuga sola, y **+0,684 neto** despues de recuperar −0,052 con el calendario
local. Es decir, las cifras publicadas sin esta correccion son OPTIMISTAS en
torno a 0,7 EUR/MWh.

**Como se arregla, y por que asi.** Recalculando sobre el dataset ya construido
en vez de tocar `_add_price_lags`, que es codigo compartido con el modelo de
Francia: duplicar antes que acoplar. Son dos capas, para poder atribuir cuanto
viene de cada una:

    sin_fuga   frontera y EMAs sobre el dia de mercado   (quita la fuga)
    local      ademas, calendario y hora locales         (corrige la hora)

Se aplica al final de `build_dataset.build()`, antes de `features_v3`.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
TZ = "Europe/Madrid"


def _local(s: pd.Series) -> pd.Series:
    return s.dt.tz_localize("UTC").dt.tz_convert(TZ).dt.tz_localize(None)


def sin_fuga(df: pd.DataFrame, log=print) -> pd.DataFrame:
    """Frontera y EMAs recalculadas sobre el DIA DE MERCADO local."""
    df = df.sort_values("period_start_utc").reset_index(drop=True)
    dia = _local(df["period_start_utc"]).dt.date

    antes = df["price_boundary_prev_day"].copy()
    ultimo = df.groupby(dia)["price_real"].last()
    ultimo.index = pd.to_datetime(ultimo.index) + pd.Timedelta(days=1)
    df["price_boundary_prev_day"] = pd.to_datetime(dia).map(ultimo)
    n = (antes != df["price_boundary_prev_day"]).sum()
    log(f"  frontera: {n:,} filas cambian de valor ({n / len(df):.1%})")

    media = df.groupby(dia)["price_real"].mean().sort_index()
    media.index = pd.to_datetime(media.index)
    for span, col in ((7, "price_ema_7d"), (28, "price_ema_28d")):
        df[col] = pd.to_datetime(dia).map(media.ewm(span=span).mean().shift(1))
    log("  EMAs 7d/28d recalculadas sobre la media del dia de mercado")
    return df


def calendario_local(df: pd.DataFrame, log=print) -> pd.DataFrame:
    """Hora del dia, dia de la semana, mes, festivo y fin de semana, en local."""
    loc = _local(df["period_start_utc"])
    con = duckdb.connect(str(DATA_DIR / "spain.duckdb"), read_only=True)
    try:
        fes = con.execute(
            "SELECT holiday_date FROM holiday_calendar WHERE location_key = 'spain'"
        ).fetchdf()
    finally:
        con.close()
    fes = set(pd.to_datetime(fes["holiday_date"]).dt.date)

    antes_h = df["hour_of_day"].copy()
    antes_w = df["is_weekend"].copy()
    df["hour_of_day"] = loc.dt.hour
    df["period_in_hour"] = loc.dt.minute // 15
    df["period_of_day"] = df["hour_of_day"] * 4 + df["period_in_hour"]
    df["day_of_week"] = loc.dt.dayofweek
    df["month"] = loc.dt.month
    df["is_holiday"] = loc.dt.date.isin(fes).astype(int)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    log(f"  hora del dia: {(antes_h != df['hour_of_day']).sum():,} filas cambian")
    log(f"  is_weekend:   {(antes_w.astype(int) != df['is_weekend']).sum():,} filas cambian")
    return df


def aplicar(df: pd.DataFrame, log=print) -> pd.DataFrame:
    """Las dos capas, que es lo que esta validado en el modelo privado."""
    df = sin_fuga(df, log)
    df = calendario_local(df, log)
    return df
