"""
Ingesta de clima OBSERVADO (reanálisis ERA5/ERA5-Land) vía la Historical Weather API
de Open-Meteo (archive-api.open-meteo.com). Sirve de "verdad terreno" para comparar
contra las previsiones de etl/sources/weather_forecast_ecmwf.py y
etl/sources/weather_forecast_weathernext.py (estudio B1, alpha de error de previsión).

Gratis, sin API key, verificado en vivo el 2026-08-27. Histórico disponible desde 1940
(ERA5, 0.25°) / 1950 (ERA5-Land, 0.1°) — Open-Meteo selecciona automáticamente la mejor
fuente por variable, sin que el pipeline tenga que elegir entre ambas.

Decisión de diseño: un único punto representativo por país (capital), no una media
ponderada por población/capacidad instalada — ver docs/status.md. Reutiliza LOCATIONS
de etl/sources/weather_common.py, compartida con los pipelines de previsión.
"""

from __future__ import annotations

import time
from datetime import date, datetime, timezone

import duckdb
import pandas as pd
import requests

from etl.sources.weather_common import LOCATIONS, document_actuals, ensure_schema

API_URL = "https://archive-api.open-meteo.com/v1/archive"
HOURLY_VARS = "temperature_2m,wind_speed_10m,wind_speed_100m,shortwave_radiation,direct_radiation,diffuse_radiation,precipitation"

_session = requests.Session()


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------

def fetch_raw(lat: float, lon: float, start_date: str, end_date: str, timeout: int = 60, retries: int = 3) -> dict:
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": start_date, "end_date": end_date,
        "hourly": HOURLY_VARS, "timezone": "UTC",
    }
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            resp = _session.get(API_URL, params=params, timeout=timeout)
            resp.raise_for_status()
            payload = resp.json()
            if "error" in payload and payload["error"]:
                raise ValueError(f"Open-Meteo error: {payload.get('reason')}")
            return payload
        except (requests.RequestException, ValueError) as exc:
            last_exc = exc
            time.sleep(2 * attempt)
    raise last_exc  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Parse
# ---------------------------------------------------------------------------

def parse(payload: dict) -> list[dict]:
    h = payload["hourly"]
    rows = []
    for i, t in enumerate(h["time"]):
        rows.append(
            {
                "interval_start_utc": datetime.fromisoformat(t),
                "temperature_2m_c": h["temperature_2m"][i],
                "wind_speed_10m_kmh": h["wind_speed_10m"][i],
                "wind_speed_100m_kmh": h["wind_speed_100m"][i],
                "shortwave_radiation_wm2": h["shortwave_radiation"][i],
                "direct_radiation_wm2": h["direct_radiation"][i],
                "diffuse_radiation_wm2": h["diffuse_radiation"][i],
                "precipitation_mm": h["precipitation"][i],
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Validate
# ---------------------------------------------------------------------------

def validate(rows: list[dict]) -> dict:
    if not rows:
        return {"status": "WARN", "messages": ["sin filas en este bloque"]}
    messages = []
    status = "OK"
    n_null_temp = sum(1 for r in rows if r["temperature_2m_c"] is None)
    if n_null_temp:
        status = "WARN"
        messages.append(f"{n_null_temp} horas con temperatura nula (rango probablemente fuera de cobertura ERA5)")
    return {"status": status, "messages": messages}


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load(con: duckdb.DuckDBPyConnection, location_key: str, rows: list[dict], source_chunk: str) -> int:
    ensure_schema(con)
    if not rows:
        return 0
    tz = LOCATIONS[location_key]["tz"]
    now = datetime.utcnow()
    records = [
        (
            location_key, r["interval_start_utc"],
            r["interval_start_utc"].replace(tzinfo=timezone.utc).astimezone(tz).replace(tzinfo=None),
            r["temperature_2m_c"], r["wind_speed_10m_kmh"], r["wind_speed_100m_kmh"],
            r["shortwave_radiation_wm2"], r["direct_radiation_wm2"], r["diffuse_radiation_wm2"],
            source_chunk, now, r["precipitation_mm"],
        )
        for r in rows
    ]
    # precipitation_mm va AL FINAL (después de ingested_at): con.append() es posicional
    # y la tabla física la tiene ahí por venir de un ALTER TABLE — ver ensure_schema().
    df = pd.DataFrame(
        records,
        columns=["location_key", "interval_start_utc", "interval_start_local", "temperature_2m_c",
                 "wind_speed_10m_kmh", "wind_speed_100m_kmh", "shortwave_radiation_wm2",
                 "direct_radiation_wm2", "diffuse_radiation_wm2", "source_chunk", "ingested_at",
                 "precipitation_mm"],
    )
    min_ts, max_ts = df["interval_start_utc"].min(), df["interval_start_utc"].max()
    con.execute("BEGIN TRANSACTION")
    try:
        con.execute(
            "DELETE FROM weather_actuals WHERE location_key = ? AND interval_start_utc >= ? AND interval_start_utc <= ?",
            [location_key, min_ts, max_ts],
        )
        con.append("weather_actuals", df)
        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK")
        raise
    return len(df)


# ---------------------------------------------------------------------------
# Orquestación + registro de auditoría
# ---------------------------------------------------------------------------

def _log_run(con, location_key, rows_loaded, status, message, started_at) -> None:
    con.execute(
        """
        INSERT INTO ingestion_log
            (source, target_table, delivery_date, rows_expected, rows_loaded, status, message, started_at, finished_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [f"weather_era5:{location_key}", "weather_actuals", date.today(), None, rows_loaded, status, message,
         started_at, datetime.utcnow()],
    )


def run_for_range(con: duckdb.DuckDBPyConnection, location_key: str, start_date: str, end_date: str, logger) -> dict:
    started_at = datetime.utcnow()
    ensure_schema(con)
    loc = LOCATIONS[location_key]
    source_chunk = f"{start_date}-{end_date}"

    try:
        payload = fetch_raw(loc["lat"], loc["lon"], start_date, end_date)
        rows = parse(payload)
    except Exception as exc:
        _log_run(con, location_key, 0, "ERROR", str(exc), started_at)
        logger.error("ERA5 %s %s: %s", location_key, source_chunk, exc)
        return {"status": "ERROR", "message": str(exc)}

    validation = validate(rows)
    rows_loaded = load(con, location_key, rows, source_chunk)

    status = validation["status"]
    message = "; ".join(validation["messages"]) if validation["messages"] else "sin incidencias"
    _log_run(con, location_key, rows_loaded, status, message, started_at)
    log_fn = logger.info if status == "OK" else logger.warning
    log_fn("ERA5 %s %s: %s filas cargadas — %s", location_key, source_chunk, rows_loaded, message)
    return {"status": status, "rows_loaded": rows_loaded, "message": message}


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(__file__).rsplit("etl", 1)[0])
    from etl.common.db import connect, document_ingestion_log
    from etl.common.logging_config import get_logger
    from etl.sources.weather_common import DB_NAME

    loc = sys.argv[1] if len(sys.argv) > 1 else "spain"
    log = get_logger("weather_era5.manual", DB_NAME)
    conn = connect(DB_NAME)
    result = run_for_range(conn, loc, "2023-01-01", "2023-01-08", log)
    print(result)
    document_actuals(conn)
    document_ingestion_log(conn)
    conn.close()
