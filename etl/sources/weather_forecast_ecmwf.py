"""
Ingesta de previsión meteorológica ECMWF IFS 0.25° CON HORIZONTE REAL preservado
(lead time 0-3 días) vía la Previous Runs API de Open-Meteo (previous-runs-api.
open-meteo.com). Es la pieza clave para el estudio B1 (alpha de error de previsión):
sin el horizonte real, cualquier comparación previsto-vs-real sería tramposa (mezclaría
información que en su momento no se conocía todavía).

Gratis, sin API key, verificado en vivo el 2026-08-27:

1. Los sufijos `_previous_dayN` del parámetro hourly dan el valor que el modelo predijo
   N días antes de la fecha objetivo (N=0 es el análisis/nowcast del propio día, no una
   previsión real) — YA NO es necesario reconstruirlo nosotros combinando corridas.
2. A diferencia de WeatherNext (ver etl/sources/weather_forecast_weathernext.py),
   soporta rango histórico real: verificado con éxito para 2024-06-01, con bisección
   se localizó el inicio real de radiación solar el 2024-03-05 (la documentación oficial
   decía "2024-02-03", pero esa fecha no tiene datos reales de shortwave_radiation —
   verificado en vivo, no asumido).
3. SÍ incluye radiación solar (GHI/directa/difusa), a diferencia de WeatherNext — por
   eso es la fuente elegida para la parte solar del estudio B1 (decisión con el usuario,
   2026-08-27).
"""

from __future__ import annotations

import time
from datetime import date, datetime, timedelta, timezone

import duckdb
import pandas as pd
import requests

from etl.sources.weather_common import LOCATIONS, document_forecast, ensure_schema

API_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
MODEL_KEY = "ecmwf_ifs025"
LEAD_TIMES = (0, 1, 2, 3)  # días — ampliable hasta 7 sin cambios de esquema

# Fecha real (verificada en vivo por bisección) desde la que ecmwf_ifs025 tiene
# shortwave_radiation no nula en este endpoint. Anterior a esto, la API responde 200
# con filas de valor NULL en vez de error — sin este umbral, un backfill que empezara
# en 2023-01-01 (como el resto del ecosistema) cargaría miles de filas vacías sin
# avisar.
EARLIEST_SOLAR_DATE = date(2024, 3, 6)

HOURLY_VARS_TEMPLATE = (
    "temperature_2m{s},wind_speed_10m{s},wind_speed_100m{s},"
    "shortwave_radiation{s},direct_radiation{s},diffuse_radiation{s},precipitation{s}"
)

_session = requests.Session()


def _suffix(lead: int) -> str:
    return "" if lead == 0 else f"_previous_day{lead}"


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------

def fetch_raw(lat: float, lon: float, start_date: str, end_date: str, timeout: int = 60, retries: int = 3) -> dict:
    hourly = ",".join(HOURLY_VARS_TEMPLATE.format(s=_suffix(lead)) for lead in LEAD_TIMES)
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": start_date, "end_date": end_date,
        "models": MODEL_KEY, "hourly": hourly, "timezone": "UTC",
    }
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            resp = _session.get(API_URL, params=params, timeout=timeout)
            resp.raise_for_status()
            payload = resp.json()
            if payload.get("error"):
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
    """Una fila por (hora objetivo, lead_time). run_date = fecha_objetivo - lead."""
    h = payload["hourly"]
    times = [datetime.fromisoformat(t) for t in h["time"]]
    rows: list[dict] = []
    for lead in LEAD_TIMES:
        s = _suffix(lead)
        temp = h[f"temperature_2m{s}"]
        w10 = h[f"wind_speed_10m{s}"]
        w100 = h[f"wind_speed_100m{s}"]
        sw = h[f"shortwave_radiation{s}"]
        direct = h[f"direct_radiation{s}"]
        diffuse = h[f"diffuse_radiation{s}"]
        precip = h[f"precipitation{s}"]
        for i, target_dt in enumerate(times):
            if temp[i] is None and w100[i] is None and sw[i] is None:
                continue  # fuera de cobertura para este lead (p.ej. antes de EARLIEST_SOLAR_DATE)
            rows.append(
                {
                    "interval_start_utc": target_dt,
                    "run_date": target_dt.date() - timedelta(days=lead),
                    "lead_time_days": lead,
                    "temperature_2m_c": temp[i], "wind_speed_10m_kmh": w10[i], "wind_speed_100m_kmh": w100[i],
                    "shortwave_radiation_wm2": sw[i], "direct_radiation_wm2": direct[i], "diffuse_radiation_wm2": diffuse[i],
                    "precipitation_mm": precip[i],
                }
            )
    return rows


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
            location_key, MODEL_KEY, r["run_date"], r["interval_start_utc"],
            r["interval_start_utc"].replace(tzinfo=timezone.utc).astimezone(tz).replace(tzinfo=None),
            r["lead_time_days"], r["temperature_2m_c"], r["wind_speed_10m_kmh"], r["wind_speed_100m_kmh"],
            r["shortwave_radiation_wm2"], r["direct_radiation_wm2"], r["diffuse_radiation_wm2"],
            source_chunk, now, r["precipitation_mm"],
        )
        for r in rows
    ]
    # precipitation_mm al final — ver comentario en weather_common.ensure_schema().
    df = pd.DataFrame(
        records,
        columns=["location_key", "model", "run_date", "interval_start_utc", "interval_start_local",
                 "lead_time_days", "temperature_2m_c", "wind_speed_10m_kmh", "wind_speed_100m_kmh",
                 "shortwave_radiation_wm2", "direct_radiation_wm2", "diffuse_radiation_wm2",
                 "source_chunk", "ingested_at", "precipitation_mm"],
    )
    min_run, max_run = df["run_date"].min(), df["run_date"].max()
    con.execute("BEGIN TRANSACTION")
    try:
        con.execute(
            "DELETE FROM weather_forecast WHERE location_key = ? AND model = ? AND run_date >= ? AND run_date <= ?",
            [location_key, MODEL_KEY, min_run, max_run],
        )
        con.append("weather_forecast", df)
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
        [f"weather_forecast_ecmwf:{location_key}", "weather_forecast", date.today(), None, rows_loaded, status,
         message, started_at, datetime.utcnow()],
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
        logger.error("ECMWF %s %s: %s", location_key, source_chunk, exc)
        return {"status": "ERROR", "message": str(exc)}

    status = "OK"
    messages = []
    if not rows:
        status = "WARN"
        messages.append("sin filas en este bloque (posiblemente fuera de EARLIEST_SOLAR_DATE)")

    rows_loaded = load(con, location_key, rows, source_chunk)
    message = "; ".join(messages) if messages else f"sin incidencias, {len(LEAD_TIMES)} horizontes"
    _log_run(con, location_key, rows_loaded, status, message, started_at)
    log_fn = logger.info if status == "OK" else logger.warning
    log_fn("ECMWF %s %s: %s filas cargadas — %s", location_key, source_chunk, rows_loaded, message)
    return {"status": status, "rows_loaded": rows_loaded, "message": message}


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(__file__).rsplit("etl", 1)[0])
    from etl.common.db import connect, document_ingestion_log
    from etl.common.logging_config import get_logger
    from etl.sources.weather_common import DB_NAME

    loc = sys.argv[1] if len(sys.argv) > 1 else "spain"
    log = get_logger("weather_forecast_ecmwf.manual", DB_NAME)
    conn = connect(DB_NAME)
    result = run_for_range(conn, loc, "2024-06-01", "2024-06-08", log)
    print(result)
    document_forecast(conn)
    document_ingestion_log(conn)
    conn.close()
