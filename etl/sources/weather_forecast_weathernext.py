"""
Acumulación diaria de la previsión Google WeatherNext (sucesor de GraphCast/GenCast,
Google DeepMind) vía la Ensemble API de Open-Meteo (ensemble-api.open-meteo.com).

Decisión de diseño clave (con el usuario, 2026-08-27): SIN BACKFILL — verificado en
vivo que el acceso gratuito vía Open-Meteo solo mantiene una ventana móvil de ~3-4
meses (rango permitido observado: 2026-05-25 → 2026-09-30; a fecha de hoy, pedir
2020-01-01 devuelve "Parameter 'start_date' is out of allowed range"). Histórico más
profundo requeriría el formulario oficial de solicitud de datos de Google WeatherNext
(aprobación ~5-7 días hábiles, acceso vía BigQuery/Earth Engine/GCS) — descartado por
ahora, mismo criterio que se aplicó a EEX/ASX Energy en su momento. Este pipeline
simplemente empieza a acumular desde hoy en adelante (mismo patrón que
etl/sources/omip_futures.py): cada ejecución diaria guarda la previsión completa del
momento, con su propio run_date, sin pisar los runs de días anteriores.

Otros hechos verificados en vivo el 2026-08-27:

1. SIN radiación solar: el modelo acepta el parámetro shortwave_radiation sin error,
   pero devuelve NULL en todos los puntos — no es un fallo de este pipeline, el modelo
   WeatherNext/GenCast no calcula esa variable. Por eso ECMWF IFS es la fuente elegida
   para la parte solar del estudio B1 (ver etl/sources/weather_forecast_ecmwf.py).
2. Es un ensemble de disponer ~50-65 miembros (varía; NO asumir un número fijo). La
   clave sin sufijo "_memberNN" NO es la media del ensemble (verificado: valores
   claramente distintos de la media calculada, es el "control run"/miembro no
   perturbado) — este pipeline calcula y guarda la MEDIA real de todos los "_memberNN"
   como valor representativo, no la clave base.
3. Actualiza cada 12h (runs 00/12 UTC), resolución espacial 0.25°, horizonte hasta 15
   días — más largo que ECMWF IFS en este ecosistema, útil para estudios de horizontes
   más allá de D-3 si se necesitan en el futuro (ampliar LEAD_TIME_DAYS_MAX).
"""

from __future__ import annotations

import statistics
import time
from datetime import date, datetime, timezone

import duckdb
import pandas as pd
import requests

from etl.sources.weather_common import LOCATIONS, document_forecast, ensure_schema

API_URL = "https://ensemble-api.open-meteo.com/v1/ensemble"
MODEL_KEY = "weathernext"
API_MODEL_PARAM = "google_weathernext2_ensemble"
FORECAST_DAYS = 15  # horizonte máximo ofrecido por el modelo
HOURLY_BASE_VARS = ["temperature_2m", "wind_speed_10m", "wind_speed_100m", "precipitation"]  # sin radiación solar

_session = requests.Session()


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------

def fetch_raw(lat: float, lon: float, timeout: int = 60, retries: int = 3) -> dict:
    params = {
        "latitude": lat, "longitude": lon,
        "models": API_MODEL_PARAM, "hourly": ",".join(HOURLY_BASE_VARS),
        "forecast_days": FORECAST_DAYS, "timezone": "UTC",
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

def parse(payload: dict, run_date: "date") -> list[dict]:
    h = payload["hourly"]
    times = [datetime.fromisoformat(t) for t in h["time"]]
    n = len(times)

    means: dict[str, list[float | None]] = {}
    for var in HOURLY_BASE_VARS:
        member_keys = [k for k in h if k.startswith(f"{var}_member")]
        if not member_keys:
            raise ValueError(f"sin miembros de ensemble para {var} — ¿cambió el esquema de la API?")
        col: list[float | None] = []
        for i in range(n):
            vals = [h[k][i] for k in member_keys if h[k][i] is not None]
            col.append(statistics.mean(vals) if vals else None)
        means[var] = col

    rows: list[dict] = []
    for i, target_dt in enumerate(times):
        if means["temperature_2m"][i] is None and means["wind_speed_100m"][i] is None:
            continue
        rows.append(
            {
                "interval_start_utc": target_dt,
                "run_date": run_date,
                "lead_time_days": (target_dt.date() - run_date).days,
                "temperature_2m_c": means["temperature_2m"][i],
                "wind_speed_10m_kmh": means["wind_speed_10m"][i],
                "wind_speed_100m_kmh": means["wind_speed_100m"][i],
                "shortwave_radiation_wm2": None,
                "direct_radiation_wm2": None,
                "diffuse_radiation_wm2": None,
                "precipitation_mm": means["precipitation"][i],
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
    run_date = rows[0]["run_date"]
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
    con.execute("BEGIN TRANSACTION")
    try:
        # Borrado escopado por run_date (NO por interval_start_utc): cada ejecución
        # diaria solo debe reemplazar SU PROPIO run, nunca los runs archivados de días
        # anteriores que cubren las mismas horas objetivo con un lead_time distinto.
        con.execute(
            "DELETE FROM weather_forecast WHERE location_key = ? AND model = ? AND run_date = ?",
            [location_key, MODEL_KEY, run_date],
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
        [f"weather_forecast_weathernext:{location_key}", "weather_forecast", date.today(), None, rows_loaded,
         status, message, started_at, datetime.utcnow()],
    )


def run_today(con: duckdb.DuckDBPyConnection, location_key: str, logger) -> dict:
    started_at = datetime.utcnow()
    ensure_schema(con)
    loc = LOCATIONS[location_key]
    run_date = datetime.utcnow().date()
    source_chunk = f"run={run_date.isoformat()}"

    try:
        payload = fetch_raw(loc["lat"], loc["lon"])
        rows = parse(payload, run_date)
    except Exception as exc:
        _log_run(con, location_key, 0, "ERROR", str(exc), started_at)
        logger.error("WeatherNext %s %s: %s", location_key, source_chunk, exc)
        return {"status": "ERROR", "message": str(exc)}

    status = "OK"
    messages = []
    if not rows:
        status = "WARN"
        messages.append("sin filas en este run")

    rows_loaded = load(con, location_key, rows, source_chunk)
    message = "; ".join(messages) if messages else f"sin incidencias, hasta D+{FORECAST_DAYS}"
    _log_run(con, location_key, rows_loaded, status, message, started_at)
    log_fn = logger.info if status == "OK" else logger.warning
    log_fn("WeatherNext %s %s: %s filas cargadas — %s", location_key, source_chunk, rows_loaded, message)
    return {"status": status, "rows_loaded": rows_loaded, "message": message}


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(__file__).rsplit("etl", 1)[0])
    from etl.common.db import connect, document_ingestion_log
    from etl.common.logging_config import get_logger
    from etl.sources.weather_common import DB_NAME

    loc = sys.argv[1] if len(sys.argv) > 1 else "spain"
    log = get_logger("weather_forecast_weathernext.manual", DB_NAME)
    conn = connect(DB_NAME)
    result = run_today(conn, loc, log)
    print(result)
    document_forecast(conn)
    document_ingestion_log(conn)
    conn.close()
