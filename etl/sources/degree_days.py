"""
Grados-día de calefacción/refrigeración (HDD/CDD), derivados de `weather_actuals`
(ERA5, ya cargado en weather.duckdb) — backlog de ideas #4 de status.md. No es una
ingesta: es agregación SQL local sobre datos que ya están en la base, sin red ni fuente
externa, mismo espíritu que `holiday_calendar.py`.

Definición estándar (base 18°C, convención europea — "temperatura de confort" en la que
no hace falta ni calentar ni enfriar):
    HDD = max(0, 18 - temperatura_media_del_día)
    CDD = max(0, temperatura_media_del_día - 18)

Por qué HDD/CDD y no la temperatura cruda: la relación entre temperatura y demanda
eléctrica no es monótona en un solo sentido (hace falta energía tanto si hace mucho frío
como si hace mucho calor, con un valle de confort en medio) — separar el efecto en dos
variables que sí correlacionan linealmente con demanda es el insumo estándar en energía
para A1/A2, más útil que la temperatura media sola.

Se calcula sobre el DÍA LOCAL (`interval_start_local`, no UTC) de cada ubicación, porque
es el día que percibe el consumidor cuya demanda se quiere explicar. Solo se agregan
días con al menos MIN_HOURS_PER_DAY lecturas horarias (de 24 esperadas) — descarta de
forma natural el día más reciente, que llega incompleto por el retraso de 1 día del
archive-api de ERA5 (ver findings.md #38) y se completará solo con la actualización
diaria siguiente.

Ubicaciones: mismas 9 claves que weather_common.py (4 países UE de zona única + 5
subregiones de precio NEM) — se recalculan aquí en vez de importarse para no acoplar
este pipeline al de clima; si LOCATIONS cambia en weather_common.py, replicar el cambio.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

import duckdb

from etl.common import catalog
from etl.common.catalog import ColumnDoc

BASE_TEMP_C = 18.0
MIN_HOURS_PER_DAY = 20

LOCATION_KEYS = ["spain", "germany", "france", "netherlands", "nsw1", "qld1", "sa1", "tas1", "vic1"]


# ---------------------------------------------------------------------------
# Compute (agregación SQL local, sin red)
# ---------------------------------------------------------------------------

def compute(con: duckdb.DuckDBPyConnection, location_key: str, start_date: date | None = None,
            end_date: date | None = None) -> list[dict]:
    where = ["location_key = ?"]
    params: list = [location_key]
    if start_date is not None:
        where.append("interval_start_local >= ?")
        params.append(datetime(start_date.year, start_date.month, start_date.day))
    if end_date is not None:
        where.append("interval_start_local < ?")
        params.append(datetime(end_date.year, end_date.month, end_date.day) + timedelta(days=1))

    query = f"""
        SELECT date_trunc('day', interval_start_local)::DATE AS calendar_date,
               avg(temperature_2m_c) AS avg_temp_c,
               count(*) AS n_hours
        FROM weather_actuals
        WHERE {' AND '.join(where)}
        GROUP BY 1
        HAVING count(*) >= {MIN_HOURS_PER_DAY}
        ORDER BY 1
    """
    rows = []
    for calendar_date, avg_temp_c, n_hours in con.execute(query, params).fetchall():
        hdd = max(0.0, BASE_TEMP_C - avg_temp_c)
        cdd = max(0.0, avg_temp_c - BASE_TEMP_C)
        rows.append({
            "calendar_date": calendar_date, "avg_temp_c": avg_temp_c,
            "hdd": hdd, "cdd": cdd, "n_hours": n_hours,
        })
    return rows


# ---------------------------------------------------------------------------
# Validate
# ---------------------------------------------------------------------------

def validate(rows: list[dict]) -> dict:
    if not rows:
        return {"status": "WARN", "messages": ["sin días con cobertura horaria suficiente en este bloque"], "actual_rows": 0}

    dates = sorted(r["calendar_date"] for r in rows)
    gaps_over_5d = sum(1 for a, b in zip(dates, dates[1:]) if (b - a).days > 5)
    messages: list[str] = []
    status = "OK"
    if gaps_over_5d:
        status = "WARN"
        messages.append(f"{gaps_over_5d} huecos de más de 5 días naturales entre días agregados consecutivos")

    return {"status": status, "messages": messages, "actual_rows": len(rows)}


# ---------------------------------------------------------------------------
# Diccionario de datos
# ---------------------------------------------------------------------------

DEGREE_DAYS_COLUMNS = {
    "location_key": ColumnDoc(
        description="Zona representada: spain/germany/france/netherlands (país, zona única) o nsw1/qld1/sa1/tas1/vic1 (subregión NEM) — mismas claves que weather_actuals.",
        source="Constante de configuración (LOCATION_KEYS en el pipeline).",
        dtype="VARCHAR", kind="categorical",
    ),
    "calendar_date": ColumnDoc(
        description="Día local (no UTC) al que se refiere el agregado.",
        source="date_trunc('day', interval_start_local) sobre weather_actuals.",
        dtype="DATE", kind="identifier",
    ),
    "avg_temp_c": ColumnDoc(
        description="Temperatura media del día local, promedio simple de las lecturas horarias disponibles ese día.",
        source="avg(temperature_2m_c) de weather_actuals, agrupado por día local.",
        dtype="DOUBLE", kind="continuous",
    ),
    "hdd": ColumnDoc(
        description="Grados-día de calefacción: max(0, 18 - avg_temp_c). Proxy de demanda de calefacción ese día.",
        source="Calculado por el pipeline a partir de avg_temp_c, base 18°C (convención europea estándar).",
        dtype="DOUBLE", kind="continuous",
    ),
    "cdd": ColumnDoc(
        description="Grados-día de refrigeración: max(0, avg_temp_c - 18). Proxy de demanda de aire acondicionado ese día.",
        source="Calculado por el pipeline a partir de avg_temp_c, base 18°C (convención europea estándar).",
        dtype="DOUBLE", kind="continuous",
    ),
    "n_hours": ColumnDoc(
        description="Número de lecturas horarias de weather_actuals usadas en el promedio ese día (de 24 esperadas). Solo se cargan días con n_hours >= 20 — descarta días con cobertura insuficiente (típicamente el más reciente, aún no completado por el retraso de origen de ERA5).",
        source="count(*) de weather_actuals, agrupado por día local.",
        dtype="SMALLINT", kind="continuous",
    ),
    "ingested_at": ColumnDoc(
        description="Marca temporal UTC del momento en que esta fila se cargó (o recargó) por última vez en la base de datos.",
        source="datetime.utcnow() en el momento de la inserción, dentro de la función load().",
        dtype="TIMESTAMP", kind="timestamp",
    ),
}


def document(con: duckdb.DuckDBPyConnection) -> None:
    catalog.refresh(con, "weather_degree_days", DEGREE_DAYS_COLUMNS)


# ---------------------------------------------------------------------------
# Schema + load
# ---------------------------------------------------------------------------

def ensure_schema(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS weather_degree_days (
            location_key   VARCHAR NOT NULL,
            calendar_date  DATE NOT NULL,
            avg_temp_c     DOUBLE NOT NULL,
            hdd            DOUBLE NOT NULL,
            cdd            DOUBLE NOT NULL,
            n_hours        SMALLINT NOT NULL,
            ingested_at    TIMESTAMP NOT NULL,
            PRIMARY KEY (location_key, calendar_date)
        )
        """
    )


def load(con: duckdb.DuckDBPyConnection, location_key: str, rows: list[dict]) -> int:
    ensure_schema(con)
    if not rows:
        return 0
    now = datetime.utcnow()
    dates = [r["calendar_date"] for r in rows]
    min_d, max_d = min(dates), max(dates)

    con.execute("BEGIN TRANSACTION")
    try:
        con.execute(
            "DELETE FROM weather_degree_days WHERE location_key = ? AND calendar_date >= ? AND calendar_date <= ?",
            [location_key, min_d, max_d],
        )
        con.executemany(
            """
            INSERT INTO weather_degree_days (location_key, calendar_date, avg_temp_c, hdd, cdd, n_hours, ingested_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [(location_key, r["calendar_date"], r["avg_temp_c"], r["hdd"], r["cdd"], r["n_hours"], now) for r in rows],
        )
        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK")
        raise
    return len(rows)


# ---------------------------------------------------------------------------
# Orquestación + registro de auditoría
# ---------------------------------------------------------------------------

def _log_run(con, location_key, source_chunk, rows_loaded, status, message, started_at) -> None:
    con.execute(
        """
        INSERT INTO ingestion_log
            (source, target_table, delivery_date, rows_expected, rows_loaded, status, message, started_at, finished_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [f"degree_days:{location_key}", "weather_degree_days", date.today(), None, rows_loaded, status, message,
         started_at, datetime.utcnow()],
    )


def run_for_location(con: duckdb.DuckDBPyConnection, location_key: str, logger,
                      start_date: date | None = None, end_date: date | None = None) -> dict:
    started_at = datetime.utcnow()
    ensure_schema(con)
    source_chunk = f"{start_date}-{end_date}"

    try:
        rows = compute(con, location_key, start_date, end_date)
    except Exception as exc:
        _log_run(con, location_key, source_chunk, 0, "ERROR", str(exc), started_at)
        logger.error("Degree days %s: %s", location_key, exc)
        return {"status": "ERROR", "location": location_key, "message": str(exc)}

    validation = validate(rows)
    rows_loaded = load(con, location_key, rows)

    status = validation["status"]
    message = "; ".join(validation["messages"]) if validation["messages"] else "sin incidencias"
    _log_run(con, location_key, source_chunk, rows_loaded, status, message, started_at)

    log_fn = logger.info if status == "OK" else logger.warning
    log_fn("Degree days %s: %s días cargados — %s", location_key, rows_loaded, message)
    return {"status": status, "location": location_key, "rows_loaded": rows_loaded, "message": message}


def run_all(logger, start_date: date | None = None, end_date: date | None = None) -> dict:
    """Recalcula HDD/CDD para las 9 ubicaciones sobre weather.duckdb (una sola conexión,
    todas las ubicaciones comparten el mismo fichero)."""
    from etl.common.db import connect, document_ingestion_log

    con = connect("weather")
    summary = {"OK": 0, "WARN": 0, "ERROR": 0}
    for location_key in LOCATION_KEYS:
        result = run_for_location(con, location_key, logger, start_date, end_date)
        summary[result["status"]] = summary.get(result["status"], 0) + 1
    document(con)
    document_ingestion_log(con)
    con.close()
    return summary


if __name__ == "__main__":
    from etl.common.logging_config import get_logger

    log = get_logger("degree_days.manual", "weather")
    print(run_all(log))
