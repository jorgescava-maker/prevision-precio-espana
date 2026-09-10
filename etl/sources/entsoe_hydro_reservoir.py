"""
Ingesta de nivel agregado de reservas hidráulicas (documentType A72, "Aggregate
Filling Rate of Water Reservoirs and Hydro Storage Plants") vía ENTSO-E Transparency
Platform. Fase 5 del roadmap.

Requiere ENTSOE_API_TOKEN en .env (mismo token que el resto de pipelines ENTSO-E).

Hechos verificados en vivo el 2026-08-27:

1. Solo España y Francia tienen este dato — Alemania y Países Bajos devuelven
   explícitamente "No matching data found" (verificado con los 4 países del
   ecosistema). Es una limitación geográfica real (Alemania sí tiene algo de
   hidráulica alpina pero no publica esta agregación vía ENTSO-E; Países Bajos
   prácticamente no tiene hidráulica), no un fallo de la petición.
2. Resolución semanal (`P7D`, formato de periodo ISO 8601 de CALENDARIO — no
   `PTxxM` como el resto de series ENTSO-E de este ecosistema). No se puede reutilizar
   `etl/common/entsoe_xml.py` (asume resolución en minutos) — este pipeline tiene su
   propio parseo, más simple porque no hay puntos omitidos que rellenar (una semana =
   un único valor, sin sub-intervalos).
3. Unidad: MWh — es el CONTENIDO ENERGÉTICO equivalente del agua embalsada (lo que se
   podría generar vaciando el embalse), no un volumen de agua. Es la convención
   estándar de ENTSO-E para este dato, y permite comparar directamente con generación
   (MWh) sin conversión.
4. Publicación con ~4 días de retraso respecto a la fecha de la petición (verificado:
   pidiendo hasta el 27-ago-2026, el último punto real cubre la semana que termina el
   23-ago-2026).
"""

from __future__ import annotations

import time
import xml.etree.ElementTree as ET
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import duckdb
import pandas as pd
import requests

from etl.common import catalog
from etl.common.catalog import ColumnDoc
from etl.common.config import require_env

API_URL = "https://web-api.tp.entsoe.eu/api"

AREAS = {
    "spain": {"eic": "10YES-REE------0", "tz": ZoneInfo("Europe/Madrid"), "db": "spain"},
    "france": {"eic": "10YFR-RTE------C", "tz": ZoneInfo("Europe/Paris"), "db": "france"},
}

FILLING_SANITY_MAX_MWH = 30_000_000.0  # generoso: máx. observado en vivo ES ~14.1M MWh

_session = requests.Session()


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------

def fetch_raw(eic: str, period_start: str, period_end: str, timeout: int = 60, retries: int = 3) -> str:
    params = {
        "securityToken": require_env("ENTSOE_API_TOKEN"),
        "documentType": "A72",
        "processType": "A16",
        "in_Domain": eic,
        "periodStart": period_start,
        "periodEnd": period_end,
    }
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            resp = _session.get(API_URL, params=params, timeout=timeout)
            resp.raise_for_status()
            return resp.text
        except requests.RequestException as exc:
            last_exc = exc
            time.sleep(2 * attempt)
    raise last_exc  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Parse
# ---------------------------------------------------------------------------

def _strip_ns(elem: ET.Element) -> ET.Element:
    for el in elem.iter():
        if "}" in el.tag:
            el.tag = el.tag.split("}", 1)[1]
    return elem


def parse(xml_text: str) -> list[dict]:
    """Una fila por semana. Resolución P7D (calendario) — sin puntos omitidos que
    rellenar (a diferencia de las series PTxxM), cada <Point> es explícito."""
    root = _strip_ns(ET.fromstring(xml_text))

    if root.tag == "Acknowledgement_MarketDocument":
        reason = root.findtext(".//Reason/text") or "sin detalle"
        raise ValueError(f"ENTSO-E devolvió un Acknowledgement (sin datos): {reason}")

    rows: list[dict] = []
    for ts in root.findall("TimeSeries"):
        for period in ts.findall("Period"):
            start_str = period.findtext("timeInterval/start")
            resolution_str = period.findtext("resolution")
            if resolution_str != "P7D":
                raise ValueError(f"resolución inesperada para A72: {resolution_str} (se esperaba P7D)")
            start_dt = datetime.strptime(start_str, "%Y-%m-%dT%H:%MZ").replace(tzinfo=timezone.utc)

            for pt in period.findall("Point"):
                pos = int(pt.findtext("position"))
                qty = float(pt.findtext("quantity"))
                week_start = (start_dt + timedelta(weeks=pos - 1)).replace(tzinfo=None)
                rows.append({"interval_start_utc": week_start, "filling_mwh": qty})

    rows.sort(key=lambda r: r["interval_start_utc"])
    return rows


# ---------------------------------------------------------------------------
# Validate
# ---------------------------------------------------------------------------

def validate(rows: list[dict]) -> dict:
    if not rows:
        return {"status": "WARN", "messages": ["sin filas en este bloque"]}
    messages = []
    status = "OK"
    n_negative = sum(1 for r in rows if r["filling_mwh"] < 0)
    if n_negative:
        status = "WARN"
        messages.append(f"{n_negative} semanas con nivel NEGATIVO (inesperado)")
    n_out_of_band = sum(1 for r in rows if r["filling_mwh"] > FILLING_SANITY_MAX_MWH)
    if n_out_of_band:
        status = "WARN"
        messages.append(f"{n_out_of_band} semanas por encima de la banda de plausibilidad ({FILLING_SANITY_MAX_MWH:.0f} MWh)")
    return {"status": status, "messages": messages}


# ---------------------------------------------------------------------------
# Diccionario de datos
# ---------------------------------------------------------------------------

def _columns(area_key: str) -> dict:
    tz_name = AREAS[area_key]["tz"].key
    return {
        "interval_start_utc": ColumnDoc(
            description="Instante UTC de inicio de la semana de reporte.",
            source="timeInterval/start de cada <Period> + offset de (position-1) semanas, del XML de ENTSO-E (documentType A72).",
            dtype="TIMESTAMP", kind="identifier",
        ),
        "interval_start_local": ColumnDoc(
            description=f"Mismo instante en hora local del área ({tz_name}).",
            source="Calculado por el pipeline a partir de interval_start_utc.",
            dtype="TIMESTAMP", kind="timestamp",
        ),
        "filling_mwh": ColumnDoc(
            description="Contenido energético equivalente agregado de las reservas hidráulicas del país, en MWh (energía que se podría generar vaciando el embalse, no volumen de agua).",
            source="Elemento <quantity> de cada <Point> del XML de ENTSO-E.",
            dtype="DOUBLE", kind="continuous",
        ),
        "area_code": ColumnDoc(
            description="Código EIC del área ENTSO-E.",
            source="Constante de configuración (AREAS en el pipeline), no viene por fila en el XML.",
            dtype="VARCHAR", kind="categorical",
        ),
        "source_chunk": ColumnDoc(
            description="Identificador del bloque de petición de origen, para trazabilidad.",
            source="Parámetro de la petición a la API de ENTSO-E.",
            dtype="VARCHAR", kind="identifier",
        ),
        "ingested_at": ColumnDoc(
            description="Marca temporal UTC del momento en que esta fila se cargó (o recargó) por última vez en la base de datos.",
            source="datetime.utcnow() en el momento de la inserción, dentro de la función load().",
            dtype="TIMESTAMP", kind="timestamp",
        ),
    }


def document(con: duckdb.DuckDBPyConnection, area_key: str) -> None:
    catalog.refresh(con, "hydro_reservoir_filling", _columns(area_key))


# ---------------------------------------------------------------------------
# Schema + load
# ---------------------------------------------------------------------------

def ensure_schema(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS hydro_reservoir_filling (
            interval_start_utc    TIMESTAMP NOT NULL,
            interval_start_local  TIMESTAMP NOT NULL,
            filling_mwh           DOUBLE NOT NULL,
            area_code              VARCHAR NOT NULL,
            source_chunk            VARCHAR NOT NULL,
            ingested_at              TIMESTAMP NOT NULL,
            PRIMARY KEY (interval_start_utc)
        )
        """
    )


def load(con: duckdb.DuckDBPyConnection, area_key: str, rows: list[dict], source_chunk: str) -> int:
    ensure_schema(con)
    if not rows:
        return 0
    tz = AREAS[area_key]["tz"]
    eic = AREAS[area_key]["eic"]
    now = datetime.utcnow()
    records = [
        (
            r["interval_start_utc"],
            r["interval_start_utc"].replace(tzinfo=timezone.utc).astimezone(tz).replace(tzinfo=None),
            r["filling_mwh"], eic, source_chunk, now,
        )
        for r in rows
    ]
    df = pd.DataFrame(records, columns=["interval_start_utc", "interval_start_local", "filling_mwh", "area_code", "source_chunk", "ingested_at"])
    min_ts, max_ts = df["interval_start_utc"].min(), df["interval_start_utc"].max()
    con.execute("BEGIN TRANSACTION")
    try:
        con.execute(
            "DELETE FROM hydro_reservoir_filling WHERE interval_start_utc >= ? AND interval_start_utc <= ?",
            [min_ts, max_ts],
        )
        con.append("hydro_reservoir_filling", df)
        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK")
        raise
    return len(df)


# ---------------------------------------------------------------------------
# Orquestación + registro de auditoría
# ---------------------------------------------------------------------------

def _log_run(con, area_key, source_chunk, rows_loaded, status, message, started_at) -> None:
    con.execute(
        """
        INSERT INTO ingestion_log
            (source, target_table, delivery_date, rows_expected, rows_loaded, status, message, started_at, finished_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [f"entsoe_hydro_reservoir:{area_key}", "hydro_reservoir_filling", date.today(), None, rows_loaded, status,
         message, started_at, datetime.utcnow()],
    )


def run_for_range(con: duckdb.DuckDBPyConnection, area_key: str, period_start: str, period_end: str, logger) -> dict:
    started_at = datetime.utcnow()
    ensure_schema(con)
    eic = AREAS[area_key]["eic"]
    source_chunk = f"{period_start}-{period_end}"

    try:
        raw = fetch_raw(eic, period_start, period_end)
        rows = parse(raw)
    except Exception as exc:
        _log_run(con, area_key, source_chunk, 0, "ERROR", str(exc), started_at)
        logger.error("ENTSO-E hydro %s %s: %s", area_key, source_chunk, exc)
        return {"status": "ERROR", "message": str(exc)}

    validation = validate(rows)
    rows_loaded = load(con, area_key, rows, source_chunk)

    status = validation["status"]
    message = "; ".join(validation["messages"]) if validation["messages"] else "sin incidencias"
    _log_run(con, area_key, source_chunk, rows_loaded, status, message, started_at)
    log_fn = logger.info if status == "OK" else logger.warning
    log_fn("ENTSO-E hydro %s %s: %s filas cargadas — %s", area_key, source_chunk, rows_loaded, message)
    return {"status": status, "rows_loaded": rows_loaded, "message": message}


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(__file__).rsplit("etl", 1)[0])
    from etl.common.db import connect, document_ingestion_log
    from etl.common.logging_config import get_logger

    area = sys.argv[1] if len(sys.argv) > 1 else "spain"
    log = get_logger("entsoe_hydro_reservoir.manual", AREAS[area]["db"])
    conn = connect(AREAS[area]["db"])
    result = run_for_range(conn, area, "202301010000", "202302010000", log)
    print(result)
    document(conn, area)
    document_ingestion_log(conn)
    conn.close()
