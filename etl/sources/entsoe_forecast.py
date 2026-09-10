"""
Ingesta de previsión día-adelantada (D-1) de demanda (documentType A65, processType
A01, "Day-ahead Total Load Forecast") y de generación eólica+solar (documentType A69,
processType A01, "Generation Forecasts for Wind and Solar") vía ENTSO-E Transparency
Platform, para Alemania, Francia y Países Bajos (mismas áreas EIC que
etl/sources/entsoe_load_generation.py, reutilizado directamente).

Es el insumo de la Fase 8 del roadmap (previsiones D-1) para estos tres países — la
contraparte de etl/sources/entsoe_load_generation.py (que trae el dato REAL, no la
previsión). Sirve al estudio B1 (comparar `*_forecast_mw` de aquí contra `load_mw` /
`generation_mw` de entsoe_load_generation por intervalo, calculando el error de
previsión).

Hechos verificados en vivo el 2026-08-28:
1. Ambos documentos existen y responden para DE/FR/NL, con historial hasta al menos
   2023-01-01 (verificado en 2023-01-01 y en fechas recientes).
2. A diferencia del dato real, aquí NO hay que elegir la resolución por fecha: el XML
   ya trae PT60M para periodos antiguos y PT15M para recientes, igual que el dato real
   — mismo corte ~2025-10-01. Se reutiliza tal cual el parseo de
   etl/common/entsoe_xml.py (mismo forward-fill + deduplicación).
3. El documentType A69 devuelve un <TimeSeries> por tecnología (psrType): B16 (solar),
   B19 (eólica terrestre), y B18 (eólica marina) SOLO si el país tiene capacidad
   offshore en esa fecha — Francia no lo tenía en 2023 (2 TimeSeries: B16+B19) pero sí
   en 2026 (3 TimeSeries: B16+B18+B19), verificado en vivo. No se puede asumir un
   conjunto fijo de tecnologías por país; se registra lo que venga cada vez.
4. Es processType=A01 ("day ahead") por definición: cada intervalo trae UN solo valor
   de previsión (la vigente el día antes de la entrega), no varias revisiones con
   distinto lead time — a diferencia de un feed intradiario que ENTSO-E también
   publica por separado (processType A18, no usado aquí; ver alcance más abajo). Es
   directamente la previsión D-1 que necesita el estudio B1, sin transformación.
5. Alcance decidido con el usuario el 2026-08-28: esta fase cubre Europa (ENTSO-E +
   e·sios, ver etl/sources/esios_forecast.py); NEM (AEMO) queda aparcado porque su
   previsión (PREDISPATCH/ST PASA) no es un documento único D-1 sino un feed que se
   revisa muchas veces antes de la entrega — requeriría decidir una convención de
   "lead time" en vez de reutilizar el patrón de esta fuente. Ver docs/status.md.
"""

from __future__ import annotations

from datetime import date, datetime, timezone

import duckdb
import pandas as pd

from etl.common import catalog, entsoe_xml
from etl.common.catalog import ColumnDoc
from etl.sources.entsoe_load_generation import AREAS, PSR_TYPE_NAMES, fetch_raw


def _psr_type_key(ts) -> str:
    return ts.findtext("MktPSRType/psrType") or ts.findtext("psrType")


# ---------------------------------------------------------------------------
# Diccionario de datos
# ---------------------------------------------------------------------------

def _base_columns(area_key: str, value_name: str, value_desc: str, extra: dict | None = None) -> dict:
    tz_name = AREAS[area_key]["tz"].key
    cols = {
        "interval_start_utc": ColumnDoc(
            description="Instante UTC de inicio del intervalo previsto.",
            source="timeInterval/start de cada <Period> + offset de (position-1) × resolución, del XML de ENTSO-E.",
            dtype="TIMESTAMP", kind="identifier",
        ),
        "interval_start_local": ColumnDoc(
            description=f"Mismo instante en hora local del área ({tz_name}, con horario de verano).",
            source="Calculado por el pipeline a partir de interval_start_utc.",
            dtype="TIMESTAMP", kind="timestamp",
        ),
        "resolution_minutes": ColumnDoc(
            description="Duración real del intervalo: 60 hasta el 2025-09-30, 15 desde el 2025-10-01 (mismo corte que el dato real, verificado también para la previsión).",
            source="Elemento <resolution> del <Period> correspondiente del XML de ENTSO-E.",
            dtype="SMALLINT", kind="categorical",
        ),
        value_name: ColumnDoc(description=value_desc, source="Elemento <quantity> de cada <Point>, con forward-fill de posiciones omitidas.", dtype="DOUBLE", kind="continuous"),
        "area_code": ColumnDoc(
            description="Código EIC del área ENTSO-E.",
            source="Constante de configuración (AREAS en entsoe_load_generation.py).",
            dtype="VARCHAR", kind="categorical",
        ),
        "source_chunk": ColumnDoc(
            description="Identificador del bloque de petición de origen, para trazabilidad.",
            source="Parámetro de la petición a la API de ENTSO-E.",
            dtype="VARCHAR", kind="identifier",
        ),
        "ingested_at": ColumnDoc(
            description="Marca temporal UTC de la última carga/recarga de esta fila.",
            source="datetime.utcnow() en el momento de la inserción.",
            dtype="TIMESTAMP", kind="timestamp",
        ),
    }
    if extra:
        cols.update(extra)
    return cols


def document_load_forecast(con: duckdb.DuckDBPyConnection, area_key: str) -> None:
    cols = _base_columns(
        area_key, "load_forecast_mw",
        "Previsión día-adelantada (D-1) de demanda (carga) del sistema en el área, en MW.",
    )
    catalog.refresh(con, "entsoe_load_forecast", cols)


def document_generation_forecast(con: duckdb.DuckDBPyConnection, area_key: str) -> None:
    cols = _base_columns(
        area_key, "generation_forecast_mw",
        "Previsión día-adelantada (D-1) de generación eólica/solar de esa tecnología en el área, en MW.",
        extra={
            "psr_type": ColumnDoc(
                description="Código de tecnología ENTSO-E (PsrType): B16=Solar, B18=Eólica marina, B19=Eólica terrestre. B18 solo presente si el área tenía capacidad offshore en esa fecha (ver docstring del módulo).",
                source="Elemento <psrType> de cada <TimeSeries> del XML de ENTSO-E.",
                dtype="VARCHAR", kind="categorical",
            ),
            "psr_type_name": ColumnDoc(
                description="Nombre legible del psr_type (misma tabla PSR_TYPE_NAMES que entsoe_load_generation.py).",
                source="Constante de configuración, no viene en el XML.",
                dtype="VARCHAR", kind="categorical",
            ),
        },
    )
    catalog.refresh(con, "entsoe_generation_forecast", cols)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

def ensure_schema(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS entsoe_load_forecast (
            interval_start_utc    TIMESTAMP NOT NULL,
            interval_start_local  TIMESTAMP NOT NULL,
            resolution_minutes    SMALLINT NOT NULL,
            load_forecast_mw      DOUBLE NOT NULL,
            area_code             VARCHAR NOT NULL,
            source_chunk          VARCHAR NOT NULL,
            ingested_at           TIMESTAMP NOT NULL,
            PRIMARY KEY (interval_start_utc)
        )
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS entsoe_generation_forecast (
            interval_start_utc     TIMESTAMP NOT NULL,
            psr_type                VARCHAR NOT NULL,
            interval_start_local    TIMESTAMP NOT NULL,
            resolution_minutes      SMALLINT NOT NULL,
            generation_forecast_mw  DOUBLE NOT NULL,
            psr_type_name           VARCHAR NOT NULL,
            area_code                VARCHAR NOT NULL,
            source_chunk             VARCHAR NOT NULL,
            ingested_at              TIMESTAMP NOT NULL,
            PRIMARY KEY (interval_start_utc, psr_type)
        )
        """
    )


# ---------------------------------------------------------------------------
# Load (log)
# ---------------------------------------------------------------------------

def _log_run(con, source, target_table, rows_loaded, status, message, started_at) -> None:
    con.execute(
        """
        INSERT INTO ingestion_log
            (source, target_table, delivery_date, rows_expected, rows_loaded, status, message, started_at, finished_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [source, target_table, date.today(), None, rows_loaded, status, message, started_at, datetime.utcnow()],
    )


# ---------------------------------------------------------------------------
# Previsión de demanda
# ---------------------------------------------------------------------------

def run_load_forecast_for_range(con: duckdb.DuckDBPyConnection, area_key: str, period_start: str, period_end: str, logger) -> dict:
    started_at = datetime.utcnow()
    ensure_schema(con)
    eic = AREAS[area_key]["eic"]
    tz = AREAS[area_key]["tz"]
    source_chunk = f"{period_start}-{period_end}"
    source = f"entsoe_load_forecast:{area_key}"

    try:
        raw = fetch_raw({"documentType": "A65", "processType": "A01", "outBiddingZone_Domain": eic, "periodStart": period_start, "periodEnd": period_end})
        grouped = entsoe_xml.parse_periods(raw)
        rows, parse_stats = grouped.get("default", ([], {"n_exact_duplicates": 0, "n_revised": 0}))
    except Exception as exc:
        _log_run(con, source, "entsoe_load_forecast", 0, "ERROR", str(exc), started_at)
        logger.error("%s %s: %s", source, source_chunk, exc)
        return {"status": "ERROR", "message": str(exc)}

    messages = []
    status = "OK"
    if parse_stats["n_revised"]:
        status = "WARN"
        messages.append(f"{parse_stats['n_revised']} intervalos revisados (valor distinto en dos TimeSeries)")
    if not rows:
        status = "WARN"
        messages.append("sin filas en este bloque")

    now = datetime.utcnow()
    rows_loaded = 0
    if rows:
        df = pd.DataFrame(
            [
                (r["interval_start_utc"], r["interval_start_utc"].replace(tzinfo=timezone.utc).astimezone(tz).replace(tzinfo=None),
                 r["resolution_minutes"], r["value"], eic, source_chunk, now)
                for r in rows
            ],
            columns=["interval_start_utc", "interval_start_local", "resolution_minutes", "load_forecast_mw", "area_code", "source_chunk", "ingested_at"],
        )
        min_ts, max_ts = df["interval_start_utc"].min(), df["interval_start_utc"].max()
        con.execute("BEGIN TRANSACTION")
        try:
            con.execute("DELETE FROM entsoe_load_forecast WHERE interval_start_utc >= ? AND interval_start_utc <= ?", [min_ts, max_ts])
            con.append("entsoe_load_forecast", df)
            con.execute("COMMIT")
        except Exception:
            con.execute("ROLLBACK")
            raise
        rows_loaded = len(df)

    message = "; ".join(messages) if messages else "sin incidencias"
    _log_run(con, source, "entsoe_load_forecast", rows_loaded, status, message, started_at)
    log_fn = logger.info if status == "OK" else logger.warning
    log_fn("%s %s: %s filas cargadas — %s", source, source_chunk, rows_loaded, message)
    return {"status": status, "rows_loaded": rows_loaded, "message": message}


# ---------------------------------------------------------------------------
# Previsión de generación eólica+solar
# ---------------------------------------------------------------------------

def run_generation_forecast_for_range(con: duckdb.DuckDBPyConnection, area_key: str, period_start: str, period_end: str, logger) -> dict:
    started_at = datetime.utcnow()
    ensure_schema(con)
    eic = AREAS[area_key]["eic"]
    tz = AREAS[area_key]["tz"]
    source_chunk = f"{period_start}-{period_end}"
    source = f"entsoe_generation_forecast:{area_key}"

    try:
        raw = fetch_raw({"documentType": "A69", "processType": "A01", "in_Domain": eic, "periodStart": period_start, "periodEnd": period_end})
        grouped = entsoe_xml.parse_periods(raw, series_key_fn=_psr_type_key)
    except Exception as exc:
        _log_run(con, source, "entsoe_generation_forecast", 0, "ERROR", str(exc), started_at)
        logger.error("%s %s: %s", source, source_chunk, exc)
        return {"status": "ERROR", "message": str(exc)}

    now = datetime.utcnow()
    all_records = []
    n_revised_total = 0
    for psr_type, (rows, parse_stats) in grouped.items():
        if psr_type is None or not rows:
            continue
        n_revised_total += parse_stats["n_revised"]
        for r in rows:
            all_records.append(
                (
                    r["interval_start_utc"], psr_type,
                    r["interval_start_utc"].replace(tzinfo=timezone.utc).astimezone(tz).replace(tzinfo=None),
                    r["resolution_minutes"], r["value"], PSR_TYPE_NAMES.get(psr_type, psr_type), eic, source_chunk, now,
                )
            )

    status = "OK"
    messages = []
    if n_revised_total:
        status = "WARN"
        messages.append(f"{n_revised_total} intervalos revisados en total")
    if not all_records:
        status = "WARN"
        messages.append("sin filas en este bloque")

    rows_loaded = 0
    if all_records:
        df = pd.DataFrame(
            all_records,
            columns=["interval_start_utc", "psr_type", "interval_start_local", "resolution_minutes", "generation_forecast_mw", "psr_type_name", "area_code", "source_chunk", "ingested_at"],
        )
        min_ts, max_ts = df["interval_start_utc"].min(), df["interval_start_utc"].max()
        con.execute("BEGIN TRANSACTION")
        try:
            con.execute(
                "DELETE FROM entsoe_generation_forecast WHERE interval_start_utc >= ? AND interval_start_utc <= ?",
                [min_ts, max_ts],
            )
            con.append("entsoe_generation_forecast", df)
            con.execute("COMMIT")
        except Exception:
            con.execute("ROLLBACK")
            raise
        rows_loaded = len(df)

    message = "; ".join(messages) if messages else f"sin incidencias, {len(grouped)} tecnologías"
    _log_run(con, source, "entsoe_generation_forecast", rows_loaded, status, message, started_at)
    log_fn = logger.info if status == "OK" else logger.warning
    log_fn("%s %s: %s filas cargadas (%d tecnologías) — %s", source, source_chunk, rows_loaded, len(grouped), message)
    return {"status": status, "rows_loaded": rows_loaded, "message": message}


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(__file__).rsplit("etl", 1)[0])
    from etl.common.db import connect, document_ingestion_log
    from etl.common.logging_config import get_logger

    area = sys.argv[1] if len(sys.argv) > 1 else "france"
    log = get_logger("entsoe_forecast.manual", AREAS[area]["db"])
    conn = connect(AREAS[area]["db"])
    print(run_load_forecast_for_range(conn, area, "202301010000", "202301080000", log))
    print(run_generation_forecast_for_range(conn, area, "202301010000", "202301080000", log))
    document_load_forecast(conn, area)
    document_generation_forecast(conn, area)
    document_ingestion_log(conn)
    conn.close()
