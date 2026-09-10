"""
Ingesta de indisponibilidad de centrales térmicas fósiles francesas (documentType
A80, "Unavailability of Generation Units" — unidades SIGNIFICATIVAS, ≥100MW) vía
ENTSO-E Transparency Platform — dato requerido por el estudio C2 de la agenda de
investigación (opción real sobre el clean spark spread: "heat rate y disponibilidad
de central" para valorar la opcionalidad de encender/apagar una CCGT).

Hermano de `etl/sources/entsoe_nuclear_outages.py` (Fase 6 del roadmap): reutiliza
`fetch_all()` de ese módulo TAL CUAL, sin duplicarlo — es el mismo documentType A80
sobre la misma zona (Francia), y ENTSO-E no permite filtrar por tecnología en la
petición, así que cada llamada a `fetch_all()` ya trae TODAS las tecnologías
mezcladas (~145 documentos/semana, ver docstring de entsoe_nuclear_outages.py). El
pipeline de nucleares descarta silenciosamente todo lo que no sea B14; este módulo
hace la llamada de red POR SEPARADO (mismo coste que duplicar la descarga, no se ha
compartido una única pasada entre ambos pipelines por simplicidad — son procesos de
backfill/actualización independientes, no se ejecutan en la misma función) y se
queda con las tecnologías térmicas fósiles en su lugar.

Tecnologías incluidas (`THERMAL_PSR_TYPES`): B02 (lignito), B03 (gas derivado de
carbón), B04 (gas natural — la relevante para el "clean SPARK spread" del estudio
C2), B05 (hulla — relevante para el "dark spread"), B06 (fuel-oil). A diferencia de
`nuclear_outage_events` (una sola tecnología, sin necesidad de columna `psr_type`),
esta tabla mezcla varias — se añade `psr_type`/`psr_type_name` explícitos.

Alcance: SOLO FRANCIA, mismo criterio de esfuerzo/valor que la decisión de nucleares
(#42-49) — si se quiere ampliar a Alemania/Países Bajos/España, verificar en vivo
antes (podría no tener los mismos ~145 doc/semana, ni la misma proporción de
tecnologías térmicas significativas).
"""

from __future__ import annotations

import io
import re
import xml.etree.ElementTree as ET
import zipfile
from datetime import date, datetime

import duckdb
import pandas as pd

from etl.common import catalog
from etl.common.catalog import ColumnDoc
from etl.sources.entsoe_load_generation import PSR_TYPE_NAMES
from etl.sources.entsoe_nuclear_outages import fetch_all

THERMAL_PSR_TYPES = {"B02", "B03", "B04", "B05", "B06"}

TABLE = "thermal_outage_events"


def _strip_ns(elem: ET.Element) -> ET.Element:
    for el in elem.iter():
        if "}" in el.tag:
            el.tag = el.tag.split("}", 1)[1]
    return elem


def _parse_one_document(xml_text: str) -> dict | None:
    root = _strip_ns(ET.fromstring(xml_text))
    if root.tag != "Unavailability_MarketDocument":
        return None

    ts = root.find("TimeSeries")
    if ts is None:
        return None
    psr_type = ts.findtext("production_RegisteredResource.pSRType.psrType")
    if psr_type not in THERMAL_PSR_TYPES:
        return None

    points = ts.findall("Available_Period/Point")
    values = [float(p.findtext("quantity")) for p in points]
    available_mw = min(values) if values else None

    doc_status_el = root.find("docStatus/value")
    reason_el = root.find("Reason")

    return {
        "event_mrid": root.findtext("mRID"),
        "revision_number": int(root.findtext("revisionNumber")),
        "psr_type": psr_type,
        "psr_type_name": PSR_TYPE_NAMES.get(psr_type, psr_type),
        "unit_resource_id": ts.findtext("production_RegisteredResource.mRID"),
        "unit_name": ts.findtext("production_RegisteredResource.name"),
        "nominal_capacity_mw": float(ts.findtext("production_RegisteredResource.pSRType.powerSystemResources.nominalP") or "nan") or None,
        "available_mw": available_mw,
        "n_points_collapsed": len(values),
        "doc_status": doc_status_el.text if doc_status_el is not None else None,
        "reason_code": reason_el.findtext("code") if reason_el is not None else None,
        "reason_text": reason_el.findtext("text") if reason_el is not None else None,
        "event_start_utc": datetime.strptime(root.findtext("unavailability_Time_Period.timeInterval/start"), "%Y-%m-%dT%H:%MZ"),
        "event_end_utc": datetime.strptime(root.findtext("unavailability_Time_Period.timeInterval/end"), "%Y-%m-%dT%H:%MZ"),
        "created_at_utc": datetime.strptime(re.sub(r"\.\d+Z$", "Z", root.findtext("createdDateTime")), "%Y-%m-%dT%H:%M:%SZ"),
    }


def parse_zip(zip_bytes: bytes) -> list[dict]:
    events = []
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        for name in z.namelist():
            xml_text = z.read(name).decode("utf-8")
            event = _parse_one_document(xml_text)
            if event is not None:
                events.append(event)
    return events


# ---------------------------------------------------------------------------
# Diccionario de datos
# ---------------------------------------------------------------------------

THERMAL_OUTAGE_COLUMNS = {
    "event_mrid": ColumnDoc(
        description="Identificador único del evento de indisponibilidad (estable a través de revisiones).",
        source="Elemento <mRID> del <Unavailability_MarketDocument> del XML de ENTSO-E (documentType A80).",
        dtype="VARCHAR", kind="identifier",
    ),
    "revision_number": ColumnDoc(
        description="Número de revisión de este evento tal como se recibió — solo se guarda la ÚLTIMA revisión vista.",
        source="Elemento <revisionNumber> del XML.",
        dtype="INTEGER", kind="continuous",
    ),
    "psr_type": ColumnDoc(
        description="Código de tecnología ENTSO-E (PsrType): B02 lignito, B03 gas derivado de carbón, B04 gas natural, B05 hulla, B06 fuel-oil.",
        source="Elemento <production_RegisteredResource.pSRType.psrType> del XML.",
        dtype="VARCHAR", kind="categorical",
    ),
    "psr_type_name": ColumnDoc(
        description="Nombre legible del psr_type (misma tabla PSR_TYPE_NAMES que entsoe_load_generation.py).",
        source="Constante de configuración, no viene en el XML.",
        dtype="VARCHAR", kind="categorical",
    ),
    "unit_resource_id": ColumnDoc(
        description="Identificador EIC de la central física.",
        source="Elemento <production_RegisteredResource.mRID> del XML.",
        dtype="VARCHAR", kind="categorical",
    ),
    "unit_name": ColumnDoc(
        description="Nombre de la central.",
        source="Elemento <production_RegisteredResource.name> del XML.",
        dtype="VARCHAR", kind="categorical",
    ),
    "nominal_capacity_mw": ColumnDoc(
        description="Potencia nominal de la unidad, en MW.",
        source="Elemento <production_RegisteredResource.pSRType.powerSystemResources.nominalP> del XML.",
        dtype="DOUBLE", kind="continuous",
    ),
    "available_mw": ColumnDoc(
        description="Potencia disponible durante el evento, en MW (0 = parada total; valor intermedio = derateo parcial).",
        source="Elemento <quantity> de <Available_Period/Point> del XML — mínimo si hay varios.",
        dtype="DOUBLE", kind="continuous",
    ),
    "unavailable_mw": ColumnDoc(
        description="Potencia NO disponible durante el evento, en MW. Columna derivada, calculada por el pipeline.",
        source="Calculado: nominal_capacity_mw - available_mw.",
        dtype="DOUBLE", kind="continuous",
    ),
    "n_points_collapsed": ColumnDoc(
        description="Número de <Point> que traía el Available_Period de este evento — casi siempre 1 (ver misma nota en nuclear_outage_events).",
        source="Recuento de <Available_Period/Point> del XML.",
        dtype="INTEGER", kind="continuous",
    ),
    "doc_status": ColumnDoc(
        description="Código de estado del documento tal como lo publica ENTSO-E.",
        source="Elemento <docStatus><value> del XML.",
        dtype="VARCHAR", kind="categorical",
    ),
    "reason_code": ColumnDoc(
        description="Código de motivo de la indisponibilidad.",
        source="Elemento <Reason><code> del XML.",
        dtype="VARCHAR", kind="categorical",
    ),
    "reason_text": ColumnDoc(
        description="Texto libre explicando el motivo.",
        source="Elemento <Reason><text> del XML.",
        dtype="VARCHAR", kind="categorical",
    ),
    "event_start_utc": ColumnDoc(
        description="Inicio UTC de la ventana de indisponibilidad de este evento.",
        source="Elemento <unavailability_Time_Period.timeInterval><start> del XML.",
        dtype="TIMESTAMP", kind="identifier",
    ),
    "event_end_utc": ColumnDoc(
        description="Fin UTC de la ventana de indisponibilidad. ENTSO-E usa 2099-12-31 como convención de 'indefinido/permanente'.",
        source="Elemento <unavailability_Time_Period.timeInterval><end> del XML.",
        dtype="TIMESTAMP", kind="identifier",
    ),
    "created_at_utc": ColumnDoc(
        description="Marca temporal UTC en que ENTSO-E publicó/actualizó ESTA revisión del evento.",
        source="Elemento <createdDateTime> del XML.",
        dtype="TIMESTAMP", kind="timestamp",
    ),
    "source_chunk": ColumnDoc(
        description="Identificador del tramo de petición de origen, para trazabilidad.",
        source="Rango pedido a la API en la ejecución que trajo esta fila.",
        dtype="VARCHAR", kind="identifier",
    ),
    "ingested_at": ColumnDoc(
        description="Marca temporal UTC del momento en que esta fila se cargó (o recargó) por última vez en la base de datos.",
        source="datetime.utcnow() en el momento de la inserción.",
        dtype="TIMESTAMP", kind="timestamp",
    ),
}


def document(con: duckdb.DuckDBPyConnection) -> None:
    catalog.refresh(con, TABLE, THERMAL_OUTAGE_COLUMNS)


# ---------------------------------------------------------------------------
# Schema + load
# ---------------------------------------------------------------------------

def ensure_schema(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {TABLE} (
            event_mrid            VARCHAR NOT NULL,
            revision_number       INTEGER NOT NULL,
            psr_type              VARCHAR NOT NULL,
            psr_type_name         VARCHAR NOT NULL,
            unit_resource_id      VARCHAR NOT NULL,
            unit_name             VARCHAR NOT NULL,
            nominal_capacity_mw   DOUBLE,
            available_mw          DOUBLE,
            unavailable_mw        DOUBLE,
            n_points_collapsed    INTEGER NOT NULL,
            doc_status            VARCHAR,
            reason_code           VARCHAR,
            reason_text           VARCHAR,
            event_start_utc       TIMESTAMP NOT NULL,
            event_end_utc         TIMESTAMP NOT NULL,
            created_at_utc        TIMESTAMP NOT NULL,
            source_chunk          VARCHAR NOT NULL,
            ingested_at           TIMESTAMP NOT NULL,
            PRIMARY KEY (event_mrid)
        )
        """
    )


def load(con: duckdb.DuckDBPyConnection, events: list[dict], source_chunk: str) -> int:
    ensure_schema(con)
    if not events:
        return 0
    now = datetime.utcnow()
    records = [
        (
            e["event_mrid"], e["revision_number"], e["psr_type"], e["psr_type_name"],
            e["unit_resource_id"], e["unit_name"],
            e["nominal_capacity_mw"], e["available_mw"],
            (e["nominal_capacity_mw"] - e["available_mw"]) if (e["nominal_capacity_mw"] is not None and e["available_mw"] is not None) else None,
            e["n_points_collapsed"], e["doc_status"], e["reason_code"], e["reason_text"],
            e["event_start_utc"], e["event_end_utc"], e["created_at_utc"], source_chunk, now,
        )
        for e in events
    ]
    df = pd.DataFrame(
        records,
        columns=["event_mrid", "revision_number", "psr_type", "psr_type_name", "unit_resource_id", "unit_name",
                 "nominal_capacity_mw", "available_mw", "unavailable_mw", "n_points_collapsed", "doc_status",
                 "reason_code", "reason_text", "event_start_utc", "event_end_utc", "created_at_utc",
                 "source_chunk", "ingested_at"],
    )
    event_mrids = df["event_mrid"].tolist()
    con.execute("BEGIN TRANSACTION")
    try:
        con.executemany(f"DELETE FROM {TABLE} WHERE event_mrid = ?", [(m,) for m in event_mrids])
        con.append(TABLE, df)
        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK")
        raise
    return len(df)


# ---------------------------------------------------------------------------
# Orquestación + registro de auditoría
# ---------------------------------------------------------------------------

def _log_run(con, source_chunk, rows_loaded, status, message, started_at) -> None:
    con.execute(
        """
        INSERT INTO ingestion_log
            (source, target_table, delivery_date, rows_expected, rows_loaded, status, message, started_at, finished_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        ["entsoe_thermal_outages:france", TABLE, date.today(), None, rows_loaded, status, message,
         started_at, datetime.utcnow()],
    )


def run_for_range(con: duckdb.DuckDBPyConnection, period_start_dt: datetime, period_end_dt: datetime, logger) -> dict:
    started_at = datetime.utcnow()
    ensure_schema(con)
    source_chunk = f"{period_start_dt:%Y%m%d%H%M}-{period_end_dt:%Y%m%d%H%M}"

    try:
        zips = fetch_all(period_start_dt, period_end_dt, logger)
        all_events: dict[str, dict] = {}
        for zb in zips:
            for event in parse_zip(zb):
                existing = all_events.get(event["event_mrid"])
                if existing is None or event["revision_number"] >= existing["revision_number"]:
                    all_events[event["event_mrid"]] = event
    except Exception as exc:
        _log_run(con, source_chunk, 0, "ERROR", str(exc), started_at)
        logger.error("Thermal outages %s: %s", source_chunk, exc)
        return {"status": "ERROR", "message": str(exc)}

    rows_loaded = load(con, list(all_events.values()), source_chunk)
    status = "OK"
    message = f"sin incidencias, {len(zips)} tramos, {rows_loaded} eventos térmicos únicos"
    _log_run(con, source_chunk, rows_loaded, status, message, started_at)
    logger.info("Thermal outages %s: %s", source_chunk, message)
    return {"status": status, "rows_loaded": rows_loaded, "message": message}


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(__file__).rsplit("etl", 1)[0])
    from etl.common.db import connect, document_ingestion_log
    from etl.common.logging_config import get_logger

    log = get_logger("entsoe_thermal_outages.manual", "france")
    conn = connect("france")
    result = run_for_range(conn, datetime(2024, 1, 1), datetime(2024, 1, 8), log)
    print(result)
    document(conn)
    document_ingestion_log(conn)
    conn.close()
