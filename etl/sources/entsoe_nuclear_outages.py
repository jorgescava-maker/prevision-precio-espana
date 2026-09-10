"""
Ingesta de indisponibilidad de unidades nucleares (documentType A80, "Unavailability
of Generation Units" — unidades SIGNIFICATIVAS, ≥100MW, identificadas individualmente)
vía ENTSO-E Transparency Platform. Fase 6 del roadmap. Alcance: SOLO FRANCIA (decisión
con el usuario, 2026-08-27) — es donde vive prácticamente todo el parque nuclear de
este ecosistema (~56 reactores) y donde la señal es más rica; España y Países Bajos
tienen muy pocos eventos nucleares por cada cientos de documentos de otras tecnologías
que hay que descargar igualmente (sin filtro de tecnología en la API), mala relación
esfuerzo/valor.

Requiere ENTSOE_API_TOKEN en .env (mismo token que el resto de pipelines ENTSO-E).

Hechos verificados en vivo el 2026-08-27, críticos para un diseño correcto:

1. `documentType=A77` ("Unavailability of PRODUCTION Units") NO tiene datos nucleares
   — es para unidades no significativas. Los reactores nucleares (cada uno de cientos
   de MW) están bajo `documentType=A80` ("Unavailability of GENERATION Units",
   unidades significativas), verificado comparando ambos sobre el mismo mes de Francia.
2. `processType=A53` (planificado) y `A54` (forzado) devuelven EXACTAMENTE los mismos
   145 mRID en una prueba de una semana — el parámetro no filtra nada en la práctica
   para este documento. Se usa un único valor fijo (A53), sin pedir dos veces.
3. LÍMITE DURO de 200 documentos por petición — verificado con el mensaje de error
   real: "The number of instances (1388) exceeds the allowed maximum (200)". Francia
   ronda ~145 documentos/semana (TODAS las tecnologías, no solo nuclear — la API no
   permite filtrar por tecnología en la petición, hay que descargar todo y filtrar
   psrType=B14 después de recibirlo). `fetch_raw()` divide automáticamente la ventana
   por la mitad si se supera el límite, en vez de asumir un tamaño de chunk fijo.
4. Cada petición devuelve un ZIP con un XML por EVENTO de indisponibilidad (un mismo
   mRID = un evento, revisado a lo largo del tiempo — se vio un caso con
   revisionNumber=26). Las ventanas de evento pueden ser MUY largas (ej. el cierre
   permanente de Fessenheim 1: 2020-02-22 → 2099-12-31, la convención de ENTSO-E para
   "indefinido"). Se guarda solo la ÚLTIMA revisión vista por evento (upsert por
   event_mrid) — igual que las republicaciones de ficheros de OMIE.
5. En un muestreo de 12 semanas de 2024 no apareció NINGÚN evento nuclear con más de
   un <Point> en su Available_Period — cuando cambia la disponibilidad, ENTSO-E emite
   una revisión o un evento nuevo, no una curva dentro del mismo documento. Por eso
   este pipeline guarda UN valor de disponibilidad por evento (el mínimo si alguna vez
   apareciera más de un punto — el escenario más conservador — con
   n_points_collapsed>1 marcando el caso para que quede visible, no oculto).
"""

from __future__ import annotations

import io
import re
import time
import xml.etree.ElementTree as ET
import zipfile
from datetime import date, datetime, timezone

import duckdb
import pandas as pd
import requests

from etl.common import catalog
from etl.common.catalog import ColumnDoc
from etl.common.config import require_env

API_URL = "https://web-api.tp.entsoe.eu/api"
EIC_FRANCE = "10YFR-RTE------C"
NUCLEAR_PSR_TYPE = "B14"
MAX_DOCS_PER_REQUEST = 200  # límite duro de ENTSO-E, verificado en vivo
MIN_CHUNK_HOURS = 6  # tope de subdivisión recursiva, para no bucear infinito

_session = requests.Session()


class TooManyDocumentsError(Exception):
    pass


# ---------------------------------------------------------------------------
# Fetch (con subdivisión automática si se supera el límite de 200 documentos)
# ---------------------------------------------------------------------------

def _fetch_once(period_start: str, period_end: str, area_eic: str = EIC_FRANCE, timeout: int = 60, retries: int = 3) -> bytes | None:
    """Devuelve los bytes del ZIP, o None si no hay datos en ese rango.
    `area_eic` parametrizado (2026-08-30, ver etl/sources/entsoe_thermal_outages_spain.py)
    — por defecto Francia, para no tocar el comportamiento de las llamadas existentes."""
    params = {
        "securityToken": require_env("ENTSOE_API_TOKEN"),
        "documentType": "A80",
        "processType": "A53",
        "biddingZone_Domain": area_eic,
        "periodStart": period_start,
        "periodEnd": period_end,
    }
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            resp = _session.get(API_URL, params=params, timeout=timeout)
            # OJO: ENTSO-E devuelve el Acknowledgement de "supera el máximo de 200
            # documentos" con status HTTP 400 (no 200 como el resto de Acknowledgements
            # "sin datos" de este ecosistema) — hay que inspeccionar el CONTENIDO antes
            # de decidir si es un error real o un caso reconocible, no llamar
            # raise_for_status() a ciegas (bug propio detectado en el primer backfill:
            # 44/45 meses fallaban con "400 Client Error" sin llegar a ver el body).
            if resp.headers.get("content-type") == "application/zip":
                return resp.content
            if b"exceeds the allowed maximum" in resp.content:
                raise TooManyDocumentsError(period_start + "-" + period_end)
            if b"No matching data found" in resp.content:
                return None
            resp.raise_for_status()
            if b"Reason" in resp.content:
                return None
            raise ValueError(f"respuesta inesperada (ni zip ni Acknowledgement reconocible): {resp.text[:200]}")
        except TooManyDocumentsError:
            raise
        except (requests.RequestException, ValueError) as exc:
            last_exc = exc
            time.sleep(2 * attempt)
    raise last_exc  # type: ignore[misc]


def fetch_all(period_start_dt: datetime, period_end_dt: datetime, logger, area_eic: str = EIC_FRANCE) -> list[bytes]:
    """Recorre [period_start_dt, period_end_dt) subdividiendo cada tramo por la mitad
    cuando ENTSO-E responde que se ha superado el límite de 200 documentos. Devuelve
    la lista de ZIPs (bytes) de todos los tramos con datos."""
    span_hours = (period_end_dt - period_start_dt).total_seconds() / 3600
    ps = period_start_dt.strftime("%Y%m%d%H%M")
    pe = period_end_dt.strftime("%Y%m%d%H%M")
    try:
        content = _fetch_once(ps, pe, area_eic)
        return [content] if content else []
    except TooManyDocumentsError:
        if span_hours <= MIN_CHUNK_HOURS:
            raise RuntimeError(f"tramo {ps}-{pe} sigue excediendo 200 documentos incluso al mínimo de {MIN_CHUNK_HOURS}h")
        mid = period_start_dt + (period_end_dt - period_start_dt) / 2
        logger.info("Tramo %s-%s excede 200 documentos, subdividiendo en dos", ps, pe)
        return fetch_all(period_start_dt, mid, logger, area_eic) + fetch_all(mid, period_end_dt, logger, area_eic)


# ---------------------------------------------------------------------------
# Parse
# ---------------------------------------------------------------------------

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
    if psr_type != NUCLEAR_PSR_TYPE:
        return None

    points = ts.findall("Available_Period/Point")
    values = [float(p.findtext("quantity")) for p in points]
    available_mw = min(values) if values else None

    doc_status_el = root.find("docStatus/value")
    reason_el = root.find("Reason")

    return {
        "event_mrid": root.findtext("mRID"),
        "revision_number": int(root.findtext("revisionNumber")),
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

NUCLEAR_OUTAGE_COLUMNS = {
    "event_mrid": ColumnDoc(
        description="Identificador único del evento de indisponibilidad (estable a través de revisiones — un mismo evento se revisa, no se duplica).",
        source="Elemento <mRID> del <Unavailability_MarketDocument> del XML de ENTSO-E (documentType A80).",
        dtype="VARCHAR", kind="identifier",
    ),
    "revision_number": ColumnDoc(
        description="Número de revisión de este evento tal como se recibió — solo se guarda la ÚLTIMA revisión vista, esta columna es informativa/de auditoría, no parte de la clave.",
        source="Elemento <revisionNumber> del XML.",
        dtype="INTEGER", kind="continuous",
    ),
    "unit_resource_id": ColumnDoc(
        description="Identificador EIC del reactor físico (estable entre distintos eventos del mismo reactor a lo largo del tiempo).",
        source="Elemento <production_RegisteredResource.mRID> del XML.",
        dtype="VARCHAR", kind="categorical",
    ),
    "unit_name": ColumnDoc(
        description="Nombre del reactor, p.ej. 'FESSENHEIM 1'.",
        source="Elemento <production_RegisteredResource.name> del XML.",
        dtype="VARCHAR", kind="categorical",
    ),
    "nominal_capacity_mw": ColumnDoc(
        description="Potencia nominal del reactor, en MW.",
        source="Elemento <production_RegisteredResource.pSRType.powerSystemResources.nominalP> del XML.",
        dtype="DOUBLE", kind="continuous",
    ),
    "available_mw": ColumnDoc(
        description="Potencia disponible durante el evento, en MW (0 = parada total; un valor intermedio = derateo parcial). Si el evento tuviera más de un valor (ver n_points_collapsed), se guarda el MÍNIMO — el escenario más conservador.",
        source="Elemento <quantity> de <Available_Period/Point> del XML — mínimo si hay varios.",
        dtype="DOUBLE", kind="continuous",
    ),
    "unavailable_mw": ColumnDoc(
        description="Potencia NO disponible durante el evento, en MW (nominal_capacity_mw - available_mw). Columna derivada, calculada por el pipeline, no viene directamente del XML.",
        source="Calculado: nominal_capacity_mw - available_mw.",
        dtype="DOUBLE", kind="continuous",
    ),
    "n_points_collapsed": ColumnDoc(
        description="Número de <Point> que traía el Available_Period de este evento. Casi siempre 1 (verificado en un muestreo de 12 semanas de 2024, ningún evento nuclear tenía más de 1) — un valor >1 señala un caso con curva de disponibilidad variable dentro del mismo evento, simplificado al mínimo.",
        source="Recuento de <Available_Period/Point> del XML.",
        dtype="INTEGER", kind="continuous",
    ),
    "doc_status": ColumnDoc(
        description="Código de estado del documento tal como lo publica ENTSO-E (p.ej. 'A09' visto en el cierre permanente de Fessenheim). No se ha construido una tabla de traducción completa y verificada de todos los códigos posibles — se guarda el código crudo.",
        source="Elemento <docStatus><value> del XML.",
        dtype="VARCHAR", kind="categorical",
    ),
    "reason_code": ColumnDoc(
        description="Código de motivo de la indisponibilidad, según la codificación de ENTSO-E.",
        source="Elemento <Reason><code> del XML.",
        dtype="VARCHAR", kind="categorical",
    ),
    "reason_text": ColumnDoc(
        description="Texto libre explicando el motivo (a menudo referencias a comunicados REMIT/EDF).",
        source="Elemento <Reason><text> del XML.",
        dtype="VARCHAR", kind="categorical",
    ),
    "event_start_utc": ColumnDoc(
        description="Inicio UTC de la ventana de indisponibilidad de este evento.",
        source="Elemento <unavailability_Time_Period.timeInterval><start> del XML.",
        dtype="TIMESTAMP", kind="identifier",
    ),
    "event_end_utc": ColumnDoc(
        description="Fin UTC de la ventana de indisponibilidad. ENTSO-E usa 2099-12-31 como convención de 'indefinido/permanente' (ver ejemplo de Fessenheim en el docstring del pipeline) — no filtrar asumiendo que es una fecha real.",
        source="Elemento <unavailability_Time_Period.timeInterval><end> del XML.",
        dtype="TIMESTAMP", kind="identifier",
    ),
    "created_at_utc": ColumnDoc(
        description="Marca temporal UTC en que ENTSO-E publicó/actualizó ESTA revisión del evento (distinto de ingested_at, que es cuándo lo cargamos nosotros).",
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
        source="datetime.utcnow() en el momento de la inserción, dentro de la función load().",
        dtype="TIMESTAMP", kind="timestamp",
    ),
}


def document(con: duckdb.DuckDBPyConnection) -> None:
    catalog.refresh(con, "nuclear_outage_events", NUCLEAR_OUTAGE_COLUMNS)


# ---------------------------------------------------------------------------
# Schema + load
# ---------------------------------------------------------------------------

def ensure_schema(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS nuclear_outage_events (
            event_mrid            VARCHAR NOT NULL,
            revision_number       INTEGER NOT NULL,
            unit_resource_id      VARCHAR NOT NULL,
            unit_name              VARCHAR NOT NULL,
            nominal_capacity_mw    DOUBLE,
            available_mw            DOUBLE,
            unavailable_mw           DOUBLE,
            n_points_collapsed        INTEGER NOT NULL,
            doc_status                 VARCHAR,
            reason_code                 VARCHAR,
            reason_text                  VARCHAR,
            event_start_utc               TIMESTAMP NOT NULL,
            event_end_utc                  TIMESTAMP NOT NULL,
            created_at_utc                   TIMESTAMP NOT NULL,
            source_chunk                      VARCHAR NOT NULL,
            ingested_at                        TIMESTAMP NOT NULL,
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
            e["event_mrid"], e["revision_number"], e["unit_resource_id"], e["unit_name"],
            e["nominal_capacity_mw"], e["available_mw"],
            (e["nominal_capacity_mw"] - e["available_mw"]) if (e["nominal_capacity_mw"] is not None and e["available_mw"] is not None) else None,
            e["n_points_collapsed"], e["doc_status"], e["reason_code"], e["reason_text"],
            e["event_start_utc"], e["event_end_utc"], e["created_at_utc"], source_chunk, now,
        )
        for e in events
    ]
    df = pd.DataFrame(
        records,
        columns=["event_mrid", "revision_number", "unit_resource_id", "unit_name", "nominal_capacity_mw",
                 "available_mw", "unavailable_mw", "n_points_collapsed", "doc_status", "reason_code", "reason_text",
                 "event_start_utc", "event_end_utc", "created_at_utc", "source_chunk", "ingested_at"],
    )
    # Upsert por event_mrid: cada evento se identifica por su mRID estable a través de
    # revisiones — se reemplaza siempre por la última revisión recibida (mismo patrón
    # que las republicaciones de ficheros de OMIE).
    event_mrids = df["event_mrid"].tolist()
    con.execute("BEGIN TRANSACTION")
    try:
        con.executemany("DELETE FROM nuclear_outage_events WHERE event_mrid = ?", [(m,) for m in event_mrids])
        con.append("nuclear_outage_events", df)
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
        ["entsoe_nuclear_outages:france", "nuclear_outage_events", date.today(), None, rows_loaded, status, message,
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
        logger.error("Nuclear outages %s: %s", source_chunk, exc)
        return {"status": "ERROR", "message": str(exc)}

    rows_loaded = load(con, list(all_events.values()), source_chunk)
    status = "OK"
    message = f"sin incidencias, {len(zips)} tramos, {rows_loaded} eventos nucleares únicos"
    _log_run(con, source_chunk, rows_loaded, status, message, started_at)
    logger.info("Nuclear outages %s: %s", source_chunk, message)
    return {"status": status, "rows_loaded": rows_loaded, "message": message}


if __name__ == "__main__":
    sys_path_fix = __file__.rsplit("etl", 1)[0]
    import sys

    sys.path.insert(0, sys_path_fix)
    from etl.common.db import connect, document_ingestion_log
    from etl.common.logging_config import get_logger

    log = get_logger("entsoe_nuclear_outages.manual", "france")
    conn = connect("france")
    result = run_for_range(conn, datetime(2024, 1, 1), datetime(2024, 1, 8), log)
    print(result)
    document(conn)
    document_ingestion_log(conn)
    conn.close()
