"""
Ingesta de capacidad instalada de generación por tecnología (documentType A68,
"Installed Generation Capacity Aggregated", processType A33 "Year ahead") vía ENTSO-E
Transparency Platform, para Alemania, Francia, Países Bajos y España — dato requerido
por el estudio A1 de la agenda de investigación (índice de canibalización solar/eólica,
necesita "capacidad instalada acumulada por trimestre" para separar el efecto
estructural del coyuntural en el value factor).

Hechos verificados en vivo el 2026-08-28:
1. Es una serie ANUAL, no horaria: cada `<TimeSeries>` trae un único `<Point>` con
   `resolution=P1Y` — no aplica el parseo de intervalos intradía de
   `etl/common/entsoe_xml.py` (pensado para PT15M/PT30M/PT60M), así que este módulo
   tiene su propio parseo, mucho más simple: un valor por (año, tecnología).
2. Funciona para los 4 países con el EIC de zona de oferta nacional — incluida
   España (`10YES-REE------0`), aunque el resto de datos españoles del proyecto
   vienen de e·sios/OMIE en vez de ENTSO-E: para capacidad instalada no hay motivo
   para preferir otra fuente, y usar ENTSO-E aquí evita tener que buscar un indicador
   de e·sios equivalente. 15-20 `<TimeSeries>` (tecnologías) por país y año.
3. No hay transición de resolución que vigilar (siempre P1Y) ni forward-fill que
   aplicar (un único punto por serie) — el módulo es deliberadamente más simple que
   el resto de fuentes ENTSO-E del proyecto.
4. La convención de fecha es la misma que el resto de pipelines anuales del proyecto
   (`entsoe_load_generation.py`, etc.): `periodStart=f"{year}01010000"`,
   `periodEnd=f"{year+1}01010000"` — ENTSO-E devuelve el documento del año natural que
   solapa con el rango pedido, sin necesitar ningún ajuste de huso horario especial
   pese a ser un dato "de fin de año" (verificado en vivo).
"""

from __future__ import annotations

from datetime import date, datetime
from zoneinfo import ZoneInfo

import duckdb
import pandas as pd
import xml.etree.ElementTree as ET

from etl.common import catalog
from etl.common.catalog import ColumnDoc
from etl.common.entsoe_xml import strip_ns
from etl.sources.entsoe_load_generation import PSR_TYPE_NAMES, fetch_raw

AREAS = {
    "germany": {"eic": "10Y1001A1001A82H", "db": "germany"},
    "france": {"eic": "10YFR-RTE------C", "db": "france"},
    "netherlands": {"eic": "10YNL----------L", "db": "netherlands"},
    "spain": {"eic": "10YES-REE------0", "db": "spain"},
}

TABLE = "entsoe_installed_capacity"


# ---------------------------------------------------------------------------
# Parse
# ---------------------------------------------------------------------------

def parse_capacity(xml_text: str) -> list[dict]:
    """Devuelve [{'psr_type':..., 'capacity_mw':...}] — un valor por tecnología, sin
    dimensión temporal intra-documento (el año lo determina el propio periodStart/
    periodEnd de la petición, no hace falta leerlo de vuelta del XML)."""
    root = strip_ns(ET.fromstring(xml_text))
    if root.tag == "Acknowledgement_MarketDocument":
        reason = root.findtext(".//Reason/text") or "sin detalle"
        raise ValueError(f"ENTSO-E devolvió un Acknowledgement (sin datos): {reason}")

    rows = []
    for ts in root.findall("TimeSeries"):
        psr_type = ts.findtext("MktPSRType/psrType")
        quantity = ts.findtext("Period/Point/quantity")
        if psr_type is None or quantity is None:
            continue
        rows.append({"psr_type": psr_type, "capacity_mw": float(quantity)})
    return rows


# ---------------------------------------------------------------------------
# Diccionario de datos
# ---------------------------------------------------------------------------

def document(con: duckdb.DuckDBPyConnection) -> None:
    cols = {
        "capacity_year": ColumnDoc(
            description="Año natural al que se refiere la capacidad instalada (dato de 'year ahead', fijado a final del año anterior).",
            source="Parámetro periodStart/periodEnd de la petición a ENTSO-E, no un campo del XML.",
            dtype="SMALLINT", kind="categorical",
        ),
        "psr_type": ColumnDoc(
            description="Código de tecnología ENTSO-E (PsrType) — mismo catálogo que entsoe_generation_by_type/entsoe_generation_forecast.",
            source="Elemento <psrType> de cada <TimeSeries> del XML de ENTSO-E (documentType A68).",
            dtype="VARCHAR", kind="categorical",
        ),
        "psr_type_name": ColumnDoc(
            description="Nombre legible del psr_type (misma tabla PSR_TYPE_NAMES que entsoe_load_generation.py).",
            source="Constante de configuración, no viene en el XML.",
            dtype="VARCHAR", kind="categorical",
        ),
        "capacity_mw": ColumnDoc(
            description="Capacidad instalada de esa tecnología a esa fecha, en MW.",
            source="Elemento <quantity> del único <Point> de cada <TimeSeries>.",
            dtype="DOUBLE", kind="continuous",
        ),
        "area_code": ColumnDoc(
            description="Código EIC del área ENTSO-E.",
            source="Constante de configuración (AREAS en este módulo).",
            dtype="VARCHAR", kind="categorical",
        ),
        "ingested_at": ColumnDoc(
            description="Marca temporal UTC de la última carga/recarga de esta fila.",
            source="datetime.utcnow() en el momento de la inserción.",
            dtype="TIMESTAMP", kind="timestamp",
        ),
    }
    catalog.refresh(con, TABLE, cols)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

def ensure_schema(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {TABLE} (
            capacity_year   SMALLINT NOT NULL,
            psr_type        VARCHAR NOT NULL,
            psr_type_name   VARCHAR NOT NULL,
            capacity_mw     DOUBLE NOT NULL,
            area_code       VARCHAR NOT NULL,
            ingested_at     TIMESTAMP NOT NULL,
            PRIMARY KEY (capacity_year, psr_type)
        )
        """
    )


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def _log_run(con, source, rows_loaded, status, message, started_at) -> None:
    con.execute(
        """
        INSERT INTO ingestion_log
            (source, target_table, delivery_date, rows_expected, rows_loaded, status, message, started_at, finished_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [source, TABLE, date.today(), None, rows_loaded, status, message, started_at, datetime.utcnow()],
    )


def run_for_year(con: duckdb.DuckDBPyConnection, area_key: str, year: int, logger) -> dict:
    started_at = datetime.utcnow()
    ensure_schema(con)
    eic = AREAS[area_key]["eic"]
    source = f"entsoe_installed_capacity:{area_key}:{year}"

    try:
        raw = fetch_raw({"documentType": "A68", "processType": "A33", "in_Domain": eic, "periodStart": f"{year}01010000", "periodEnd": f"{year + 1}01010000"})
        rows = parse_capacity(raw)
    except Exception as exc:
        _log_run(con, source, 0, "ERROR", str(exc), started_at)
        logger.error("%s: %s", source, exc)
        return {"status": "ERROR", "message": str(exc)}

    now = datetime.utcnow()
    status = "OK"
    message = f"sin incidencias, {len(rows)} tecnologías"
    rows_loaded = 0
    if not rows:
        status = "WARN"
        message = "sin filas"
    else:
        df = pd.DataFrame(
            [(year, r["psr_type"], PSR_TYPE_NAMES.get(r["psr_type"], r["psr_type"]), r["capacity_mw"], eic, now) for r in rows],
            columns=["capacity_year", "psr_type", "psr_type_name", "capacity_mw", "area_code", "ingested_at"],
        )
        con.execute("BEGIN TRANSACTION")
        try:
            con.execute(f"DELETE FROM {TABLE} WHERE capacity_year = ?", [year])
            con.append(TABLE, df)
            con.execute("COMMIT")
        except Exception:
            con.execute("ROLLBACK")
            raise
        rows_loaded = len(df)

    _log_run(con, source, rows_loaded, status, message, started_at)
    log_fn = logger.info if status == "OK" else logger.warning
    log_fn("%s: %s filas cargadas — %s", source, rows_loaded, message)
    return {"status": status, "rows_loaded": rows_loaded, "message": message}


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(__file__).rsplit("etl", 1)[0])
    from etl.common.db import connect, document_ingestion_log
    from etl.common.logging_config import get_logger

    area = sys.argv[1] if len(sys.argv) > 1 else "france"
    log = get_logger("entsoe_installed_capacity.manual", AREAS[area]["db"])
    conn = connect(AREAS[area]["db"])
    print(run_for_year(conn, area, 2023, log))
    document(conn)
    document_ingestion_log(conn)
    conn.close()
