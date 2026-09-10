"""
Ingesta de capacidad instalada POR CENTRAL individual (documentType A71,
"Installed Capacity per Generation Unit", processType A33 "Year ahead") vía
ENTSO-E Transparency Platform — registro real de unidades de generación con
nombre, EIC y capacidad nominal en MW, a diferencia de `entsoe_installed_capacity`
(documentType A68), que solo da el TOTAL agregado por tecnología. Motivado por el
simulador de merit order (`studies/merit_order/`, ver findings.md #85/RESULTS.md
"Fase 0b"): sustituye la aproximación sintética de repartir la capacidad total en
tramos iguales de heat rate por la distribución REAL de tamaños de central.

Alcance: España desde 2026-08-30 y **Alemania desde 2026-09-03**. La decisión
original era "solo España" porque el único estudio que lo necesitaba era el merit
order español; Alemania entra al preguntarse por la estructura de su mercado —
cuántas unidades hay, de qué tamaño y qué concentración tienen (findings.md #136).

Requiere ENTSOE_API_TOKEN en .env (mismo token que el resto de pipelines ENTSO-E).

Hechos verificados en vivo el 2026-08-30, críticos para un diseño correcto:

1. El XML trae un <TimeSeries> por (central, año) — NO un único snapshot: pedir
   un rango de UN año natural (`periodStart=f"{year}01010000"`,
   `periodEnd=f"{year+1}01010000"`, el mismo patrón que `entsoe_installed_capacity`)
   devuelve DOS <Period> por central, uno para el año pedido y otro "de propina"
   para el año siguiente — por el desfase de huso horario (el `timeInterval.start`
   de un año natural completo siempre cae en "AAAA-12-31T23:00:00Z", medianoche
   local Europe/Madrid del 1 de enero). Regla verificada: el año natural que
   representa un `<Period>` es SIEMPRE `timeInterval.start.year + 1`. Este módulo
   filtra y se queda solo con los periodos cuyo año calculado coincide con el año
   pedido, descartando la propina (el año adyacente se recogerá en su propia
   llamada del bucle).
2. Un rango de más de ~2 años en una sola petición devuelve 400 Bad Request
   (verificado: 1 año OK, 3 años falla) — a diferencia de `entsoe_installed_capacity`
   no hay beneficio en pedir varios años de golpe de todas formas, dado el punto 1.
3. `nominalIP_PowerSystemResources.nominalP` (capacidad de LA CENTRAL) es el
   campo relevante — el XML también incluye sub-elementos `<PowerSystemResources>`
   por cada turbina/grupo individual dentro de la central (ej. "BESOS 5" tiene
   turbinas de gas TG1/TG2 individuales), con más detalle del que hace falta para
   el merit order (que solo necesita la capacidad de despacho de la central
   completa) — no se cargan, solo el total de la central.
4. Solo centrales SIGNIFICATIVAS (mismo umbral que las demás fuentes ENTSO-E de
   este proyecto, ≥100MW aprox.) — España tiene ~350 unidades en total en 2023,
   de las cuales 51 son de gas (`psrType=B04`, 400-860 MW cada una) y 4 de carbón
   (`psrType=B05`).
"""

from __future__ import annotations

from datetime import date, datetime
import xml.etree.ElementTree as ET

import duckdb
import pandas as pd

from etl.common import catalog
from etl.common.catalog import ColumnDoc
from etl.common.entsoe_xml import strip_ns
from etl.sources.entsoe_load_generation import PSR_TYPE_NAMES, fetch_raw

AREAS = {
    "spain": {"eic": "10YES-REE------0", "db": "spain"},
    # Alemania añadida 2026-09-03: la pregunta era quién genera y con cuánto peso
    # en ese mercado (findings.md #136). Es el registro real de unidades, mucho
    # más completo que deducirlo de quién publica indisponibilidad — que en
    # Alemania son solo las ~100 unidades significativas.
    "germany": {"eic": "10Y1001A1001A82H", "db": "germany"},
}

TABLE = "entsoe_generation_units"


# ---------------------------------------------------------------------------
# Parse
# ---------------------------------------------------------------------------

def parse_generation_units(xml_text: str, target_year: int) -> list[dict]:
    """Un valor por central para el año pedido — descarta el <Period> "de propina"
    del año adyacente, ver docstring del módulo punto 1."""
    root = strip_ns(ET.fromstring(xml_text))
    if root.tag == "Acknowledgement_MarketDocument":
        reason = root.findtext(".//Reason/text") or "sin detalle"
        raise ValueError(f"ENTSO-E devolvió un Acknowledgement (sin datos): {reason}")

    rows = []
    for ts in root.findall("TimeSeries"):
        unit_eic = ts.findtext("registeredResource.mRID")
        unit_name = ts.findtext("registeredResource.name")
        psr_type = ts.findtext("MktPSRType/psrType")
        capacity_mw = ts.findtext("MktPSRType/nominalIP_PowerSystemResources.nominalP")
        period = ts.find("Period")
        if not (unit_eic and psr_type and capacity_mw and period is not None):
            continue
        period_start = period.findtext("timeInterval.start")
        if not period_start:
            continue
        start_dt = datetime.fromisoformat(period_start.replace("Z", "+00:00"))
        computed_year = start_dt.year + 1
        if computed_year != target_year:
            continue
        rows.append({
            "unit_eic": unit_eic,
            "unit_name": (unit_name or unit_eic).strip(),
            "psr_type": psr_type,
            "capacity_mw": float(capacity_mw),
        })
    return rows


# ---------------------------------------------------------------------------
# Diccionario de datos
# ---------------------------------------------------------------------------

def document(con: duckdb.DuckDBPyConnection) -> None:
    cols = {
        "capacity_year": ColumnDoc(
            description="Año natural al que se refiere la capacidad instalada (dato de 'year ahead').",
            source="Calculado como timeInterval.start.year + 1 del <Period> del XML (documentType A71) — ver docstring del módulo.",
            dtype="SMALLINT", kind="categorical",
        ),
        "unit_eic": ColumnDoc(
            description="Código EIC de la central (registeredResource.mRID) — identificador estable entre años.",
            source="Elemento <registeredResource.mRID> de cada <TimeSeries> del XML de ENTSO-E.",
            dtype="VARCHAR", kind="identifier",
        ),
        "unit_name": ColumnDoc(
            description="Nombre comercial de la central tal como lo publica ENTSO-E (p.ej. 'BESOS 5', 'CASTELLO 4').",
            source="Elemento <registeredResource.name> del XML.",
            dtype="VARCHAR", kind="categorical",
        ),
        "psr_type": ColumnDoc(
            description="Código de tecnología ENTSO-E (PsrType) — mismo catálogo que entsoe_installed_capacity/entsoe_generation_by_type.",
            source="Elemento <psrType> de cada <TimeSeries> del XML.",
            dtype="VARCHAR", kind="categorical",
        ),
        "psr_type_name": ColumnDoc(
            description="Nombre legible del psr_type (misma tabla PSR_TYPE_NAMES que entsoe_load_generation.py).",
            source="Constante de configuración, no viene en el XML.",
            dtype="VARCHAR", kind="categorical",
        ),
        "capacity_mw": ColumnDoc(
            description="Capacidad nominal de LA CENTRAL completa (no por turbina individual) en MW.",
            source="Elemento <nominalIP_PowerSystemResources.nominalP> del XML.",
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
            unit_eic        VARCHAR NOT NULL,
            unit_name       VARCHAR NOT NULL,
            psr_type        VARCHAR NOT NULL,
            psr_type_name   VARCHAR NOT NULL,
            capacity_mw     DOUBLE NOT NULL,
            area_code       VARCHAR NOT NULL,
            ingested_at     TIMESTAMP NOT NULL,
            PRIMARY KEY (capacity_year, unit_eic)
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
    source = f"entsoe_generation_units:{area_key}:{year}"

    try:
        raw = fetch_raw({
            "documentType": "A71", "processType": "A33", "in_Domain": eic,
            "periodStart": f"{year}01010000", "periodEnd": f"{year + 1}01010000",
        })
        rows = parse_generation_units(raw, year)
    except Exception as exc:
        _log_run(con, source, 0, "ERROR", str(exc), started_at)
        logger.error("%s: %s", source, exc)
        return {"status": "ERROR", "message": str(exc)}

    now = datetime.utcnow()
    status = "OK"
    message = f"sin incidencias, {len(rows)} centrales"
    rows_loaded = 0
    if not rows:
        status = "WARN"
        message = "sin filas"
    else:
        df = pd.DataFrame(
            [(year, r["unit_eic"], r["unit_name"], r["psr_type"],
              PSR_TYPE_NAMES.get(r["psr_type"], r["psr_type"]), r["capacity_mw"], eic, now) for r in rows],
            columns=["capacity_year", "unit_eic", "unit_name", "psr_type", "psr_type_name",
                     "capacity_mw", "area_code", "ingested_at"],
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

    area = sys.argv[1] if len(sys.argv) > 1 else "spain"
    log = get_logger("entsoe_generation_units.manual", AREAS[area]["db"])
    conn = connect(AREAS[area]["db"])
    print(run_for_year(conn, area, 2023, log))
    document(conn)
    document_ingestion_log(conn)
    conn.close()
