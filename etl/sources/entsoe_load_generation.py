"""
Ingesta de demanda real (documentType A65, "Actual Total Load") y generación real por
tecnología (documentType A75, "Actual Generation per Type") vía ENTSO-E Transparency
Platform, para Alemania, Francia y Países Bajos (mismas áreas EIC que
etl/sources/entsoe_day_ahead.py).

Hechos verificados en vivo el 2026-08-26:
1. Igual que el precio day-ahead, tanto la carga como la generación pasan de
   resolución horaria a cuartohoraria — verificado para Francia: PT60M en 2023-01,
   PT15M en 2026-08. Se asume el mismo corte 2025-10-01 salvo que la validación
   detecte lo contrario (se registra la resolución real de cada fila, no se asume).
2. La generación por tecnología (A75) da timeout en el servidor de ENTSO-E al pedir
   un año completo (~90s sin respuesta) aunque el precio day-ahead (A44) sí lo permite.
   Se trocea por MES para generación; la demanda (A65), más ligera, se trocea por AÑO
   como los precios.
3. Reutiliza el parseo común de etl/common/entsoe_xml.py (forward-fill de puntos
   omitidos + deduplicación de TimeSeries repetidos), agrupando por <psrType> en el
   caso de generación.
"""

from __future__ import annotations

import time
from datetime import date, datetime, timezone

import duckdb
import pandas as pd
import requests

from etl.common import catalog, entsoe_xml
from etl.common.catalog import ColumnDoc
from etl.common.config import require_env
from etl.sources.entsoe_day_ahead import AREAS as PRICE_AREAS

API_URL = "https://web-api.tp.entsoe.eu/api"

# Demanda y generación por tecnología están disponibles en ENTSO-E para Alemania
# también, aunque el precio day-ahead alemán lo cubrimos vía SMARD (ver
# etl/sources/smard_day_ahead.py) — por eso este AREAS es un superset del de
# entsoe_day_ahead.py, no el mismo diccionario: añadirla ahí habría hecho que
# daily_update_entsoe.py empezara a duplicar el precio alemán innecesariamente.
from zoneinfo import ZoneInfo

AREAS = {
    "germany": {"eic": "10Y1001A1001A82H", "tz": ZoneInfo("Europe/Berlin"), "db": "germany"},
    **PRICE_AREAS,
}

PSR_TYPE_NAMES = {
    "B01": "Biomasa", "B02": "Lignito", "B03": "Gas derivado de carbón", "B04": "Gas natural",
    "B05": "Hulla", "B06": "Fuel-oil", "B07": "Pizarra bituminosa", "B08": "Turba",
    "B09": "Geotérmica", "B10": "Hidráulica de bombeo", "B11": "Hidráulica fluyente",
    "B12": "Hidráulica de embalse", "B13": "Marina", "B14": "Nuclear", "B15": "Otras renovables",
    "B16": "Solar", "B17": "Residuos", "B18": "Eólica marina", "B19": "Eólica terrestre",
    "B20": "Otras", "B21": "Enlace AC", "B22": "Enlace DC", "B23": "Subestación", "B24": "Transformador",
    "B25": "Almacenamiento de energía (código introducido tras el mapeo original de ENTSO-E; nombre provisional)",
}


def _generation_series_key(ts) -> str:
    """Algunos psrType (sobre todo B10, hidráulica de bombeo) tienen DOS <TimeSeries>
    para el mismo día: una de generación (inBiddingZone_Domain) y otra de consumo/
    bombeo (outBiddingZone_Domain). Agrupar solo por psrType las fusiona por error
    (verificado en vivo: miles de "revisiones" falsas por mes en Francia). La clave
    incluye la dirección del flujo para mantenerlas separadas."""
    psr = ts.findtext("MktPSRType/psrType") or ts.findtext("psrType")
    if psr is None:
        return None
    direction = "generation" if ts.find("inBiddingZone_Domain.mRID") is not None else "consumption"
    return f"{psr}|{direction}"

_session = requests.Session()


def fetch_raw(params: dict, timeout: int = 90, retries: int = 3) -> str:
    full_params = {"securityToken": require_env("ENTSOE_API_TOKEN"), **params}
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            resp = _session.get(API_URL, params=full_params, timeout=timeout)
            resp.raise_for_status()
            # ENTSO-E devuelve XML en UTF-8 pero NO lo declara en la cabecera
            # Content-Type, y en ese caso `requests` cae por defecto a ISO-8859-1
            # (lo manda el RFC 2616 para text/*). El resultado es mojibake en
            # cualquier campo con acentos: "CASTAÑO" llegaba como "CASTAÃO" y
            # "Niederaußem" como "NiederauÃem" — verificado el 2026-09-03 en las
            # tablas ya cargadas de España y Alemania. Los números no se ven
            # afectados, solo el texto. findings.md #136.
            resp.encoding = "utf-8"
            return resp.text
        except requests.RequestException as exc:
            last_exc = exc
            time.sleep(2 * attempt)
    raise last_exc  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Diccionario de datos
# ---------------------------------------------------------------------------

def _base_columns(area_key: str, value_name: str, value_desc: str, extra: dict | None = None) -> dict:
    tz_name = AREAS[area_key]["tz"].key
    cols = {
        "interval_start_utc": ColumnDoc(
            description="Instante UTC de inicio del intervalo.",
            source="timeInterval/start de cada <Period> + offset de (position-1) × resolución, del XML de ENTSO-E.",
            dtype="TIMESTAMP", kind="identifier",
        ),
        "interval_start_local": ColumnDoc(
            description=f"Mismo instante en hora local del área ({tz_name}, con horario de verano).",
            source="Calculado por el pipeline a partir de interval_start_utc.",
            dtype="TIMESTAMP", kind="timestamp",
        ),
        "resolution_minutes": ColumnDoc(
            description="Duración real del intervalo: 60 hasta el 2025-09-30, 15 desde el 2025-10-01 (misma reforma paneuropea, verificada también para esta serie).",
            source="Elemento <resolution> del <Period> correspondiente del XML de ENTSO-E.",
            dtype="SMALLINT", kind="categorical",
        ),
        value_name: ColumnDoc(description=value_desc, source="Elemento <quantity> de cada <Point>, con forward-fill de posiciones omitidas.", dtype="DOUBLE", kind="continuous"),
        "area_code": ColumnDoc(
            description="Código EIC del área ENTSO-E.",
            source="Constante de configuración (AREAS en entsoe_day_ahead.py).",
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


def document_load(con: duckdb.DuckDBPyConnection, area_key: str) -> None:
    cols = _base_columns(area_key, "load_mw", "Demanda real (carga) del sistema en el área, en MW.")
    catalog.refresh(con, "entsoe_load", cols)


def document_generation(con: duckdb.DuckDBPyConnection, area_key: str) -> None:
    cols = _base_columns(
        area_key, "generation_mw", "Generación real de esa tecnología en el área, en MW.",
        extra={
            "psr_type": ColumnDoc(
                description="Código de tecnología ENTSO-E (PsrType): B01=Biomasa, B04=Gas, B05=Hulla, B10/B11/B12=Hidráulica (bombeo/fluyente/embalse), B14=Nuclear, B16=Solar, B19=Eólica terrestre, etc.",
                source="Elemento <psrType> de cada <TimeSeries> del XML de ENTSO-E.",
                dtype="VARCHAR", kind="categorical",
            ),
            "flow_direction": ColumnDoc(
                description=(
                    "'generation' o 'consumption'. Algunas tecnologías (sobre todo B10, hidráulica de bombeo) "
                    "publican dos series para el mismo psr_type: la energía generada y la consumida para bombear. "
                    "Tratarlas como una sola serie fusiona por error ambos flujos (ver docs/findings.md)."
                ),
                source="Derivado de si el <TimeSeries> trae inBiddingZone_Domain.mRID (generation) o outBiddingZone_Domain.mRID (consumption).",
                dtype="VARCHAR", kind="categorical",
            ),
            "psr_type_name": ColumnDoc(
                description="Nombre legible del psr_type (tabla de traducción fija PSR_TYPE_NAMES del pipeline).",
                source="Constante de configuración, no viene en el XML.",
                dtype="VARCHAR", kind="categorical",
            ),
        },
    )
    catalog.refresh(con, "entsoe_generation_by_type", cols)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

def ensure_schema(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS entsoe_load (
            interval_start_utc    TIMESTAMP NOT NULL,
            interval_start_local  TIMESTAMP NOT NULL,
            resolution_minutes    SMALLINT NOT NULL,
            load_mw               DOUBLE NOT NULL,
            area_code             VARCHAR NOT NULL,
            source_chunk          VARCHAR NOT NULL,
            ingested_at           TIMESTAMP NOT NULL,
            PRIMARY KEY (interval_start_utc)
        )
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS entsoe_generation_by_type (
            interval_start_utc    TIMESTAMP NOT NULL,
            psr_type              VARCHAR NOT NULL,
            flow_direction        VARCHAR NOT NULL,
            interval_start_local  TIMESTAMP NOT NULL,
            resolution_minutes    SMALLINT NOT NULL,
            generation_mw         DOUBLE NOT NULL,
            psr_type_name         VARCHAR NOT NULL,
            area_code             VARCHAR NOT NULL,
            source_chunk          VARCHAR NOT NULL,
            ingested_at           TIMESTAMP NOT NULL,
            PRIMARY KEY (interval_start_utc, psr_type, flow_direction)
        )
        """
    )


# ---------------------------------------------------------------------------
# Load (demanda)
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


def run_load_for_range(con: duckdb.DuckDBPyConnection, area_key: str, period_start: str, period_end: str, logger) -> dict:
    started_at = datetime.utcnow()
    ensure_schema(con)
    eic = AREAS[area_key]["eic"]
    tz = AREAS[area_key]["tz"]
    source_chunk = f"{period_start}-{period_end}"
    source = f"entsoe_load:{area_key}"

    try:
        raw = fetch_raw({"documentType": "A65", "processType": "A16", "outBiddingZone_Domain": eic, "periodStart": period_start, "periodEnd": period_end})
        grouped = entsoe_xml.parse_periods(raw)
        rows, parse_stats = grouped.get("default", ([], {"n_exact_duplicates": 0, "n_revised": 0}))
    except Exception as exc:
        _log_run(con, source, "entsoe_load", 0, "ERROR", str(exc), started_at)
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
    if rows:
        df = pd.DataFrame(
            [
                (r["interval_start_utc"], r["interval_start_utc"].replace(tzinfo=timezone.utc).astimezone(tz).replace(tzinfo=None),
                 r["resolution_minutes"], r["value"], eic, source_chunk, now)
                for r in rows
            ],
            columns=["interval_start_utc", "interval_start_local", "resolution_minutes", "load_mw", "area_code", "source_chunk", "ingested_at"],
        )
        min_ts, max_ts = df["interval_start_utc"].min(), df["interval_start_utc"].max()
        con.execute("BEGIN TRANSACTION")
        try:
            con.execute("DELETE FROM entsoe_load WHERE interval_start_utc >= ? AND interval_start_utc <= ?", [min_ts, max_ts])
            con.append("entsoe_load", df)
            con.execute("COMMIT")
        except Exception:
            con.execute("ROLLBACK")
            raise
        rows_loaded = len(df)
    else:
        rows_loaded = 0

    message = "; ".join(messages) if messages else "sin incidencias"
    _log_run(con, source, "entsoe_load", rows_loaded, status, message, started_at)
    log_fn = logger.info if status == "OK" else logger.warning
    log_fn("%s %s: %s filas cargadas — %s", source, source_chunk, rows_loaded, message)
    return {"status": status, "rows_loaded": rows_loaded, "message": message}


# ---------------------------------------------------------------------------
# Generación por tecnología
# ---------------------------------------------------------------------------

def run_generation_for_range(con: duckdb.DuckDBPyConnection, area_key: str, period_start: str, period_end: str, logger) -> dict:
    started_at = datetime.utcnow()
    ensure_schema(con)
    eic = AREAS[area_key]["eic"]
    tz = AREAS[area_key]["tz"]
    source_chunk = f"{period_start}-{period_end}"
    source = f"entsoe_generation:{area_key}"

    try:
        raw = fetch_raw({"documentType": "A75", "processType": "A16", "in_Domain": eic, "periodStart": period_start, "periodEnd": period_end})
        grouped = entsoe_xml.parse_periods(raw, series_key_fn=_generation_series_key)
    except Exception as exc:
        _log_run(con, source, "entsoe_generation_by_type", 0, "ERROR", str(exc), started_at)
        logger.error("%s %s: %s", source, source_chunk, exc)
        return {"status": "ERROR", "message": str(exc)}

    now = datetime.utcnow()
    all_records = []
    n_revised_total = 0
    for composite_key, (rows, parse_stats) in grouped.items():
        if composite_key is None or not rows:
            continue
        psr_type, _, flow_direction = composite_key.partition("|")
        n_revised_total += parse_stats["n_revised"]
        for r in rows:
            all_records.append(
                (
                    r["interval_start_utc"], psr_type, flow_direction,
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
            columns=["interval_start_utc", "psr_type", "flow_direction", "interval_start_local", "resolution_minutes", "generation_mw", "psr_type_name", "area_code", "source_chunk", "ingested_at"],
        )
        min_ts, max_ts = df["interval_start_utc"].min(), df["interval_start_utc"].max()
        con.execute("BEGIN TRANSACTION")
        try:
            con.execute(
                "DELETE FROM entsoe_generation_by_type WHERE interval_start_utc >= ? AND interval_start_utc <= ?",
                [min_ts, max_ts],
            )
            con.append("entsoe_generation_by_type", df)
            con.execute("COMMIT")
        except Exception:
            con.execute("ROLLBACK")
            raise
        rows_loaded = len(df)

    message = "; ".join(messages) if messages else f"sin incidencias, {len(grouped)} tecnologías"
    _log_run(con, source, "entsoe_generation_by_type", rows_loaded, status, message, started_at)
    log_fn = logger.info if status == "OK" else logger.warning
    log_fn("%s %s: %s filas cargadas (%d tecnologías) — %s", source, source_chunk, rows_loaded, len(grouped), message)
    return {"status": status, "rows_loaded": rows_loaded, "message": message}


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(__file__).rsplit("etl", 1)[0])
    from etl.common.db import connect, document_ingestion_log
    from etl.common.logging_config import get_logger

    area = sys.argv[1] if len(sys.argv) > 1 else "france"
    log = get_logger("entsoe_load_generation.manual", AREAS[area]["db"])
    conn = connect(AREAS[area]["db"])
    print(run_load_for_range(conn, area, "202301010000", "202301080000", log))
    print(run_generation_for_range(conn, area, "202301010000", "202301080000", log))
    document_load(conn, area)
    document_generation(conn, area)
    document_ingestion_log(conn)
    conn.close()
