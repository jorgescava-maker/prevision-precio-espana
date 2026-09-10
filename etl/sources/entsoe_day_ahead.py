"""
Ingesta de precios day-ahead vía ENTSO-E Transparency Platform (documentType A44),
para cualquier área de precio del acoplamiento único europeo identificada por su
código EIC. Usado aquí para Francia y Países Bajos (Alemania ya se cubre sin token
vía SMARD, ver etl/sources/smard_day_ahead.py).

Requiere ENTSOE_API_TOKEN en .env (token gratuito, registro en transparency.entsoe.eu).

Hechos verificados en vivo el 2026-08-26, críticos para un parseo correcto:

1. Cuando el precio no cambia entre posiciones consecutivas dentro de un <Period>,
   el <Point> correspondiente se OMITE del XML (no se repite el valor). Hay que
   rellenar hacia adelante (forward-fill) con el último precio visto — un parser que
   solo lea los <Point> presentes pierde silenciosamente esas horas/cuartos.
2. Cada <Period> corresponde a UN DÍA de mercado local del área (no al rango exacto
   solicitado): si la ventana pedida solo toca parcialmente ese día en UTC, ENTSO-E
   devuelve igualmente el día COMPLETO. Por eso una petición de un año entero es
   suficiente y eficiente (~365-366 <Period>, uno por día), y por eso los chunks por
   año pueden solaparse un día en el borde — el load() es idempotente por rango real
   de timestamps devueltos, así que un solape no causa duplicados ni errores.
3. La resolución (PT60M / PT15M) viene declarada explícitamente en cada <Period>. El
   corte a cuarto de hora ocurre el 2025-10-01 tanto en Francia como en Países Bajos
   (misma reforma paneuropea ya verificada en OMIE y SMARD/Alemania).
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

# Un área = una base de datos de país (mismo patrón que OMIE/AEMO/SMARD).
AREAS = {
    "france": {"eic": "10YFR-RTE------C", "tz": ZoneInfo("Europe/Paris"), "db": "france"},
    "netherlands": {"eic": "10YNL----------L", "tz": ZoneInfo("Europe/Amsterdam"), "db": "netherlands"},
}

PRICE_SANITY_MIN, PRICE_SANITY_MAX = -500.0, 4000.0  # mismo límite armonizado SDAC que OMIE/SMARD

_session = requests.Session()


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------

def fetch_raw(eic: str, period_start: str, period_end: str, timeout: int = 60, retries: int = 3) -> str:
    """period_start/period_end en formato ENTSO-E: 'AAAAMMDDHHmm' (UTC)."""
    params = {
        "securityToken": require_env("ENTSOE_API_TOKEN"),
        "documentType": "A44",
        "in_Domain": eic,
        "out_Domain": eic,
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
# Parse (con forward-fill de posiciones omitidas)
# ---------------------------------------------------------------------------

def _strip_ns(elem: ET.Element) -> ET.Element:
    for el in elem.iter():
        if "}" in el.tag:
            el.tag = el.tag.split("}", 1)[1]
    return elem


def parse(xml_text: str) -> tuple[list[dict], dict]:
    """Devuelve (filas, stats). ENTSO-E puede incluir más de un <TimeSeries> cubriendo
    el mismo día (verificado en vivo: 2026-08-07 y 2026-08-08 en Francia llegaron
    duplicados, con contenido byte a byte idéntico en ambos casos). Se deduplica por
    interval_start_utc quedándose con la ÚLTIMA ocurrencia en orden del documento; si
    dos ocurrencias del mismo instante tuvieran precios DISTINTOS (una revisión real,
    no un duplicado), se cuenta aparte para que quede reflejado en la validación."""
    root = _strip_ns(ET.fromstring(xml_text))

    if root.tag == "Acknowledgement_MarketDocument":
        reason = root.findtext(".//Reason/text") or "sin detalle"
        raise ValueError(f"ENTSO-E devolvió un Acknowledgement (sin datos): {reason}")

    by_ts: dict[datetime, dict] = {}
    n_exact_duplicates = 0
    n_revised = 0

    for ts in root.findall("TimeSeries"):
        for period in ts.findall("Period"):
            start_str = period.findtext("timeInterval/start")
            end_str = period.findtext("timeInterval/end")
            resolution_str = period.findtext("resolution")
            resolution_minutes = {"PT60M": 60, "PT15M": 15, "PT30M": 30}.get(resolution_str)
            if resolution_minutes is None:
                raise ValueError(f"resolución no soportada: {resolution_str}")

            start_dt = datetime.strptime(start_str, "%Y-%m-%dT%H:%MZ").replace(tzinfo=timezone.utc)
            end_dt = datetime.strptime(end_str, "%Y-%m-%dT%H:%MZ").replace(tzinfo=timezone.utc)
            n_positions = round((end_dt - start_dt).total_seconds() / 60 / resolution_minutes)

            points: dict[int, float] = {}
            for pt in period.findall("Point"):
                pos = int(pt.findtext("position"))
                points[pos] = float(pt.findtext("price.amount"))

            last_price = None
            for pos in range(1, n_positions + 1):
                if pos in points:
                    last_price = points[pos]
                if last_price is None:
                    continue  # no debería ocurrir: position=1 siempre viene explícita
                interval_start = (start_dt + timedelta(minutes=(pos - 1) * resolution_minutes)).replace(tzinfo=None)
                row = {"interval_start_utc": interval_start, "resolution_minutes": resolution_minutes, "price_eur_mwh": last_price}

                existing = by_ts.get(interval_start)
                if existing is not None:
                    if existing["price_eur_mwh"] == last_price:
                        n_exact_duplicates += 1
                    else:
                        n_revised += 1
                by_ts[interval_start] = row  # última ocurrencia gana

    rows = [by_ts[k] for k in sorted(by_ts)]
    stats = {"n_exact_duplicates": n_exact_duplicates, "n_revised": n_revised}
    return rows, stats


# ---------------------------------------------------------------------------
# Validate
# ---------------------------------------------------------------------------

def validate(rows: list[dict], parse_stats: dict) -> dict:
    messages: list[str] = []
    status = "OK"

    if not rows:
        return {"status": "WARN", "messages": ["sin filas en este bloque"], "actual_rows": 0}

    if parse_stats["n_exact_duplicates"]:
        messages.append(f"{parse_stats['n_exact_duplicates']} intervalos venían duplicados en el XML (contenido idéntico, deduplicados)")
    if parse_stats["n_revised"]:
        status = "WARN"
        messages.append(f"{parse_stats['n_revised']} intervalos tenían DOS valores distintos en el XML (revisión real, se usó el último)")

    n_out_of_band = sum(1 for r in rows if not (PRICE_SANITY_MIN <= r["price_eur_mwh"] <= PRICE_SANITY_MAX))
    if n_out_of_band:
        status = "WARN"
        messages.append(f"{n_out_of_band} intervalos fuera de banda plausible [{PRICE_SANITY_MIN},{PRICE_SANITY_MAX}]")

    n_negative = sum(1 for r in rows if r["price_eur_mwh"] < 0)
    if n_negative:
        messages.append(f"{n_negative} intervalos con precio negativo (informativo, no es anomalía)")

    return {"status": status, "messages": messages, "actual_rows": len(rows)}


# ---------------------------------------------------------------------------
# Diccionario de datos
# ---------------------------------------------------------------------------

def _columns(area_key: str) -> dict:
    tz_name = AREAS[area_key]["tz"].key
    return {
        "interval_start_utc": ColumnDoc(
            description="Instante UTC de inicio del intervalo de precio.",
            source="timeInterval/start de cada <Period> + offset de (position-1) × resolución, del XML de ENTSO-E (documentType A44).",
            dtype="TIMESTAMP", kind="identifier",
        ),
        "interval_start_local": ColumnDoc(
            description=f"Mismo instante en hora local del área ({tz_name}, con horario de verano).",
            source="Calculado por el pipeline a partir de interval_start_utc.",
            dtype="TIMESTAMP", kind="timestamp",
        ),
        "resolution_minutes": ColumnDoc(
            description="Duración real del intervalo de mercado: 60 hasta el 2025-09-30, 15 desde el 2025-10-01 (misma reforma paneuropea que OMIE/SMARD).",
            source="Elemento <resolution> del <Period> correspondiente del XML de ENTSO-E.",
            dtype="SMALLINT", kind="categorical",
        ),
        "price_eur_mwh": ColumnDoc(
            description="Precio day-ahead armonizado del acoplamiento único europeo (SDAC) para esta área, en EUR/MWh.",
            source="Elemento <price.amount> de cada <Point>; para posiciones omitidas por el XML (precio igual al anterior) se arrastra hacia adelante el último valor explícito.",
            dtype="DOUBLE", kind="continuous",
        ),
        "area_code": ColumnDoc(
            description="Código EIC del área de precio ENTSO-E.",
            source="Constante de configuración (AREAS en el pipeline), no viene por fila en el XML.",
            dtype="VARCHAR", kind="categorical",
        ),
        "source_chunk": ColumnDoc(
            description="Identificador del bloque de petición de origen (año o rango solicitado a la API), para trazabilidad.",
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
    catalog.refresh(con, "entsoe_day_ahead_prices", _columns(area_key))


# ---------------------------------------------------------------------------
# Schema + load
# ---------------------------------------------------------------------------

def ensure_schema(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS entsoe_day_ahead_prices (
            interval_start_utc    TIMESTAMP NOT NULL,
            interval_start_local  TIMESTAMP NOT NULL,
            resolution_minutes    SMALLINT NOT NULL,
            price_eur_mwh         DOUBLE NOT NULL,
            area_code             VARCHAR NOT NULL,
            source_chunk          VARCHAR NOT NULL,
            ingested_at           TIMESTAMP NOT NULL,
            PRIMARY KEY (interval_start_utc)
        )
        """
    )


def load(con: duckdb.DuckDBPyConnection, area_key: str, rows: list[dict], source_chunk: str) -> int:
    """Rango de borrado calculado a partir del min/max real de las filas (misma
    lección de etl/sources/aemo_spot.py): los <Period> de años consecutivos pueden
    solaparse un día en el borde, y esto lo hace seguro sin duplicados."""
    ensure_schema(con)
    if not rows:
        return 0
    tz = AREAS[area_key]["tz"]
    eic = AREAS[area_key]["eic"]
    now = datetime.utcnow()

    records = []
    for r in rows:
        start_utc = r["interval_start_utc"].replace(tzinfo=timezone.utc)
        start_local = start_utc.astimezone(tz).replace(tzinfo=None)
        records.append((r["interval_start_utc"], start_local, r["resolution_minutes"], r["price_eur_mwh"], eic, source_chunk, now))

    df = pd.DataFrame(
        records,
        columns=["interval_start_utc", "interval_start_local", "resolution_minutes", "price_eur_mwh", "area_code", "source_chunk", "ingested_at"],
    )
    min_ts, max_ts = df["interval_start_utc"].min(), df["interval_start_utc"].max()

    con.execute("BEGIN TRANSACTION")
    try:
        con.execute(
            "DELETE FROM entsoe_day_ahead_prices WHERE interval_start_utc >= ? AND interval_start_utc <= ?",
            [min_ts, max_ts],
        )
        con.append("entsoe_day_ahead_prices", df)
        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK")
        raise
    return len(df)


# ---------------------------------------------------------------------------
# Orquestación de un rango + registro de auditoría
# ---------------------------------------------------------------------------

def _log_run(con, area_key, source_chunk, rows_loaded, status, message, started_at) -> None:
    con.execute(
        """
        INSERT INTO ingestion_log
            (source, target_table, delivery_date, rows_expected, rows_loaded, status, message, started_at, finished_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [f"entsoe:{area_key}", "entsoe_day_ahead_prices", date.today(), None, rows_loaded, status, message,
         started_at, datetime.utcnow()],
    )


def run_for_range(con: duckdb.DuckDBPyConnection, area_key: str, period_start: str, period_end: str, logger) -> dict:
    started_at = datetime.utcnow()
    ensure_schema(con)
    eic = AREAS[area_key]["eic"]
    source_chunk = f"{period_start}-{period_end}"

    try:
        raw = fetch_raw(eic, period_start, period_end)
        rows, parse_stats = parse(raw)
    except Exception as exc:
        _log_run(con, area_key, source_chunk, 0, "ERROR", str(exc), started_at)
        logger.error("ENTSO-E %s %s: %s", area_key, source_chunk, exc)
        return {"status": "ERROR", "area": area_key, "chunk": source_chunk, "message": str(exc)}

    validation = validate(rows, parse_stats)
    rows_loaded = load(con, area_key, rows, source_chunk)

    status = validation["status"]
    message = "; ".join(validation["messages"]) if validation["messages"] else "sin incidencias"
    _log_run(con, area_key, source_chunk, rows_loaded, status, message, started_at)

    log_fn = logger.info if status == "OK" else logger.warning
    log_fn("ENTSO-E %s %s: %s filas cargadas — %s", area_key, source_chunk, rows_loaded, message)
    return {"status": status, "area": area_key, "chunk": source_chunk, "rows_loaded": rows_loaded, "message": message}


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(__file__).rsplit("etl", 1)[0])
    from etl.common.db import connect, document_ingestion_log
    from etl.common.logging_config import get_logger

    area = sys.argv[1] if len(sys.argv) > 1 else "france"
    log = get_logger("entsoe_day_ahead.manual", AREAS[area]["db"])
    conn = connect(AREAS[area]["db"])
    result = run_for_range(conn, area, "202301010000", "202301080000", log)
    print(result)
    document(conn, area)
    document_ingestion_log(conn)
    conn.close()
