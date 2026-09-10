"""
Ingesta de precios day-ahead de Alemania/Luxemburgo (SMARD, Bundesnetzagentur).

Fuente: https://www.smard.de/app/chart_data/4169/DE-LU/... (API interna del visor
público de SMARD, sin token, verificada en vivo el 2026-08-26). Filtro 4169 =
"Großhandelspreise" (día-adelantado), región DE-LU. Los datos se sirven en bloques
fijos de 7 días (672 puntos de 15 min), en timestamps epoch-ms **ya en UTC puro**
(a diferencia de OMIE/AEMO no hace falta convertir zona horaria: SMARD entrega un
reloj de 15 min continuo en UTC que atraviesa los cambios de hora sin huecos ni
duplicados).

Hecho verificado en vivo, crítico para la integridad de la serie:
Antes del 2025-10-01 00:00 hora local alemana (reforma europea de armonización del
Market Time Unit a 15 min, la misma que afecta a OMIE), el feed de 15 min de SMARD
repite el mismo precio en los 4 cuartos de cada hora (la subasta real era horaria).
Este pipeline NO almacena esa duplicación: agrupa por hora UTC y, si los 4 valores
son idénticos, guarda UNA fila con resolution_minutes=60; si difieren, guarda las 4
filas de 15 min. La resolución real se detecta empíricamente a partir del propio
dato (no de una fecha fija), como comprobación cruzada frente al calendario.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo

import duckdb
import pandas as pd
import requests

from etl.common import catalog
from etl.common.catalog import ColumnDoc

INDEX_URL = "https://www.smard.de/app/chart_data/4169/DE-LU/index_quarterhour.json"
CHUNK_URL = "https://www.smard.de/app/chart_data/4169/DE-LU/4169_DE-LU_quarterhour_{ts}.json"
TZ_BERLIN = ZoneInfo("Europe/Berlin")
CHUNK_SPAN_MS = 7 * 24 * 3600 * 1000

# Mismo límite armonizado del acoplamiento único europeo (SDAC/Euphemia) que usa
# OMIE — Alemania participa del mismo mecanismo de casación que España.
PRICE_SANITY_MIN, PRICE_SANITY_MAX = -500.0, 4000.0

_session = requests.Session()
_session.headers.update(
    {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json",
    }
)


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------

def fetch_index(timeout: int = 30) -> list[int]:
    resp = _session.get(INDEX_URL, timeout=timeout)
    resp.raise_for_status()
    return resp.json()["timestamps"]


def fetch_chunk(chunk_ts: int, timeout: int = 30) -> list[list]:
    resp = _session.get(CHUNK_URL.format(ts=chunk_ts), timeout=timeout)
    if resp.status_code == 404:
        return []
    resp.raise_for_status()
    return resp.json()["series"]


# ---------------------------------------------------------------------------
# Parse + agregación horaria empírica
# ---------------------------------------------------------------------------

def parse(series: list[list]) -> list[dict]:
    """Agrupa los puntos de 15 min por hora UTC. Si los 4 valores no nulos de una
    hora son idénticos, colapsa a una fila horaria (resolution_minutes=60); si
    difieren, conserva las filas de 15 min individuales (resolution_minutes=15).
    Los puntos con precio None (horizonte aún no publicado) se descartan."""
    by_hour: dict[int, list[tuple[int, float]]] = {}
    for ts_ms, price in series:
        if price is None:
            continue
        hour_bucket = (ts_ms // 3_600_000) * 3_600_000
        by_hour.setdefault(hour_bucket, []).append((ts_ms, price))

    rows: list[dict] = []
    for hour_bucket, points in sorted(by_hour.items()):
        prices = {p for _, p in points}
        if len(points) == 4 and len(prices) == 1:
            rows.append({"interval_start_ms": hour_bucket, "resolution_minutes": 60, "price_eur_mwh": points[0][1]})
        else:
            for ts_ms, price in points:
                rows.append({"interval_start_ms": ts_ms, "resolution_minutes": 15, "price_eur_mwh": price})
    return rows


# ---------------------------------------------------------------------------
# Validate
# ---------------------------------------------------------------------------

def validate(rows: list[dict], chunk_ts: int) -> dict:
    messages: list[str] = []
    status = "OK"

    if not rows:
        return {"status": "WARN", "messages": ["sin datos publicados en este bloque"], "actual_rows": 0}

    n_out_of_band = sum(1 for r in rows if not (PRICE_SANITY_MIN <= r["price_eur_mwh"] <= PRICE_SANITY_MAX))
    if n_out_of_band:
        status = "WARN"
        messages.append(f"{n_out_of_band} intervalos fuera de banda plausible [{PRICE_SANITY_MIN},{PRICE_SANITY_MAX}]")

    n_negative = sum(1 for r in rows if r["price_eur_mwh"] < 0)
    if n_negative:
        messages.append(f"{n_negative} intervalos con precio negativo (informativo, no es anomalía)")

    n_hourly = sum(1 for r in rows if r["resolution_minutes"] == 60)
    n_quarter = sum(1 for r in rows if r["resolution_minutes"] == 15)
    if n_hourly and n_quarter:
        messages.append(f"bloque mixto: {n_hourly} horas a 60min + {n_quarter} intervalos a 15min (día de transición 15-min, informativo)")

    return {"status": status, "messages": messages, "actual_rows": len(rows)}


# ---------------------------------------------------------------------------
# Diccionario de datos
# ---------------------------------------------------------------------------

SMARD_COLUMNS = {
    "interval_start_utc": ColumnDoc(
        description="Instante UTC de inicio del intervalo de precio. SMARD publica ya en UTC puro, sin necesidad de resolución de zona horaria.",
        source="Campo epoch-ms de cada punto de series[] en el JSON de SMARD (filtro 4169, región DE-LU).",
        dtype="TIMESTAMP", kind="identifier",
    ),
    "interval_start_local": ColumnDoc(
        description="Mismo instante expresado en hora local de Alemania (Europe/Berlin, con horario de verano), para cruces con calendario alemán.",
        source="Calculado por el pipeline a partir de interval_start_utc vía zoneinfo Europe/Berlin.",
        dtype="TIMESTAMP", kind="timestamp",
    ),
    "resolution_minutes": ColumnDoc(
        description=(
            "Duración real del intervalo de mercado: 60 hasta el 2025-09-30, 15 desde el 2025-10-01. "
            "SMARD siempre sirve el feed en grano de 15 min, pero antes de la reforma repite el mismo "
            "precio en los 4 cuartos de cada hora; el pipeline detecta esto empíricamente y colapsa esas "
            "filas a una sola de 60 min en vez de almacenar precisión falsa."
        ),
        source="Calculado por parse(): agrupación horaria + comprobación de valores idénticos.",
        dtype="SMALLINT", kind="categorical",
    ),
    "price_eur_mwh": ColumnDoc(
        description="Precio day-ahead armonizado del acoplamiento único europeo (SDAC) para la zona DE-LU, en EUR/MWh.",
        source="Campo 'series' del JSON de SMARD, filtro 4169.",
        dtype="DOUBLE", kind="continuous",
    ),
    "source_chunk_ts": ColumnDoc(
        description="Timestamp epoch-ms del bloque semanal de origen descargado de SMARD, para trazabilidad completa del dato.",
        source="Parámetro chunk_ts de la petición a la API de SMARD.",
        dtype="BIGINT", kind="identifier",
    ),
    "ingested_at": ColumnDoc(
        description="Marca temporal UTC del momento en que esta fila se cargó (o recargó) por última vez en la base de datos.",
        source="datetime.utcnow() en el momento de la inserción, dentro de la función load().",
        dtype="TIMESTAMP", kind="timestamp",
    ),
}


def document(con: duckdb.DuckDBPyConnection) -> None:
    catalog.refresh(con, "smard_day_ahead_prices", SMARD_COLUMNS)


# ---------------------------------------------------------------------------
# Schema + load
# ---------------------------------------------------------------------------

def ensure_schema(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS smard_day_ahead_prices (
            interval_start_utc    TIMESTAMP NOT NULL,
            interval_start_local  TIMESTAMP NOT NULL,
            resolution_minutes    SMALLINT NOT NULL,
            price_eur_mwh         DOUBLE NOT NULL,
            source_chunk_ts       BIGINT NOT NULL,
            ingested_at           TIMESTAMP NOT NULL,
            PRIMARY KEY (interval_start_utc)
        )
        """
    )


def load(con: duckdb.DuckDBPyConnection, chunk_ts: int, rows: list[dict]) -> int:
    ensure_schema(con)
    now = datetime.utcnow()
    if not rows:
        return 0

    records = []
    for r in rows:
        start_utc = datetime.fromtimestamp(r["interval_start_ms"] / 1000, tz=timezone.utc)
        start_local = start_utc.astimezone(TZ_BERLIN).replace(tzinfo=None)
        records.append(
            (start_utc.replace(tzinfo=None), start_local, r["resolution_minutes"], r["price_eur_mwh"], chunk_ts, now)
        )
    df = pd.DataFrame(
        records,
        columns=["interval_start_utc", "interval_start_local", "resolution_minutes", "price_eur_mwh", "source_chunk_ts", "ingested_at"],
    )
    min_ts, max_ts = df["interval_start_utc"].min(), df["interval_start_utc"].max()

    con.execute("BEGIN TRANSACTION")
    try:
        con.execute(
            "DELETE FROM smard_day_ahead_prices WHERE interval_start_utc >= ? AND interval_start_utc <= ?",
            [min_ts, max_ts],
        )
        con.append("smard_day_ahead_prices", df)
        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK")
        raise
    return len(df)


# ---------------------------------------------------------------------------
# Orquestación de un bloque semanal + registro de auditoría
# ---------------------------------------------------------------------------

def _log_run(con, chunk_ts, rows_loaded, status, message, started_at) -> None:
    chunk_date = date.fromtimestamp(chunk_ts / 1000)
    con.execute(
        """
        INSERT INTO ingestion_log
            (source, target_table, delivery_date, rows_expected, rows_loaded, status, message, started_at, finished_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        ["smard_day_ahead", "smard_day_ahead_prices", chunk_date, None, rows_loaded, status, message,
         started_at, datetime.utcnow()],
    )


def run_for_chunk(con: duckdb.DuckDBPyConnection, chunk_ts: int, logger) -> dict:
    started_at = datetime.utcnow()
    ensure_schema(con)

    try:
        series = fetch_chunk(chunk_ts)
    except Exception as exc:
        _log_run(con, chunk_ts, None, "ERROR", f"fallo de red: {exc}", started_at)
        logger.error("SMARD chunk %s: fallo de red: %s", chunk_ts, exc)
        return {"status": "ERROR", "chunk_ts": chunk_ts, "message": str(exc)}

    rows = parse(series)
    validation = validate(rows, chunk_ts)
    rows_loaded = load(con, chunk_ts, rows)

    status = validation["status"]
    message = "; ".join(validation["messages"]) if validation["messages"] else "sin incidencias"
    _log_run(con, chunk_ts, rows_loaded, status, message, started_at)

    log_fn = logger.info if status == "OK" else logger.warning
    log_fn("SMARD chunk %s: %s filas cargadas — %s", date.fromtimestamp(chunk_ts / 1000), rows_loaded, message)
    return {"status": status, "chunk_ts": chunk_ts, "rows_loaded": rows_loaded, "message": message}


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(__file__).rsplit("etl", 1)[0])
    from etl.common.db import connect, document_ingestion_log
    from etl.common.logging_config import get_logger

    log = get_logger("smard_day_ahead.manual", "germany")
    conn = connect("germany")
    idx = fetch_index()
    target_ts = idx[int(sys.argv[1])] if len(sys.argv) > 1 else idx[-1]
    result = run_for_chunk(conn, target_ts, log)
    print(result)
    document(conn)
    document_ingestion_log(conn)
    conn.close()
