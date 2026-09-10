"""
Ingesta de precios spot del mercado diario español (OMIE, fichero marginalpdbc).

Hecho verificado en vivo el 2026-08-26 y crítico para la integridad de la serie:
OMIE pasó de resolución HORARIA a resolución CUARTOHORARIA (15 min) a partir del
día de entrega 2025-10-01 (armonización europea del Market Time Unit a 15 min).
- Antes de esa fecha: 23/24/25 filas por día según cambio de hora (DST).
- Desde esa fecha:     92/96/100 filas por día (23/24/25 horas x 4).
Este módulo detecta la resolución a partir de la fecha y valida el nº de periodos
esperado en cada caso, en vez de asumir un formato fijo.
"""

from __future__ import annotations

import time
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import duckdb
import pandas as pd
import requests

from etl.common import catalog
from etl.common.catalog import ColumnDoc

BASE_URL = "https://www.omie.es/es/file-download"

# Sesión persistente con cabeceras de navegador — ver la nota equivalente en
# etl/sources/aemo_spot.py sobre mitigación de bots tras muchas peticiones seguidas.
_session = requests.Session()
_session.headers.update(
    {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "text/csv,*/*",
    }
)
TZ_MADRID = ZoneInfo("Europe/Madrid")
RESOLUTION_CHANGE_DATE = date(2025, 10, 1)  # primer día de entrega en cuarto-horario
PRICE_SANITY_MIN, PRICE_SANITY_MAX = -500.0, 4000.0  # banda armonizada UE, con margen


# ---------------------------------------------------------------------------
# Calendario / resolución
# ---------------------------------------------------------------------------

def resolution_minutes_for(delivery_date: date) -> int:
    return 15 if delivery_date >= RESOLUTION_CHANGE_DATE else 60


def hours_in_local_day(delivery_date: date) -> float:
    """Duración real (UTC) del día natural en Europe/Madrid: 23, 24 o 25 horas."""
    next_day = delivery_date + timedelta(days=1)
    start = datetime(delivery_date.year, delivery_date.month, delivery_date.day, 0, 0, tzinfo=TZ_MADRID)
    end = datetime(next_day.year, next_day.month, next_day.day, 0, 0, tzinfo=TZ_MADRID)
    return (end.astimezone(timezone.utc) - start.astimezone(timezone.utc)).total_seconds() / 3600


def expected_period_count(delivery_date: date) -> int:
    resolution = resolution_minutes_for(delivery_date)
    return round(hours_in_local_day(delivery_date) * (60 / resolution))


def period_start_utc(delivery_date: date, period_index: int, resolution_minutes: int) -> datetime:
    local_midnight = datetime(delivery_date.year, delivery_date.month, delivery_date.day, 0, 0, tzinfo=TZ_MADRID)
    start_utc = local_midnight.astimezone(timezone.utc)
    return start_utc + timedelta(minutes=(period_index - 1) * resolution_minutes)


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------

MAX_SUFFIX = 5  # OMIE reemite un fichero corregido incrementando el sufijo (.1 -> .2 -> ...)


def _fetch_suffix(delivery_date: date, suffix: int, timeout: int, retries: int) -> str | None:
    """Descarga un sufijo concreto. Devuelve None si no existe esa versión (404 real, o
    HTTP 200 con página HTML de error — OMIE usa ambas convenciones según el caso)."""
    filename = f"marginalpdbc_{delivery_date.strftime('%Y%m%d')}.{suffix}"
    params = {"parents[0]": "marginalpdbc", "filename": filename}
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            resp = _session.get(BASE_URL, params=params, timeout=timeout)
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            text = resp.text
            return text if text.strip().startswith("MARGINALPDBC") else None
        except requests.RequestException as exc:
            last_exc = exc
            time.sleep(1.5 * attempt)
    raise last_exc  # type: ignore[misc]


def fetch_raw(delivery_date: date, timeout: int = 30, retries: int = 3) -> tuple[str | None, str | None]:
    """Descarga el fichero marginalpdbc de una fecha, con fallback a versiones corregidas.

    OMIE republica un fichero corregido con sufijo incremental (.2, .3...) cuando el
    original tiene un error; verificado en vivo que ESTO OCURRE incluso cuando el .1
    original nunca existió (404), no solo como añadido posterior. Por eso: si .1 no
    está disponible, se escanean los sufijos siguientes hasta MAX_SUFFIX y se toma el
    de mayor número disponible (la corrección más reciente), sin asumir contigüidad.
    Devuelve (texto, nombre_de_fichero_realmente_usado) o (None, None) si no hay nada.
    """
    text = _fetch_suffix(delivery_date, 1, timeout, retries)
    if text is not None:
        return text, f"marginalpdbc_{delivery_date.strftime('%Y%m%d')}.1"

    best_text, best_suffix = None, None
    for suffix in range(2, MAX_SUFFIX + 1):
        candidate = _fetch_suffix(delivery_date, suffix, timeout, retries)
        if candidate is not None:
            best_text, best_suffix = candidate, suffix
    if best_text is not None:
        return best_text, f"marginalpdbc_{delivery_date.strftime('%Y%m%d')}.{best_suffix}"
    return None, None


# ---------------------------------------------------------------------------
# Parse
# ---------------------------------------------------------------------------

def parse(raw_text: str, delivery_date: date) -> list[dict]:
    rows: list[dict] = []
    lines = raw_text.strip().splitlines()
    for line in lines[1:]:  # línea 0 es la cabecera "MARGINALPDBC;"
        line = line.strip()
        if not line or line.startswith("*"):
            continue
        parts = [p for p in line.split(";") if p != ""]
        if len(parts) < 6:
            continue
        yyyy, mm, dd, period, price_es, price_pt = parts[:6]
        file_date = date(int(yyyy), int(mm), int(dd))
        if file_date != delivery_date:
            raise ValueError(f"fecha en fichero ({file_date}) no coincide con la solicitada ({delivery_date})")
        rows.append(
            {
                "period_index": int(period),
                "price_eur_mwh_es": float(price_es),
                "price_eur_mwh_pt": float(price_pt),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Validate
# ---------------------------------------------------------------------------

def validate(rows: list[dict], delivery_date: date) -> dict:
    messages: list[str] = []
    status = "OK"
    expected = expected_period_count(delivery_date)

    if len(rows) != expected:
        status = "WARN"
        messages.append(f"periodos={len(rows)} distinto de esperado={expected}")

    indices = [r["period_index"] for r in rows]
    if len(indices) != len(set(indices)):
        status = "WARN"
        messages.append("índices de periodo duplicados en el fichero")

    if len(rows) == expected and sorted(indices) != list(range(1, len(rows) + 1)):
        status = "WARN"
        messages.append("los índices de periodo no forman una secuencia contigua 1..N")

    n_null = sum(1 for r in rows if r["price_eur_mwh_es"] is None or r["price_eur_mwh_pt"] is None)
    if n_null:
        status = "WARN"
        messages.append(f"{n_null} periodos con precio nulo")

    n_out_of_band = sum(
        1
        for r in rows
        if not (PRICE_SANITY_MIN <= r["price_eur_mwh_es"] <= PRICE_SANITY_MAX)
        or not (PRICE_SANITY_MIN <= r["price_eur_mwh_pt"] <= PRICE_SANITY_MAX)
    )
    if n_out_of_band:
        status = "WARN"
        messages.append(f"{n_out_of_band} periodos con precio fuera de banda plausible [{PRICE_SANITY_MIN},{PRICE_SANITY_MAX}]")

    n_decoupled = sum(1 for r in rows if r["price_eur_mwh_es"] != r["price_eur_mwh_pt"])
    if n_decoupled:
        messages.append(f"{n_decoupled} periodos con desacoplo ES/PT (informativo, no es anomalía)")

    return {"status": status, "messages": messages, "expected_periods": expected, "actual_periods": len(rows)}


# ---------------------------------------------------------------------------
# Diccionario de datos (documentación por columna + estadísticos en vivo)
# ---------------------------------------------------------------------------

OMIE_SPOT_COLUMNS = {
    "delivery_date": ColumnDoc(
        description="Día natural (hora local Europe/Madrid) al que corresponde la entrega de energía subastada.",
        source="Campos 1-3 (año, mes, día) de cada fila del fichero marginalpdbc de OMIE.",
        dtype="DATE", kind="identifier",
    ),
    "period_index": ColumnDoc(
        description=(
            "Índice secuencial (1-based) del periodo de mercado dentro del día natural. Su unidad depende de "
            "resolution_minutes: horas 1..23/24/25 hasta el 2025-09-30; cuartos de hora 1..92/96/100 desde el "
            "2025-10-01 (armonización europea del Market Time Unit a 15 minutos)."
        ),
        source="Campo 4 ('Hora') de cada fila del fichero marginalpdbc de OMIE.",
        dtype="INTEGER", kind="identifier",
    ),
    "resolution_minutes": ColumnDoc(
        description="Duración en minutos de cada periodo de mercado: 60 hasta el 2025-09-30, 15 desde el 2025-10-01.",
        source="Calculado por resolution_minutes_for(delivery_date); no viene explícito en el fichero fuente.",
        dtype="SMALLINT", kind="categorical",
    ),
    "delivery_start_utc": ColumnDoc(
        description=(
            "Instante UTC de inicio del periodo de entrega. El cálculo es consciente del cambio de hora: los "
            "días de cambio DST tienen 23 o 25 periodos horarios reales, no 24."
        ),
        source="Calculado por period_start_utc(): medianoche local Europe/Madrid del delivery_date + (period_index-1) × resolution_minutes, en aritmética UTC.",
        dtype="TIMESTAMP", kind="timestamp",
    ),
    "price_eur_mwh_es": ColumnDoc(
        description="Precio marginal del mercado diario (day-ahead) para la zona de precio de España, en EUR/MWh.",
        source="Campo 5 de cada fila del fichero marginalpdbc de OMIE.",
        dtype="DOUBLE", kind="continuous",
    ),
    "price_eur_mwh_pt": ColumnDoc(
        description=(
            "Precio marginal del mercado diario para la zona de precio de Portugal, en EUR/MWh. Coincide con "
            "price_eur_mwh_es salvo en periodos de desacoplo entre zonas (congestión del interconector ES-PT)."
        ),
        source="Campo 6 de cada fila del fichero marginalpdbc de OMIE.",
        dtype="DOUBLE", kind="continuous",
    ),
    "source_file": ColumnDoc(
        description="Nombre del fichero fuente descargado de OMIE del que procede la fila, para trazabilidad completa del dato.",
        source="Construido por el pipeline como marginalpdbc_YYYYMMDD.1 a partir de delivery_date.",
        dtype="VARCHAR", kind="identifier",
    ),
    "ingested_at": ColumnDoc(
        description="Marca temporal UTC del momento en que esta fila se cargó (o recargó) por última vez en la base de datos.",
        source="datetime.utcnow() en el momento de la inserción, dentro de la función load().",
        dtype="TIMESTAMP", kind="timestamp",
    ),
}


def document(con: duckdb.DuckDBPyConnection) -> None:
    """Recalcula el diccionario de datos de omie_spot_prices sobre el contenido real de la tabla.
    Llamar una vez al final de un backfill o de la actualización diaria, no por cada fecha."""
    catalog.refresh(con, "omie_spot_prices", OMIE_SPOT_COLUMNS)


# ---------------------------------------------------------------------------
# Schema + load
# ---------------------------------------------------------------------------

def ensure_schema(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS omie_spot_prices (
            delivery_date       DATE NOT NULL,
            period_index        INTEGER NOT NULL,
            resolution_minutes  SMALLINT NOT NULL,
            delivery_start_utc  TIMESTAMP NOT NULL,
            price_eur_mwh_es    DOUBLE,
            price_eur_mwh_pt    DOUBLE,
            source_file         VARCHAR NOT NULL,
            ingested_at         TIMESTAMP NOT NULL,
            PRIMARY KEY (delivery_date, period_index)
        )
        """
    )


def load(con: duckdb.DuckDBPyConnection, delivery_date: date, rows: list[dict], source_file: str) -> int:
    """Nota de rendimiento: se inserta vía con.append() con un DataFrame (Appender nativo de
    DuckDB), no vía executemany(). executemany() en el driver Python de DuckDB no está
    vectorizado (ejecuta fila a fila con overhead de Python) y para tablas de miles de
    filas es ~1000x más lento que el Appender — verificado en vivo: 8928 filas pasaron de
    ~55s con executemany a ~0.01s con append()."""
    ensure_schema(con)
    resolution_minutes = resolution_minutes_for(delivery_date)
    now = datetime.utcnow()
    df = pd.DataFrame(
        [
            (
                delivery_date,
                r["period_index"],
                resolution_minutes,
                period_start_utc(delivery_date, r["period_index"], resolution_minutes).replace(tzinfo=None),
                r["price_eur_mwh_es"],
                r["price_eur_mwh_pt"],
                source_file,
                now,
            )
            for r in rows
        ],
        columns=[
            "delivery_date", "period_index", "resolution_minutes", "delivery_start_utc",
            "price_eur_mwh_es", "price_eur_mwh_pt", "source_file", "ingested_at",
        ],
    )
    con.execute("BEGIN TRANSACTION")
    try:
        con.execute("DELETE FROM omie_spot_prices WHERE delivery_date = ?", [delivery_date])
        con.append("omie_spot_prices", df)
        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK")
        raise
    return len(df)


# ---------------------------------------------------------------------------
# Orquestación de un día + registro de auditoría
# ---------------------------------------------------------------------------

def _log_run(con, delivery_date, rows_expected, rows_loaded, status, message, started_at) -> None:
    con.execute(
        """
        INSERT INTO ingestion_log
            (source, target_table, delivery_date, rows_expected, rows_loaded, status, message, started_at, finished_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        ["omie_spot", "omie_spot_prices", delivery_date, rows_expected, rows_loaded, status, message,
         started_at, datetime.utcnow()],
    )


def run_for_date(con: duckdb.DuckDBPyConnection, delivery_date: date, logger) -> dict:
    started_at = datetime.utcnow()
    ensure_schema(con)

    try:
        raw, source_file = fetch_raw(delivery_date)
    except Exception as exc:  # error de red tras reintentos
        _log_run(con, delivery_date, None, None, "ERROR", f"fallo de red: {exc}", started_at)
        logger.error("OMIE %s: fallo de red: %s", delivery_date, exc)
        return {"status": "ERROR", "delivery_date": delivery_date, "message": str(exc)}

    if raw is None:
        _log_run(con, delivery_date, None, 0, "ERROR", "fichero no publicado (ni .1 ni versiones corregidas .2-.5)", started_at)
        logger.warning("OMIE %s: fichero no publicado (ni .1 ni versiones corregidas .2-.5)", delivery_date)
        return {"status": "ERROR", "delivery_date": delivery_date, "message": "no publicado"}

    if source_file != f"marginalpdbc_{delivery_date.strftime('%Y%m%d')}.1":
        logger.warning("OMIE %s: usando versión corregida %s (.1 no disponible)", delivery_date, source_file)

    try:
        rows = parse(raw, delivery_date)
    except Exception as exc:
        _log_run(con, delivery_date, None, 0, "ERROR", f"fallo de parseo: {exc}", started_at)
        logger.error("OMIE %s: fallo de parseo: %s", delivery_date, exc)
        return {"status": "ERROR", "delivery_date": delivery_date, "message": str(exc)}

    validation = validate(rows, delivery_date)
    rows_loaded = load(con, delivery_date, rows, source_file)

    status = validation["status"] if rows_loaded else "ERROR"
    message = "; ".join(validation["messages"]) if validation["messages"] else "sin incidencias"
    _log_run(con, delivery_date, validation["expected_periods"], rows_loaded, status, message, started_at)

    log_fn = logger.info if status == "OK" else logger.warning
    log_fn(
        "OMIE %s: %s filas cargadas (esperadas %s) — %s",
        delivery_date, rows_loaded, validation["expected_periods"], message,
    )
    return {"status": status, "delivery_date": delivery_date, "rows_loaded": rows_loaded, "message": message}


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(__file__).rsplit("etl", 1)[0])
    from etl.common.db import connect
    from etl.common.logging_config import get_logger

    target_date = date.fromisoformat(sys.argv[1]) if len(sys.argv) > 1 else date.today()
    log = get_logger("omie_spot.manual", "spain")
    conn = connect("spain")
    result = run_for_date(conn, target_date, log)
    print(result)
    document(conn)
    from etl.common.db import document_ingestion_log
    document_ingestion_log(conn)
    conn.close()
