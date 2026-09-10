"""
Calendario de festivos nacionales/regionales por mercado — insumo para el estudio A2
del catálogo de investigación (efecto festivo sobre demanda).

A diferencia de todas las demás fuentes del ecosistema, esto NO es una ingesta: no hay
llamada de red ni fuente externa que pueda fallar o tener huecos. Se calcula localmente
con la librería `holidays` (reglas oficiales por país/subdivisión, incluye festivos
móviles como Pascua). Se mantiene el mismo patrón fetch/parse/validate/load que el resto
de pipelines por consistencia y para que aparezca igual en ingestion_log.

Simplificación deliberada, mismo criterio que weather_common.py: para los 4 mercados
europeos de zona única (España/Alemania/Francia/Países Bajos) se usan los festivos
NACIONALES (los que rigen en todo el país), no los festivos regionales de cada
Land/región — en Alemania esto excluye festivos que solo aplican en algunos Länder
(p.ej. Fronleichnam, Reformationstag): importan para consumo local, pero no hay forma de
agregarlos a una demanda de zona única sin una ponderación por población/carga que el
proyecto no tiene. NEM/Australia SÍ se resuelve por subregión de precio
(nsw1/qld1/sa1/tas1/vic1, mismas claves que weather_common.py y aemo_spot.py) porque los
festivos SÍ difieren de forma relevante entre estados australianos y cada subregión ya
tiene su propio precio/demanda en el ecosistema.

Rango de años cargado: YEAR_START-YEAR_END, amplio a propósito por ser cálculo y no
ingesta (coste marginal nulo) — cubre cualquier backtest anterior a 2023 y da margen de
planificación futura.
"""

from __future__ import annotations

from datetime import date, datetime

import duckdb
import holidays

from etl.common import catalog
from etl.common.catalog import ColumnDoc

YEAR_START = 2015
YEAR_END = 2036

LOCATIONS = {
    "spain": {"db": "spain", "country": "ES", "subdiv": None},
    "germany": {"db": "germany", "country": "DE", "subdiv": None},
    "france": {"db": "france", "country": "FR", "subdiv": None},
    "netherlands": {"db": "netherlands", "country": "NL", "subdiv": None},
    # NEM Australia: por subregión de precio, no por país (ver docstring del módulo).
    "nsw1": {"db": "nem", "country": "AU", "subdiv": "NSW"},
    "qld1": {"db": "nem", "country": "AU", "subdiv": "QLD"},
    "sa1": {"db": "nem", "country": "AU", "subdiv": "SA"},
    "tas1": {"db": "nem", "country": "AU", "subdiv": "TAS"},
    "vic1": {"db": "nem", "country": "AU", "subdiv": "VIC"},
}


# ---------------------------------------------------------------------------
# Fetch + parse (cálculo local, sin red)
# ---------------------------------------------------------------------------

def compute(location_key: str) -> list[dict]:
    cfg = LOCATIONS[location_key]
    cal = holidays.country_holidays(
        cfg["country"], subdiv=cfg["subdiv"], years=range(YEAR_START, YEAR_END + 1)
    )
    return [
        {"holiday_date": d, "holiday_name": name, "country_code": cfg["country"]}
        for d, name in sorted(cal.items())
    ]


# ---------------------------------------------------------------------------
# Validate
# ---------------------------------------------------------------------------

def validate(rows: list[dict]) -> dict:
    if not rows:
        return {"status": "ERROR", "messages": ["sin festivos calculados"], "actual_rows": 0}

    n_years = YEAR_END - YEAR_START + 1
    avg_per_year = len(rows) / n_years
    messages: list[str] = []
    status = "OK"
    if avg_per_year < 4 or avg_per_year > 25:
        status = "WARN"
        messages.append(
            f"media de {avg_per_year:.1f} festivos/año, fuera del rango esperado (4-25) "
            "— revisar librería 'holidays'"
        )

    return {"status": status, "messages": messages, "actual_rows": len(rows)}


# ---------------------------------------------------------------------------
# Diccionario de datos
# ---------------------------------------------------------------------------

HOLIDAY_COLUMNS = {
    "location_key": ColumnDoc(
        description="Zona representada: spain/germany/france/netherlands (país, festivos nacionales únicamente) o nsw1/qld1/sa1/tas1/vic1 (subregión NEM, festivos del estado australiano correspondiente — mismas claves que weather_common.py).",
        source="Constante de configuración (LOCATIONS en el pipeline).",
        dtype="VARCHAR", kind="categorical",
    ),
    "holiday_date": ColumnDoc(
        description="Fecha del festivo.",
        source="Librería 'holidays' (reglas oficiales por país/subdivisión, incluye festivos móviles como Pascua).",
        dtype="DATE", kind="identifier",
    ),
    "holiday_name": ColumnDoc(
        description="Nombre del festivo en el idioma nativo del país de origen (inglés para Australia).",
        source="Librería 'holidays'.",
        dtype="VARCHAR", kind="categorical",
    ),
    "country_code": ColumnDoc(
        description="Código de país ISO 3166-1 alfa-2 (ES/DE/FR/NL/AU).",
        source="Constante de configuración (LOCATIONS en el pipeline).",
        dtype="VARCHAR", kind="categorical",
    ),
    "ingested_at": ColumnDoc(
        description="Marca temporal UTC del momento en que esta fila se cargó (o recargó) por última vez en la base de datos.",
        source="datetime.utcnow() en el momento de la inserción, dentro de la función load().",
        dtype="TIMESTAMP", kind="timestamp",
    ),
}


def document(con: duckdb.DuckDBPyConnection) -> None:
    catalog.refresh(con, "holiday_calendar", HOLIDAY_COLUMNS)


# ---------------------------------------------------------------------------
# Schema + load
# ---------------------------------------------------------------------------

def ensure_schema(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS holiday_calendar (
            location_key   VARCHAR NOT NULL,
            holiday_date   DATE NOT NULL,
            holiday_name   VARCHAR NOT NULL,
            country_code   VARCHAR NOT NULL,
            ingested_at    TIMESTAMP NOT NULL,
            PRIMARY KEY (location_key, holiday_date)
        )
        """
    )


def load(con: duckdb.DuckDBPyConnection, location_key: str, rows: list[dict]) -> int:
    ensure_schema(con)
    if not rows:
        return 0
    now = datetime.utcnow()

    con.execute("BEGIN TRANSACTION")
    try:
        con.execute("DELETE FROM holiday_calendar WHERE location_key = ?", [location_key])
        con.executemany(
            """
            INSERT INTO holiday_calendar (location_key, holiday_date, holiday_name, country_code, ingested_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            [(location_key, r["holiday_date"], r["holiday_name"], r["country_code"], now) for r in rows],
        )
        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK")
        raise
    return len(rows)


# ---------------------------------------------------------------------------
# Orquestación + registro de auditoría
# ---------------------------------------------------------------------------

def _log_run(con, location_key, rows_loaded, status, message, started_at) -> None:
    con.execute(
        """
        INSERT INTO ingestion_log
            (source, target_table, delivery_date, rows_expected, rows_loaded, status, message, started_at, finished_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [f"holiday_calendar:{location_key}", "holiday_calendar", date.today(), None, rows_loaded, status, message,
         started_at, datetime.utcnow()],
    )


def run_for_location(con: duckdb.DuckDBPyConnection, location_key: str, logger) -> dict:
    started_at = datetime.utcnow()
    ensure_schema(con)

    try:
        rows = compute(location_key)
    except Exception as exc:
        _log_run(con, location_key, 0, "ERROR", str(exc), started_at)
        logger.error("Holiday calendar %s: %s", location_key, exc)
        return {"status": "ERROR", "location": location_key, "message": str(exc)}

    validation = validate(rows)
    rows_loaded = load(con, location_key, rows)

    status = validation["status"]
    message = "; ".join(validation["messages"]) if validation["messages"] else "sin incidencias"
    _log_run(con, location_key, rows_loaded, status, message, started_at)

    log_fn = logger.info if status == "OK" else logger.warning
    log_fn("Holiday calendar %s: %s festivos cargados — %s", location_key, rows_loaded, message)
    return {"status": status, "location": location_key, "rows_loaded": rows_loaded, "message": message}


def run_all(logger) -> dict:
    """Recalcula y recarga el calendario completo para todas las ubicaciones, agrupando
    por base de datos para abrir una única conexión por fichero .duckdb."""
    from etl.common.db import connect, document_ingestion_log

    dbs: dict[str, list[str]] = {}
    for key, cfg in LOCATIONS.items():
        dbs.setdefault(cfg["db"], []).append(key)

    summary = {"OK": 0, "WARN": 0, "ERROR": 0}
    for db_name, keys in dbs.items():
        con = connect(db_name)
        for key in keys:
            result = run_for_location(con, key, logger)
            summary[result["status"]] = summary.get(result["status"], 0) + 1
        document(con)
        document_ingestion_log(con)
        con.close()
    return summary


if __name__ == "__main__":
    from etl.common.logging_config import get_logger

    log = get_logger("holiday_calendar.manual", "orchestrator")
    print(run_all(log))
