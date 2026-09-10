"""
Ingesta de previsión día-adelantada (D-1) de demanda, eólica y solar fotovoltaica de
España (Península) vía la API de e·sios/REE, reutilizando fetch_indicator/_parse_values
de etl/sources/esios_load_generation.py.

Es el insumo de la Fase 8 del roadmap (previsiones D-1) para España — la contraparte de
etl/sources/esios_load_generation.py (que trae el dato REAL, no la previsión). Sirve al
estudio B1 (comparar `*_forecast_mw` de aquí contra `demand_mw` / `value_mw` de
esios_load_generation por intervalo, calculando el error de previsión).

Decisiones de indicador, verificadas en vivo el 2026-08-28 (e·sios tiene, igual que en
etl/sources/esios_load_generation.py, varias familias de indicadores para lo mismo —
legacy y vigentes mezcladas sin aviso):
1. Se eligieron los indicadores explícitamente etiquetados "D+1" (día-adelantado, un
   único valor de previsión por intervalo, fijado la víspera): 1775 ("Previsión diaria
   D+1 demanda"), 1777 ("Previsión diaria D+1 eólica"), 1779 ("Previsión diaria D+1
   fotovoltaica"). Los tres devuelven 25 puntos/día (resolución horaria) y tienen
   histórico verificado hasta 2019 (más que suficiente para el rango 2023→hoy del
   proyecto).
2. Familias descartadas (NO usar si se retoma este pipeline): 460/541/542/543
   ("Previsión diaria de la demanda"/"eólica"/"solar FV"/"solar térmica", sin
   sufijo D+1) devuelven 97 puntos/día (5 en 5 min extrapolados desde el escalón
   horario) pero no está verificado que sean el mismo valor D-1 congelado en vez de
   una previsión que se va actualizando — no se usan porque 1775/1777/1779 son más
   explícitos sobre el horizonte. 1776/1778/1780 ("Previsión intradiaria H+3...") son
   directamente el feed intradiario que actualiza la previsión conforme se acerca la
   entrega — NO es D-1, deliberadamente excluido. 10358/10359 (eólica+fotovoltaica
   combinada D+1/H+3) son solo la suma de los dos indicadores ya cargados por
   separado — redundante.
3. No se incluye solar térmica (543 o equivalente D+1): no se encontró una familia
   D+1 explícita para ella (solo aparece en la familia legacy sin sufijo D+1); su peso
   en el mix español es marginal frente a fotovoltaica (ver esios_generation_by_type,
   category_key='solar_termica' vs 'solar_fotovoltaica'). Si se quiere en el futuro,
   revisar si e·sios publicó desde entonces un indicador D+1 dedicado.
4. Mismas notas de formato que esios_load_generation.py: JSON con datetime_utc propio
   por punto (no hay que reconstruir con resolution+position), geo_id=8741 (Península).
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo

import duckdb
import pandas as pd

from etl.common import catalog
from etl.common.catalog import ColumnDoc
from etl.sources.esios_load_generation import _parse_values, fetch_indicator

DB_NAME = "spain"
TZ_MADRID = ZoneInfo("Europe/Madrid")

# category_key -> (indicator_id, nombre legible)
FORECAST_CATEGORIES = {
    "demanda": (1775, "Demanda"),
    "eolica": (1777, "Eólica"),
    "solar_fotovoltaica": (1779, "Solar fotovoltaica"),
}

_session_sleep = 0.3


# ---------------------------------------------------------------------------
# Diccionario de datos
# ---------------------------------------------------------------------------

def document(con: duckdb.DuckDBPyConnection) -> None:
    cols = {
        "interval_start_utc": ColumnDoc(
            description="Instante UTC de inicio del intervalo horario previsto.",
            source="Campo datetime_utc de la API de e·sios.",
            dtype="TIMESTAMP", kind="identifier",
        ),
        "interval_start_local": ColumnDoc(
            description="Mismo instante en hora local de España peninsular (Europe/Madrid).",
            source="Calculado por el pipeline a partir de interval_start_utc.",
            dtype="TIMESTAMP", kind="timestamp",
        ),
        "category_key": ColumnDoc(
            description="Categoría prevista: 'demanda', 'eolica' o 'solar_fotovoltaica' (FORECAST_CATEGORIES).",
            source="Constante de configuración del pipeline, no viene en la respuesta de e·sios.",
            dtype="VARCHAR", kind="categorical",
        ),
        "category_name": ColumnDoc(
            description="Nombre legible de la categoría.",
            source="Constante de configuración del pipeline.",
            dtype="VARCHAR", kind="categorical",
        ),
        "forecast_mw": ColumnDoc(
            description="Previsión día-adelantada (D+1, fijada la víspera de la entrega) de la categoría, en MW.",
            source="Campo value del indicador correspondiente de e·sios (1775/1777/1779), geo_id=8741 (Península).",
            dtype="DOUBLE", kind="continuous",
        ),
        "indicator_id": ColumnDoc(
            description="Identificador numérico del indicador de e·sios de origen, para trazabilidad si e·sios cambia de convención.",
            source="Constante de configuración del pipeline.",
            dtype="INTEGER", kind="categorical",
        ),
        "source_chunk": ColumnDoc(
            description="Identificador del bloque de petición de origen, para trazabilidad.",
            source="Parámetro de la petición a la API de e·sios.",
            dtype="VARCHAR", kind="identifier",
        ),
        "ingested_at": ColumnDoc(
            description="Marca temporal UTC de la última carga/recarga de esta fila.",
            source="datetime.utcnow() en el momento de la inserción.",
            dtype="TIMESTAMP", kind="timestamp",
        ),
    }
    catalog.refresh(con, "esios_forecast", cols)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

def ensure_schema(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS esios_forecast (
            interval_start_utc    TIMESTAMP NOT NULL,
            interval_start_local  TIMESTAMP NOT NULL,
            category_key          VARCHAR NOT NULL,
            category_name         VARCHAR NOT NULL,
            forecast_mw           DOUBLE NOT NULL,
            indicator_id          INTEGER NOT NULL,
            source_chunk          VARCHAR NOT NULL,
            ingested_at           TIMESTAMP NOT NULL,
            PRIMARY KEY (interval_start_utc, category_key)
        )
        """
    )


# ---------------------------------------------------------------------------
# Load
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


def _run_category_for_range(con: duckdb.DuckDBPyConnection, category_key: str, start_iso: str, end_iso: str, logger) -> dict:
    started_at = datetime.utcnow()
    indicator_id, category_name = FORECAST_CATEGORIES[category_key]
    source_chunk = f"{start_iso}-{end_iso}"
    source = f"esios_forecast:{category_key}"

    try:
        raw_values = fetch_indicator(indicator_id, start_iso, end_iso)
        rows, n_other_geo = _parse_values(raw_values)
    except Exception as exc:
        _log_run(con, source, "esios_forecast", 0, "ERROR", str(exc), started_at)
        logger.error("%s %s: %s", source, source_chunk, exc)
        return {"status": "ERROR", "message": str(exc)}

    messages = []
    status = "OK"
    if n_other_geo:
        status = "WARN"
        messages.append(f"{n_other_geo} puntos descartados por geo_id distinto de Península")
    if not rows:
        status = "WARN"
        messages.append("sin filas en este bloque")

    now = datetime.utcnow()
    rows_loaded = 0
    if rows:
        df = pd.DataFrame(
            [
                (
                    r["interval_start_utc"], r["interval_start_utc"].replace(tzinfo=timezone.utc).astimezone(TZ_MADRID).replace(tzinfo=None),
                    category_key, category_name, r["value"], indicator_id, source_chunk, now,
                )
                for r in rows
            ],
            columns=["interval_start_utc", "interval_start_local", "category_key", "category_name",
                     "forecast_mw", "indicator_id", "source_chunk", "ingested_at"],
        )
        min_ts, max_ts = df["interval_start_utc"].min(), df["interval_start_utc"].max()
        con.execute("BEGIN TRANSACTION")
        try:
            con.execute(
                "DELETE FROM esios_forecast WHERE category_key = ? AND interval_start_utc >= ? AND interval_start_utc <= ?",
                [category_key, min_ts, max_ts],
            )
            con.append("esios_forecast", df)
            con.execute("COMMIT")
        except Exception:
            con.execute("ROLLBACK")
            raise
        rows_loaded = len(df)

    message = "; ".join(messages) if messages else "sin incidencias"
    _log_run(con, source, "esios_forecast", rows_loaded, status, message, started_at)
    log_fn = logger.info if status == "OK" else logger.warning
    log_fn("%s %s: %s filas cargadas — %s", source, source_chunk, rows_loaded, message)
    return {"status": status, "rows_loaded": rows_loaded, "message": message}


def run_forecast_for_range(con: duckdb.DuckDBPyConnection, start_iso: str, end_iso: str, logger) -> dict:
    """Ejecuta las 3 categorías (demanda, eólica, solar fotovoltaica) para el mismo rango."""
    import time

    ensure_schema(con)
    results = {}
    for category_key in FORECAST_CATEGORIES:
        results[category_key] = _run_category_for_range(con, category_key, start_iso, end_iso, logger)
        time.sleep(_session_sleep)

    rows_loaded = sum(r.get("rows_loaded", 0) for r in results.values())
    if any(r["status"] == "ERROR" for r in results.values()):
        status = "ERROR"
    elif any(r["status"] == "WARN" for r in results.values()):
        status = "WARN"
    else:
        status = "OK"
    return {"status": status, "rows_loaded": rows_loaded, "by_category": results}


if __name__ == "__main__":
    from etl.common.db import connect, document_ingestion_log
    from etl.common.logging_config import get_logger

    log = get_logger("esios_forecast.manual", DB_NAME)
    conn = connect(DB_NAME)
    print(run_forecast_for_range(conn, "2023-01-01T00:00:00", "2023-01-08T00:00:00", log))
    document(conn)
    document_ingestion_log(conn)
    conn.close()
