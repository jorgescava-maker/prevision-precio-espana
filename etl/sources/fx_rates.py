"""
Ingesta de tipos de cambio diarios (EUR/USD, EUR/AUD) vía el mismo endpoint de gráficos
de Yahoo Finance ya usado en `etl/sources/commodity_prices.py` — reutiliza sus funciones
`fetch_raw`/`parse`/`validate` sin modificarlas (mismo formato de respuesta, mismos
`period1`/`period2`), igual patrón que `etl/sources/germany_power_futures.py`.

Contexto (backlog de ideas #5, status.md): Brent/JKM se cotizan en USD y los futuros de
electricidad del NEM (si algún día se resuelven) cotizarían en AUD, mientras que el resto
del ecosistema (TTF, EUA, precios spot europeos) está en EUR. Sin FX, cualquier análisis
cross-market con las series europeas mezcla efecto-precio con efecto-divisa. Se guarda en
`commodities.duckdb` (no en un país concreto) por el mismo criterio que TTF/JKM/Brent/EUA:
es un insumo cross-market, no específico de un país.

Verificado en vivo (2026-08-29): `EURUSD=X` y `EURAUD=X` devuelven histórico diario real
2023-01-03→hoy vía este endpoint, mismo comportamiento que los tickers de materias
primas — no hace falta trocear por año.
"""

from __future__ import annotations

from datetime import date, datetime

import duckdb
import pandas as pd

from etl.common import catalog
from etl.common.catalog import ColumnDoc
from etl.sources.commodity_prices import fetch_raw, parse, validate

PAIRS = {
    "eur_usd": {
        "ticker": "EURUSD=X", "name": "Euro / US Dollar",
        "base_currency": "EUR", "quote_currency": "USD",
    },
    "eur_aud": {
        "ticker": "EURAUD=X", "name": "Euro / Australian Dollar",
        "base_currency": "EUR", "quote_currency": "AUD",
    },
}


# ---------------------------------------------------------------------------
# Diccionario de datos
# ---------------------------------------------------------------------------

FX_COLUMNS = {
    "pair_key": ColumnDoc(
        description="Identificador interno del par de divisas: eur_usd o eur_aud (ver PAIRS en el pipeline).",
        source="Constante de configuración, no viene por fila en la respuesta de Yahoo Finance.",
        dtype="VARCHAR", kind="categorical",
    ),
    "trade_date": ColumnDoc(
        description="Fecha de la sesión (día del cierre diario), derivada del timestamp UTC de Yahoo Finance.",
        source="Campo 'timestamp' de la respuesta de Yahoo Finance, convertido a fecha UTC.",
        dtype="DATE", kind="identifier",
    ),
    "open_price": ColumnDoc(
        description="Tipo de cambio de apertura de la sesión (unidades de quote_currency por 1 base_currency).",
        source="indicators.quote[0].open de la respuesta de Yahoo Finance.",
        dtype="DOUBLE", kind="continuous",
    ),
    "high_price": ColumnDoc(
        description="Tipo de cambio máximo de la sesión.",
        source="indicators.quote[0].high de la respuesta de Yahoo Finance.",
        dtype="DOUBLE", kind="continuous",
    ),
    "low_price": ColumnDoc(
        description="Tipo de cambio mínimo de la sesión.",
        source="indicators.quote[0].low de la respuesta de Yahoo Finance.",
        dtype="DOUBLE", kind="continuous",
    ),
    "close_price": ColumnDoc(
        description="Tipo de cambio de cierre de la sesión — valor de referencia principal.",
        source="indicators.quote[0].close de la respuesta de Yahoo Finance.",
        dtype="DOUBLE", kind="continuous",
    ),
    "volume": ColumnDoc(
        description="Volumen negociado. Habitualmente 0 o nulo — el mercado FX spot no tiene un volumen centralizado real en este endpoint.",
        source="indicators.quote[0].volume de la respuesta de Yahoo Finance.",
        dtype="BIGINT", kind="continuous",
    ),
    "base_currency": ColumnDoc(
        description="Divisa base del par (EUR en ambos pares cargados).",
        source="Constante de configuración (PAIRS en el pipeline).",
        dtype="VARCHAR", kind="categorical",
    ),
    "quote_currency": ColumnDoc(
        description="Divisa de cotización: cuántas unidades de esta divisa vale 1 unidad de base_currency (USD o AUD).",
        source="Constante de configuración (PAIRS en el pipeline).",
        dtype="VARCHAR", kind="categorical",
    ),
    "ticker": ColumnDoc(
        description="Símbolo de Yahoo Finance usado para obtener la fila, para trazabilidad.",
        source="Constante de configuración (PAIRS en el pipeline).",
        dtype="VARCHAR", kind="identifier",
    ),
    "ingested_at": ColumnDoc(
        description="Marca temporal UTC del momento en que esta fila se cargó (o recargó) por última vez en la base de datos.",
        source="datetime.utcnow() en el momento de la inserción, dentro de la función load().",
        dtype="TIMESTAMP", kind="timestamp",
    ),
}


def document(con: duckdb.DuckDBPyConnection) -> None:
    catalog.refresh(con, "fx_rates", FX_COLUMNS)


# ---------------------------------------------------------------------------
# Schema + load
# ---------------------------------------------------------------------------

def ensure_schema(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS fx_rates (
            pair_key        VARCHAR NOT NULL,
            trade_date      DATE NOT NULL,
            open_price      DOUBLE,
            high_price      DOUBLE,
            low_price       DOUBLE,
            close_price     DOUBLE NOT NULL,
            volume          BIGINT,
            base_currency   VARCHAR NOT NULL,
            quote_currency  VARCHAR NOT NULL,
            ticker          VARCHAR NOT NULL,
            ingested_at     TIMESTAMP NOT NULL,
            PRIMARY KEY (pair_key, trade_date)
        )
        """
    )


def load(con: duckdb.DuckDBPyConnection, pair_key: str, rows: list[dict]) -> int:
    ensure_schema(con)
    if not rows:
        return 0
    cfg = PAIRS[pair_key]
    now = datetime.utcnow()
    df = pd.DataFrame(
        [
            (pair_key, r["trade_date"], r["open_price"], r["high_price"], r["low_price"],
             r["close_price"], r["volume"], cfg["base_currency"], cfg["quote_currency"], cfg["ticker"], now)
            for r in rows
        ],
        columns=["pair_key", "trade_date", "open_price", "high_price", "low_price",
                 "close_price", "volume", "base_currency", "quote_currency", "ticker", "ingested_at"],
    )
    min_d, max_d = df["trade_date"].min(), df["trade_date"].max()

    con.execute("BEGIN TRANSACTION")
    try:
        con.execute(
            "DELETE FROM fx_rates WHERE pair_key = ? AND trade_date >= ? AND trade_date <= ?",
            [pair_key, min_d, max_d],
        )
        con.append("fx_rates", df)
        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK")
        raise
    return len(df)


# ---------------------------------------------------------------------------
# Orquestación + registro de auditoría
# ---------------------------------------------------------------------------

def _log_run(con, pair_key, source_chunk, rows_loaded, status, message, started_at) -> None:
    con.execute(
        """
        INSERT INTO ingestion_log
            (source, target_table, delivery_date, rows_expected, rows_loaded, status, message, started_at, finished_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [f"fx:{pair_key}", "fx_rates", date.today(), None, rows_loaded, status, message,
         started_at, datetime.utcnow()],
    )


def run_for_range(con: duckdb.DuckDBPyConnection, pair_key: str, period1: int, period2: int, logger) -> dict:
    started_at = datetime.utcnow()
    ensure_schema(con)
    ticker = PAIRS[pair_key]["ticker"]

    try:
        result = fetch_raw(ticker, period1, period2)
        rows = parse(result)
    except Exception as exc:
        _log_run(con, pair_key, f"{period1}-{period2}", 0, "ERROR", str(exc), started_at)
        logger.error("FX %s: %s", pair_key, exc)
        return {"status": "ERROR", "pair": pair_key, "message": str(exc)}

    validation = validate(rows)
    rows_loaded = load(con, pair_key, rows)

    status = validation["status"]
    message = "; ".join(validation["messages"]) if validation["messages"] else "sin incidencias"
    _log_run(con, pair_key, f"{period1}-{period2}", rows_loaded, status, message, started_at)

    log_fn = logger.info if status == "OK" else logger.warning
    log_fn("FX %s: %s filas cargadas — %s", pair_key, rows_loaded, message)
    return {"status": status, "pair": pair_key, "rows_loaded": rows_loaded, "message": message}


if __name__ == "__main__":
    import sys

    from etl.common.db import connect, document_ingestion_log
    from etl.common.logging_config import get_logger

    key = sys.argv[1] if len(sys.argv) > 1 else "eur_usd"
    log = get_logger("fx_rates.manual", "commodities")
    conn = connect("commodities")
    p1 = int(datetime(2023, 1, 1).timestamp())
    p2 = int(datetime.utcnow().timestamp())
    result = run_for_range(conn, key, p1, p2, log)
    print(result)
    document(conn)
    document_ingestion_log(conn)
    conn.close()
