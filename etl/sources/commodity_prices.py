"""
Ingesta de precios diarios de materias primas de referencia (gas TTF, LNG asiático
JKM, crudo Brent, carbono EUA) vía el endpoint de gráficos de Yahoo Finance.

IMPORTANTE — calidad de fuente distinta al resto del ecosistema: este endpoint
(`query1.finance.yahoo.com/v8/finance/chart/...`) es una API interna no documentada
ni respaldada oficialmente por Yahoo (a diferencia de OMIE/AEMO/SMARD/ENTSO-E, que son
operadores de mercado o reguladores publicando sus propios datos oficiales). Es de uso
extendido en la comunidad de datos financieros pero puede cambiar o dejar de responder
sin aviso. Se documenta aquí para que quede explícito el escalón de confianza distinto.

No hay un equivalente diario gratuito y oficial mejor localizado para TTF y JKM (ver
docs/findings.md #10 sobre la ausencia de fuente pública para gas en AEMO — el portal
de gas de AEMO está protegido por Cloudflare y devuelve 403). Brent sí tiene una
alternativa oficial (FRED, serie DCOILBRENTEU) que se documenta como opción futura si
se quiere subir el nivel de rigor de esa serie en concreto.

Referencia elegida para Australia: JKM (Platts Japan-Korea Marker, LNG spot Asia-
Pacífico) en vez de un hub doméstico australiano — el mercado de gas del este de
Australia (Queensland/Wallumbilla) está cada vez más ligado a la paridad de
exportación de GNL, y JKM es el marcador estándar de esa referencia. Es una decisión
de diseño explícita, no la ausencia de alternativa: documentada para que el usuario
pueda cuestionarla si prefiere otro proxy.

Carbón (API2, añadido 2026-08-28 para el estudio E2 de la agenda de investigación —
fuel-switching gas/carbón): `MTF=F` ("Coal (API2) CIF ARA", ARGUS-McCloskey) tiene
histórico diario real vía este mismo endpoint, mismo patrón que TTF/Brent — pero
**verificado en vivo que Yahoo dejó de actualizar este ticker el 2025-12-26** (todos
los `close` posteriores son `null` en la respuesta cruda, incluso pidiendo hasta hoy;
`regularMarketTime` de la metadata tampoco es reciente). No es un fallo del pipeline:
el dato real 2023-01-03→2025-12-26 (750 sesiones) es válido y sigue siendo útil para
E2, pero la actualización diaria no traerá filas nuevas hasta que (si) Yahoo retome
la publicación — `validate()` lo registrará como WARN "sin filas en este bloque"
indefinidamente, comportamiento esperado, no investigar de nuevo salvo que se
sospeche que Yahoo ya lo arregló.

Carbono (EUA, Fase 11 del roadmap): no existe un ticker de futuros EUA nativo en
Yahoo Finance (verificado en vivo — solo aparecen índices `^ICEEUA`/`^ICEUANI`/
`^ICEUACN` sin histórico diario descargable vía este endpoint, un único punto
"instantáneo" cada uno). Se usa en su lugar `CO2.L` (SparkChange Physical Carbon EUA,
ETC físicamente respaldado por derechos de emisión reales, cotizado en LSE en EUR):
sigue el precio spot de EUA de cerca (con un pequeño drag por comisión de gestión del
ETC, no de mercado), y da cobertura diaria completa 2023-01-03 → hoy verificada en
vivo (923 puntos, 1 solo nulo). Ver docs/findings.md para el detalle de por qué se
descartaron los índices ICE.
"""

from __future__ import annotations

from datetime import date, datetime, timezone

import duckdb
import pandas as pd
import requests

from etl.common import catalog
from etl.common.catalog import ColumnDoc

CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"

COMMODITIES = {
    "ttf_gas": {
        "ticker": "TTF=F", "name": "Dutch TTF Natural Gas (front-month future)",
        "currency": "EUR", "unit": "EUR/MWh",
    },
    "jkm_lng": {
        "ticker": "JKM=F", "name": "Platts Japan-Korea Marker LNG (Asia-Pacífico, referencia para Australia)",
        "currency": "USD", "unit": "USD/MMBtu",
    },
    "brent_crude": {
        "ticker": "BZ=F", "name": "Brent Crude Oil (front-month future)",
        "currency": "USD", "unit": "USD/bbl",
    },
    "eua_carbon": {
        "ticker": "CO2.L", "name": "EU Allowances (EUA) — SparkChange Physical Carbon ETC, proxy del precio spot",
        "currency": "EUR", "unit": "EUR/tCO2",
    },
    "api2_coal": {
        "ticker": "MTF=F", "name": "Coal (API2) CIF ARA (ARGUS-McCloskey, front-month future) — necesario para el fuel-switching de findings.md/estudio E2",
        "currency": "USD", "unit": "USD/t",
    },
}

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

def fetch_raw(ticker: str, period1: int, period2: int, timeout: int = 30) -> dict:
    params = {"period1": period1, "period2": period2, "interval": "1d"}
    resp = _session.get(CHART_URL.format(ticker=ticker), params=params, timeout=timeout)
    resp.raise_for_status()
    payload = resp.json()
    if payload["chart"].get("error"):
        raise ValueError(f"Yahoo Finance error para {ticker}: {payload['chart']['error']}")
    return payload["chart"]["result"][0]


# ---------------------------------------------------------------------------
# Parse
# ---------------------------------------------------------------------------

def parse(result: dict) -> list[dict]:
    timestamps = result["timestamp"]
    quote = result["indicators"]["quote"][0]
    rows: list[dict] = []
    for i, ts in enumerate(timestamps):
        close = quote["close"][i]
        if close is None:
            continue  # sin sesión de negociación ese día (festivo / contrato ilíquido)
        rows.append(
            {
                "trade_date": datetime.fromtimestamp(ts, tz=timezone.utc).date(),
                "open_price": quote["open"][i],
                "high_price": quote["high"][i],
                "low_price": quote["low"][i],
                "close_price": close,
                "volume": quote["volume"][i],
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Validate
# ---------------------------------------------------------------------------

def validate(rows: list[dict]) -> dict:
    messages: list[str] = []
    status = "OK"
    if not rows:
        return {"status": "WARN", "messages": ["sin filas en este bloque"], "actual_rows": 0}

    n_negative = sum(1 for r in rows if r["close_price"] < 0)
    if n_negative:
        messages.append(f"{n_negative} días con precio negativo (informativo, revisar si es TTF/JKM o Brent en episodio extremo)")

    dates = sorted(r["trade_date"] for r in rows)
    gaps_over_5d = sum(1 for a, b in zip(dates, dates[1:]) if (b - a).days > 5)
    if gaps_over_5d:
        status = "WARN"
        messages.append(f"{gaps_over_5d} huecos de más de 5 días naturales entre cotizaciones consecutivas")

    return {"status": status, "messages": messages, "actual_rows": len(rows)}


# ---------------------------------------------------------------------------
# Diccionario de datos
# ---------------------------------------------------------------------------

COMMODITY_COLUMNS = {
    "commodity_key": ColumnDoc(
        description="Identificador interno de la materia prima: ttf_gas, jkm_lng, brent_crude o eua_carbon (ver COMMODITIES en el pipeline).",
        source="Constante de configuración, no viene por fila en la respuesta de Yahoo Finance.",
        dtype="VARCHAR", kind="categorical",
    ),
    "trade_date": ColumnDoc(
        description="Fecha de la sesión de negociación (día del cierre diario), derivada del timestamp UTC de Yahoo Finance (America/New_York para TTF/JKM/Brent, Europe/London para EUA — cada bolsa de origen).",
        source="Campo 'timestamp' de la respuesta de Yahoo Finance, convertido a fecha UTC.",
        dtype="DATE", kind="identifier",
    ),
    "open_price": ColumnDoc(
        description="Precio de apertura de la sesión, en la unidad nativa del contrato (ver columna unit).",
        source="indicators.quote[0].open de la respuesta de Yahoo Finance.",
        dtype="DOUBLE", kind="continuous",
    ),
    "high_price": ColumnDoc(
        description="Precio máximo de la sesión.",
        source="indicators.quote[0].high de la respuesta de Yahoo Finance.",
        dtype="DOUBLE", kind="continuous",
    ),
    "low_price": ColumnDoc(
        description="Precio mínimo de la sesión.",
        source="indicators.quote[0].low de la respuesta de Yahoo Finance.",
        dtype="DOUBLE", kind="continuous",
    ),
    "close_price": ColumnDoc(
        description="Precio de cierre de la sesión — es el valor de referencia principal para los estudios del catálogo.",
        source="indicators.quote[0].close de la respuesta de Yahoo Finance.",
        dtype="DOUBLE", kind="continuous",
    ),
    "volume": ColumnDoc(
        description="Volumen negociado en la sesión. A menudo 0 o nulo en JKM por ser un contrato poco líquido.",
        source="indicators.quote[0].volume de la respuesta de Yahoo Finance.",
        dtype="BIGINT", kind="continuous",
    ),
    "currency": ColumnDoc(
        description="Divisa del precio (EUR para TTF y EUA, USD para JKM y Brent).",
        source="Constante de configuración — el campo 'currency' de la respuesta de Yahoo es poco fiable (nulo para JKM en pruebas en vivo).",
        dtype="VARCHAR", kind="categorical",
    ),
    "unit": ColumnDoc(
        description="Unidad física del precio: EUR/MWh (TTF), USD/MMBtu (JKM), USD/bbl (Brent), EUR/tCO2 (EUA).",
        source="Constante de configuración, no viene en la respuesta de Yahoo Finance.",
        dtype="VARCHAR", kind="categorical",
    ),
    "ticker": ColumnDoc(
        description="Símbolo de Yahoo Finance usado para obtener la fila, para trazabilidad.",
        source="Constante de configuración (COMMODITIES en el pipeline).",
        dtype="VARCHAR", kind="identifier",
    ),
    "ingested_at": ColumnDoc(
        description="Marca temporal UTC del momento en que esta fila se cargó (o recargó) por última vez en la base de datos.",
        source="datetime.utcnow() en el momento de la inserción, dentro de la función load().",
        dtype="TIMESTAMP", kind="timestamp",
    ),
}


def document(con: duckdb.DuckDBPyConnection) -> None:
    catalog.refresh(con, "commodity_prices", COMMODITY_COLUMNS)


# ---------------------------------------------------------------------------
# Schema + load
# ---------------------------------------------------------------------------

def ensure_schema(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS commodity_prices (
            commodity_key  VARCHAR NOT NULL,
            trade_date     DATE NOT NULL,
            open_price     DOUBLE,
            high_price     DOUBLE,
            low_price      DOUBLE,
            close_price    DOUBLE NOT NULL,
            volume         BIGINT,
            currency       VARCHAR NOT NULL,
            unit           VARCHAR NOT NULL,
            ticker         VARCHAR NOT NULL,
            ingested_at    TIMESTAMP NOT NULL,
            PRIMARY KEY (commodity_key, trade_date)
        )
        """
    )


def load(con: duckdb.DuckDBPyConnection, commodity_key: str, rows: list[dict]) -> int:
    ensure_schema(con)
    if not rows:
        return 0
    cfg = COMMODITIES[commodity_key]
    now = datetime.utcnow()
    df = pd.DataFrame(
        [
            (commodity_key, r["trade_date"], r["open_price"], r["high_price"], r["low_price"],
             r["close_price"], r["volume"], cfg["currency"], cfg["unit"], cfg["ticker"], now)
            for r in rows
        ],
        columns=["commodity_key", "trade_date", "open_price", "high_price", "low_price",
                 "close_price", "volume", "currency", "unit", "ticker", "ingested_at"],
    )
    min_d, max_d = df["trade_date"].min(), df["trade_date"].max()

    con.execute("BEGIN TRANSACTION")
    try:
        con.execute(
            "DELETE FROM commodity_prices WHERE commodity_key = ? AND trade_date >= ? AND trade_date <= ?",
            [commodity_key, min_d, max_d],
        )
        con.append("commodity_prices", df)
        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK")
        raise
    return len(df)


# ---------------------------------------------------------------------------
# Orquestación + registro de auditoría
# ---------------------------------------------------------------------------

def _log_run(con, commodity_key, source_chunk, rows_loaded, status, message, started_at) -> None:
    con.execute(
        """
        INSERT INTO ingestion_log
            (source, target_table, delivery_date, rows_expected, rows_loaded, status, message, started_at, finished_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [f"commodity:{commodity_key}", "commodity_prices", date.today(), None, rows_loaded, status, message,
         started_at, datetime.utcnow()],
    )


def run_for_range(con: duckdb.DuckDBPyConnection, commodity_key: str, period1: int, period2: int, logger) -> dict:
    started_at = datetime.utcnow()
    ensure_schema(con)
    ticker = COMMODITIES[commodity_key]["ticker"]
    source_chunk = f"{period1}-{period2}"

    try:
        result = fetch_raw(ticker, period1, period2)
        rows = parse(result)
    except Exception as exc:
        _log_run(con, commodity_key, source_chunk, 0, "ERROR", str(exc), started_at)
        logger.error("Commodity %s: %s", commodity_key, exc)
        return {"status": "ERROR", "commodity": commodity_key, "message": str(exc)}

    validation = validate(rows)
    rows_loaded = load(con, commodity_key, rows)

    status = validation["status"]
    message = "; ".join(validation["messages"]) if validation["messages"] else "sin incidencias"
    _log_run(con, commodity_key, source_chunk, rows_loaded, status, message, started_at)

    log_fn = logger.info if status == "OK" else logger.warning
    log_fn("Commodity %s: %s filas cargadas — %s", commodity_key, rows_loaded, message)
    return {"status": status, "commodity": commodity_key, "rows_loaded": rows_loaded, "message": message}


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(__file__).rsplit("etl", 1)[0])
    from etl.common.db import connect, document_ingestion_log
    from etl.common.logging_config import get_logger

    key = sys.argv[1] if len(sys.argv) > 1 else "ttf_gas"
    log = get_logger("commodity_prices.manual", "commodities")
    conn = connect("commodities")
    p1 = int(datetime(2023, 1, 1).timestamp())
    p2 = int(datetime.utcnow().timestamp())
    result = run_for_range(conn, key, p1, p2, log)
    print(result)
    document(conn)
    document_ingestion_log(conn)
    conn.close()
