"""
Ingesta de demanda real y generación real por tecnología de España (Península) vía la
API de e·sios/REE (https://api.esios.ree.es), autenticada con REE_ESIOS_TOKEN.

A diferencia de ENTSO-E, e·sios no tiene un único indicador "generación por
tecnología": son series sueltas por indicador, y muchas familias son legacy y ya no se
actualizan. El conjunto usado aquí se eligió el 2026-08-27 tras un reconocimiento en
vivo documentado en docs/findings.md ("Decisiones de fuente pendientes... España"):

1. Demanda: indicador 1293 ("Demanda real"), geo_id 8741 (Península).
2. Generación por tecnología: indicadores 546/547/548/549/550/551/1294/1295/1296/1297
   (hidráulica/carbón/fuel-gas/nuclear/ciclo combinado/eólica/solar térmica/solar
   fotovoltaica/térmica renovable/cogeneración y resto), más 553 (intercambios
   internacionales) y 554 (enlace Baleares) — estas dos últimas no son generación en
   sí, pero son necesarias para que el balance cierre (verificado: la suma de las 12
   categorías reconstruye la demanda real con <1% de desviación en la comprobación
   puntual hecha). Se marcan con category_type='exchange' en vez de 'generation' para
   poder excluirlas de un análisis de mix tecnológico puro.
3. Familias descartadas (verificadas en vivo, NO usar si se retoma este pipeline):
   el indicador 10195 ("Generación medida total tipo producción") y la familia
   10035-10047/10062 ("Generación medida X") devuelven 0 filas — parecen
   discontinuados. El indicador 552 ("Solar", agregado legacy) NO coincide con la suma
   de 1294+1295 (552 muy por debajo) — no es la fuente correcta. La familia "nacional"
   (2037-2051, geo_id=3 España) da valores casi idénticos a la familia Península
   (546-555) en la comprobación puntual — parecen alias/duplicados, no cobertura
   adicional real.

Notas de formato de la API, verificadas en vivo el 2026-08-27 (distintas de ENTSO-E):
- JSON, no XML. Cada punto ya trae su propio timestamp completo (datetime_utc) — no
  hay que reconstruir instantes a partir de un `resolution` + `position` como en
  ENTSO-E, así que no se guarda una columna `resolution_minutes` (no aplica al modelo
  de datos de esta API).
- Resolución de 5 minutos constante, verificada tanto en 2023-01-01 como en 2026 — a
  diferencia de ENTSO-E, no hay transición 60→15 min conocida en esta serie.
- Un año completo (~105.000 filas) da timeout (>60s); un trimestre (~26.000 filas) se
  resuelve en ~15s. Por eso el backfill trocea por trimestre, no por año como la
  demanda ENTSO-E.
- Cada indicador puede devolver puntos de otros geo_id además de Península en teoría
  (la API los agrupa por indicador, no por área) — se filtra explícitamente a
  geo_id=8741 y se registra WARN si aparece cualquier otro, en vez de asumir que
  siempre va a ser el único.
"""

from __future__ import annotations

import time
from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo

import duckdb
import pandas as pd
import requests

from etl.common import catalog
from etl.common.catalog import ColumnDoc
from etl.common.config import require_env

API_URL = "https://api.esios.ree.es/indicators"
DB_NAME = "spain"
TZ_MADRID = ZoneInfo("Europe/Madrid")
PENINSULA_GEO_ID = 8741

DEMAND_INDICATOR = 1293

# category_key -> (indicator_id, nombre legible, category_type, banda_plausibilidad_abs_mw)
# La banda es None salvo evidencia empírica propia (ver docs/findings.md): solo
# "enlace_baleares" la tiene, porque el backfill 2023-2026 mostró un percentil 99.9%
# de 0 MW pero 37 puntos (de 384k) disparados hasta 5.792 MW — un enlace físico de
# este tamaño no tiene esa capacidad. La banda (1.000 MW) sale de la distribución
# empírica observada, no de una cifra de capacidad nominal verificada en vivo.
GENERATION_CATEGORIES = {
    "hidraulica": (546, "Hidráulica", "generation", None),
    "carbon": (547, "Carbón", "generation", None),
    "fuel_gas": (548, "Fuel-gas", "generation", None),
    "nuclear": (549, "Nuclear", "generation", None),
    "ciclo_combinado": (550, "Ciclo combinado", "generation", None),
    "eolica": (551, "Eólica", "generation", None),
    "solar_termica": (1294, "Solar térmica", "generation", None),
    "solar_fotovoltaica": (1295, "Solar fotovoltaica", "generation", None),
    "termica_renovable": (1296, "Térmica renovable", "generation", None),
    "cogeneracion_resto": (1297, "Cogeneración y resto", "generation", None),
    "intercambios_internacionales": (553, "Intercambios internacionales", "exchange", None),
    "enlace_baleares": (554, "Enlace Baleares", "exchange", 1000.0),
}

_session = requests.Session()


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------

def fetch_indicator(indicator_id: int, start_iso: str, end_iso: str, timeout: int = 60, retries: int = 3) -> list[dict]:
    """start_iso/end_iso: 'AAAA-MM-DDTHH:MM:SS' (hora local de Madrid, sin offset —
    formato que espera la API de e·sios, verificado en vivo)."""
    headers = {
        "Accept": "application/json; application/vnd.esios-api-v2+json",
        "Content-Type": "application/json",
        "x-api-key": require_env("REE_ESIOS_TOKEN"),
    }
    params = {"start_date": start_iso, "end_date": end_iso}
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            resp = _session.get(f"{API_URL}/{indicator_id}", headers=headers, params=params, timeout=timeout)
            resp.raise_for_status()
            return resp.json()["indicator"]["values"]
        except requests.RequestException as exc:
            last_exc = exc
            time.sleep(2 * attempt)
    raise last_exc  # type: ignore[misc]


def _parse_values(values: list[dict]) -> tuple[list[dict], int]:
    """Filtra a geo_id=8741 (Península) y parsea datetime_utc. Devuelve (filas, n_otro_geo)."""
    rows = []
    n_other_geo = 0
    for v in values:
        if v.get("geo_id") != PENINSULA_GEO_ID:
            n_other_geo += 1
            continue
        interval_start_utc = datetime.strptime(v["datetime_utc"], "%Y-%m-%dT%H:%M:%SZ")
        rows.append({"interval_start_utc": interval_start_utc, "value": v["value"]})
    return rows, n_other_geo


# ---------------------------------------------------------------------------
# Diccionario de datos
# ---------------------------------------------------------------------------

def document_demand(con: duckdb.DuckDBPyConnection) -> None:
    cols = {
        "interval_start_utc": ColumnDoc(
            description="Instante UTC de inicio del intervalo de 5 minutos.",
            source="Campo datetime_utc de la API de e·sios (indicador 1293).",
            dtype="TIMESTAMP", kind="identifier",
        ),
        "interval_start_local": ColumnDoc(
            description="Mismo instante en hora local de España peninsular (Europe/Madrid).",
            source="Calculado por el pipeline a partir de interval_start_utc.",
            dtype="TIMESTAMP", kind="timestamp",
        ),
        "demand_mw": ColumnDoc(
            description="Demanda real de España peninsular, en MW.",
            source="Campo value del indicador 1293 ('Demanda real') de e·sios, geo_id=8741 (Península).",
            dtype="DOUBLE", kind="continuous",
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
    catalog.refresh(con, "esios_demand", cols)


def document_generation(con: duckdb.DuckDBPyConnection) -> None:
    cols = {
        "interval_start_utc": ColumnDoc(
            description="Instante UTC de inicio del intervalo de 5 minutos.",
            source="Campo datetime_utc de la API de e·sios.",
            dtype="TIMESTAMP", kind="identifier",
        ),
        "interval_start_local": ColumnDoc(
            description="Mismo instante en hora local de España peninsular (Europe/Madrid).",
            source="Calculado por el pipeline a partir de interval_start_utc.",
            dtype="TIMESTAMP", kind="timestamp",
        ),
        "category_key": ColumnDoc(
            description="Categoría: una de GENERATION_CATEGORIES (hidraulica, carbon, fuel_gas, nuclear, ciclo_combinado, eolica, solar_termica, solar_fotovoltaica, termica_renovable, cogeneracion_resto, intercambios_internacionales, enlace_baleares).",
            source="Constante de configuración del pipeline (GENERATION_CATEGORIES), no viene en la respuesta de e·sios.",
            dtype="VARCHAR", kind="categorical",
        ),
        "category_name": ColumnDoc(
            description="Nombre legible de la categoría.",
            source="Constante de configuración del pipeline.",
            dtype="VARCHAR", kind="categorical",
        ),
        "category_type": ColumnDoc(
            description=(
                "'generation' (una tecnología de generación real) o 'exchange' (intercambios "
                "internacionales o enlace con Baleares — no es generación en sí, pero es necesario "
                "sumarlo junto a las tecnologías para que el balance frente a la demanda real "
                "cierre; ver docstring del módulo)."
            ),
            source="Constante de configuración del pipeline.",
            dtype="VARCHAR", kind="categorical",
        ),
        "value_mw": ColumnDoc(
            description=(
                "Valor de la categoría en MW. Puede ser negativo: en 'hidraulica' cuando el consumo "
                "por bombeo supera a la generación en ese instante (REE publica un único valor neto, "
                "a diferencia de ENTSO-E que separa generación/consumo de bombeo en dos series — ver "
                "etl/sources/entsoe_load_generation.py); en 'intercambios_internacionales' cuando "
                "España es exportadora neta en ese instante."
            ),
            source="Campo value del indicador correspondiente de e·sios, geo_id=8741 (Península).",
            dtype="DOUBLE", kind="continuous",
        ),
        "indicator_id": ColumnDoc(
            description="Identificador numérico del indicador de e·sios de origen (p.ej. 549 para nuclear) — para trazabilidad si e·sios cambia de convención.",
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
    catalog.refresh(con, "esios_generation_by_type", cols)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

def ensure_schema(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS esios_demand (
            interval_start_utc    TIMESTAMP NOT NULL,
            interval_start_local  TIMESTAMP NOT NULL,
            demand_mw             DOUBLE NOT NULL,
            source_chunk          VARCHAR NOT NULL,
            ingested_at           TIMESTAMP NOT NULL,
            PRIMARY KEY (interval_start_utc)
        )
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS esios_generation_by_type (
            interval_start_utc    TIMESTAMP NOT NULL,
            interval_start_local  TIMESTAMP NOT NULL,
            category_key          VARCHAR NOT NULL,
            category_name         VARCHAR NOT NULL,
            category_type         VARCHAR NOT NULL,
            value_mw              DOUBLE NOT NULL,
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


def run_demand_for_range(con: duckdb.DuckDBPyConnection, start_iso: str, end_iso: str, logger) -> dict:
    started_at = datetime.utcnow()
    ensure_schema(con)
    source_chunk = f"{start_iso}-{end_iso}"
    source = "esios_demand"

    try:
        raw_values = fetch_indicator(DEMAND_INDICATOR, start_iso, end_iso)
        rows, n_other_geo = _parse_values(raw_values)
    except Exception as exc:
        _log_run(con, source, "esios_demand", 0, "ERROR", str(exc), started_at)
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
                (r["interval_start_utc"], r["interval_start_utc"].replace(tzinfo=timezone.utc).astimezone(TZ_MADRID).replace(tzinfo=None),
                 r["value"], source_chunk, now)
                for r in rows
            ],
            columns=["interval_start_utc", "interval_start_local", "demand_mw", "source_chunk", "ingested_at"],
        )
        min_ts, max_ts = df["interval_start_utc"].min(), df["interval_start_utc"].max()
        con.execute("BEGIN TRANSACTION")
        try:
            con.execute("DELETE FROM esios_demand WHERE interval_start_utc >= ? AND interval_start_utc <= ?", [min_ts, max_ts])
            con.append("esios_demand", df)
            con.execute("COMMIT")
        except Exception:
            con.execute("ROLLBACK")
            raise
        rows_loaded = len(df)

    message = "; ".join(messages) if messages else "sin incidencias"
    _log_run(con, source, "esios_demand", rows_loaded, status, message, started_at)
    log_fn = logger.info if status == "OK" else logger.warning
    log_fn("%s %s: %s filas cargadas — %s", source, source_chunk, rows_loaded, message)
    return {"status": status, "rows_loaded": rows_loaded, "message": message}


def _run_category_for_range(con: duckdb.DuckDBPyConnection, category_key: str, start_iso: str, end_iso: str, logger) -> dict:
    started_at = datetime.utcnow()
    indicator_id, category_name, category_type, sanity_max_abs_mw = GENERATION_CATEGORIES[category_key]
    source_chunk = f"{start_iso}-{end_iso}"
    source = f"esios_generation:{category_key}"

    try:
        raw_values = fetch_indicator(indicator_id, start_iso, end_iso)
        rows, n_other_geo = _parse_values(raw_values)
    except Exception as exc:
        _log_run(con, source, "esios_generation_by_type", 0, "ERROR", str(exc), started_at)
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
    if sanity_max_abs_mw is not None:
        n_out_of_band = sum(1 for r in rows if abs(r["value"]) > sanity_max_abs_mw)
        if n_out_of_band:
            status = "WARN"
            messages.append(f"{n_out_of_band} puntos por encima de la banda de plausibilidad (±{sanity_max_abs_mw:.0f} MW)")

    now = datetime.utcnow()
    rows_loaded = 0
    if rows:
        df = pd.DataFrame(
            [
                (
                    r["interval_start_utc"], r["interval_start_utc"].replace(tzinfo=timezone.utc).astimezone(TZ_MADRID).replace(tzinfo=None),
                    category_key, category_name, category_type, r["value"], indicator_id, source_chunk, now,
                )
                for r in rows
            ],
            columns=["interval_start_utc", "interval_start_local", "category_key", "category_name",
                     "category_type", "value_mw", "indicator_id", "source_chunk", "ingested_at"],
        )
        min_ts, max_ts = df["interval_start_utc"].min(), df["interval_start_utc"].max()
        con.execute("BEGIN TRANSACTION")
        try:
            con.execute(
                "DELETE FROM esios_generation_by_type WHERE category_key = ? AND interval_start_utc >= ? AND interval_start_utc <= ?",
                [category_key, min_ts, max_ts],
            )
            con.append("esios_generation_by_type", df)
            con.execute("COMMIT")
        except Exception:
            con.execute("ROLLBACK")
            raise
        rows_loaded = len(df)

    message = "; ".join(messages) if messages else "sin incidencias"
    _log_run(con, source, "esios_generation_by_type", rows_loaded, status, message, started_at)
    log_fn = logger.info if status == "OK" else logger.warning
    log_fn("%s %s: %s filas cargadas — %s", source, source_chunk, rows_loaded, message)
    return {"status": status, "rows_loaded": rows_loaded, "message": message}


def run_generation_for_range(con: duckdb.DuckDBPyConnection, start_iso: str, end_iso: str, logger) -> dict:
    """Ejecuta las 12 categorías (10 tecnologías + 2 de intercambio) para el mismo rango."""
    ensure_schema(con)
    results = {}
    for category_key in GENERATION_CATEGORIES:
        results[category_key] = _run_category_for_range(con, category_key, start_iso, end_iso, logger)
        time.sleep(0.3)

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

    log = get_logger("esios_load_generation.manual", DB_NAME)
    conn = connect(DB_NAME)
    print(run_demand_for_range(conn, "2023-01-01T00:00:00", "2023-01-08T00:00:00", log))
    print(run_generation_for_range(conn, "2023-01-01T00:00:00", "2023-01-08T00:00:00", log))
    document_demand(conn)
    document_generation(conn)
    document_ingestion_log(conn)
    conn.close()
