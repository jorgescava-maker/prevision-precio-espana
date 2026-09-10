"""
Ingesta de NTC estimada (Net Transfer Capacity, documentType A61, contrato diario
A01 — "Estimated Net Transfer Capacity", asignación explícita día-adelantada) vía
ENTSO-E Transparency Platform. Complementa el flujo físico real ya cargado en
etl/sources/entsoe_cross_border_flows.py: NTC es el TECHO que el operador considera
seguro asignar; el flujo físico es lo que realmente circuló (normalmente por debajo).

Requiere ENTSOE_API_TOKEN en .env (mismo token que el resto de pipelines ENTSO-E).

Decisión de alcance (2026-08-27, con el usuario): SOLO ES-FR y ES-PT, no las cuatro
fronteras de etl/sources/entsoe_cross_border_flows.py. Verificado en vivo:

1. DE-FR no tiene NTC explícita en absoluto — ENTSO-E devuelve "No matching data
   found" para ese par. Francia-Alemania asigna su capacidad de forma implícita
   (acoplamiento único de mercado, sin publicar una cifra de NTC separada).
2. DE-NL sí devuelve datos, pero son ENGAÑOSOS si se presentan como "la capacidad de
   la interconexión": los valores observados (0-1468 MW en 2023 y 2026) son un orden
   de magnitud menores que la capacidad física real ya vista en el flujo físico (hasta
   6572 MW, ver docs/findings.md). La razón: Alemania-Países Bajos forma parte de la
   región Core Flow-Based Market Coupling desde 2022 — casi toda la capacidad se
   asigna mediante parámetros basados en flujo (una matriz de restricciones,
   documentType B11, no una cifra NTC simple); esta serie A61 solo captura un
   remanente explícito residual, no la capacidad total. Cargarlo sin este contexto
   induciría a error a cualquier estudio que lo usara como "capacidad de DE-NL".
3. ES-FR y ES-PT sí tienen NTC explícita representativa (mismo orden de magnitud que
   la capacidad física observada): ES-FR 250-3838 MW, ES-PT 1999-4950 MW (verificado
   en 2023 completo) — la región ibérica todavía usa asignación NTC coordinada, no
   flow-based.

Si en el futuro se quiere capacidad para DE-FR/DE-NL, la vía correcta es investigar
Flow-Based Parameters (documentType B11) como pipeline separado, no extender este.

Ampliación 2026-08-27 (verificado en vivo, ver docs/findings.md): al añadir fronteras
alemanas nuevas en etl/sources/entsoe_cross_border_flows.py (BE, DK1, DK2, PL, AT, CH),
se comprobó A61 para las 6. Mismo patrón que arriba:

- DE-BE y DE-PL: "No matching data found" — Bélgica y Polonia están en la región Core
  Flow-Based Market Coupling (como Alemania/Países Bajos), sin NTC explícita por el
  mismo motivo que DE-NL (nota 2 de arriba). NO añadidas a NTC_BORDERS.
- DE-AT: "No matching data found" — igual que BE/PL, Austria también forma parte de
  Core FBMC. NO añadida.
- DE-DK2 (Kriegers Flak, enlace híbrido con el parque eólico marino homónimo): "No
  matching data found" — un enlace híbrido combina interconexión y evacuación eólica
  en un único activo con su propio mecanismo de asignación, no NTC A61 estándar. NO
  añadida.
- DE-DK1 (Jutlandia, AC clásica): SÍ tiene NTC explícita real y representativa
  (250-3500 MW en 2023 completo, resolución PT15M) — Dinamarca no está en Core FBMC.
  AÑADIDA a NTC_BORDERS.
- DE-CH: SÍ tiene NTC explícita real y representativa (hasta 4000 MW en 2023
  completo) — Suiza no es UE, sigue con subastas explícitas de capacidad coordinadas
  vía JAO en sus fronteras. AÑADIDA a NTC_BORDERS.

Reutiliza la configuración de áreas/EIC/tz de etl/sources/entsoe_cross_border_flows.BORDERS
(mismas fronteras, mismo país -> tz) para no duplicar esas constantes.

Hechos verificados en vivo el 2026-08-27:

1. Al igual que el flujo físico, la NTC se pide y publica como dos series
   independientes por sentido (in_Domain=destino, out_Domain=origen; valor =
   capacidad origen->destino). A diferencia del flujo físico, SÍ puede llegar a 0
   (capacidad nula, p.ej. por mantenimiento) pero nunca negativa.
2. La NTC estimada se mantiene en resolución horaria (PT60M) de forma constante desde
   2023 hasta hoy — verificado explícitamente que NO sigue la reforma paneuropea de
   cuartos de hora del 2025-10-01 (a diferencia del precio/demanda/generación/flujo
   físico). Tiene sentido: es una previsión operativa día-adelantada, no una medición
   de mercado sujeta al Market Time Unit del acoplamiento único.
3. La NTC de "mañana" ya está publicada en el momento de la carga diaria (es una
   previsión día-adelantada, como el precio) — la actualización diaria pide una
   ventana con lookahead, a diferencia de etl/sources/entsoe_cross_border_flows.py
   (flujo físico real, sin lookahead).
4. Reutiliza el parseo común de etl/common/entsoe_xml.py sin modificarlo.
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
from etl.sources.entsoe_cross_border_flows import BORDERS as FLOW_BORDERS

API_URL = "https://web-api.tp.entsoe.eu/api"
CONTRACT_TYPE = "A01"  # Daily — asignación explícita día-adelantada

# Subconjunto deliberado de etl.sources.entsoe_cross_border_flows.BORDERS — ver nota
# de alcance en el docstring del módulo. ntc_sanity_max_mw generoso sobre el máximo
# observado en vivo en 2023 completo por frontera.
NTC_BORDERS = {
    "es_fr": {**FLOW_BORDERS["es_fr"], "ntc_sanity_max_mw": 6000.0},  # máx. observado 2023: 3838 MW
    "es_pt": {**FLOW_BORDERS["es_pt"], "ntc_sanity_max_mw": 6500.0},  # máx. observado 2023: 4950 MW
    "de_dk1": {**FLOW_BORDERS["de_dk1"], "ntc_sanity_max_mw": 3500.0},  # máx. observado 2023: 2500 MW
    "de_ch": {**FLOW_BORDERS["de_ch"], "ntc_sanity_max_mw": 5500.0},  # máx. observado 2023: 4000 MW
    # Fronteras menores añadidas 2026-08-28 (ver docs/findings.md #68): FR-CH/FR-IT/
    # AT-CH/AT-IT sí tienen NTC (A61) porque Suiza e Italia están fuera de la región
    # Core Flow-Based — verificado en vivo, a diferencia de FR-BE/PL-CZ/PL-SE4 (dentro
    # de Core o sin A61 por otro motivo, ver nota de alcance). Bandas provisionales.
    "fr_ch": {**FLOW_BORDERS["fr_ch"], "ntc_sanity_max_mw": 4500.0},
    "fr_it": {**FLOW_BORDERS["fr_it"], "ntc_sanity_max_mw": 5500.0},
    "at_ch": {**FLOW_BORDERS["at_ch"], "ntc_sanity_max_mw": 2000.0},
    "at_it": {**FLOW_BORDERS["at_it"], "ntc_sanity_max_mw": 800.0},
}

_session = requests.Session()


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------

def fetch_raw(in_eic: str, out_eic: str, period_start: str, period_end: str, timeout: int = 60, retries: int = 3) -> str:
    """in_eic = área destino, out_eic = área origen — el valor devuelto es la NTC
    origen->destino (mismo convenio verificado que el flujo físico, ver nota 1)."""
    params = {
        "securityToken": require_env("ENTSOE_API_TOKEN"),
        "documentType": "A61",
        "contract_MarketAgreement.Type": CONTRACT_TYPE,
        "in_Domain": in_eic,
        "out_Domain": out_eic,
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
# Diccionario de datos
# ---------------------------------------------------------------------------

NTC_COLUMNS = {
    "interval_start_utc": ColumnDoc(
        description="Instante UTC de inicio del intervalo de NTC estimada.",
        source="timeInterval/start de cada <Period> + offset de (position-1) × resolución, del XML de ENTSO-E (documentType A61, contrato A01).",
        dtype="TIMESTAMP", kind="identifier",
    ),
    "interval_start_local": ColumnDoc(
        description="Mismo instante en hora local del área exportadora (from_area). Ver la misma advertencia sobre ES-PT (husos WET/WEST vs CET/CEST) documentada en cross_border_flows.",
        source="Calculado por el pipeline a partir de interval_start_utc y el tz del área exportadora (from_area).",
        dtype="TIMESTAMP", kind="timestamp",
    ),
    "resolution_minutes": ColumnDoc(
        description="Duración real del intervalo. Verificado que se mantiene en 60 min de forma constante 2023->2026 para ES-FR y ES-PT — NO sigue la reforma paneuropea de cuartos de hora de 2025-10-01 (a diferencia de precio/demanda/generación/flujo físico), por ser una previsión operativa, no una medición de mercado.",
        source="Elemento <resolution> del <Period> correspondiente del XML de ENTSO-E.",
        dtype="SMALLINT", kind="categorical",
    ),
    "border_key": ColumnDoc(
        description="Identificador de la frontera, códigos de país en orden alfabético ('ES-FR').",
        source="Constante de configuración (NTC_BORDERS en el pipeline).",
        dtype="VARCHAR", kind="categorical",
    ),
    "from_area": ColumnDoc(
        description="Área de origen en este sentido de la NTC (hacia donde se permitiría exportar hasta ntc_mw).",
        source="Constante de configuración — corresponde al out_Domain de la petición a ENTSO-E.",
        dtype="VARCHAR", kind="categorical",
    ),
    "to_area": ColumnDoc(
        description="Área de destino en este sentido de la NTC.",
        source="Constante de configuración — corresponde al in_Domain de la petición a ENTSO-E.",
        dtype="VARCHAR", kind="categorical",
    ),
    "ntc_mw": ColumnDoc(
        description="Capacidad neta de transferencia ESTIMADA (día-adelantada) en el sentido from_area -> to_area, en MW. Puede ser 0 (p.ej. mantenimiento) pero nunca negativa. Es un TECHO operativo, no una medición — comparar con flow_mw de cross_border_flows para ver cuánto de la capacidad asignada se usó realmente.",
        source="Elemento <quantity> de cada <Point>, con forward-fill de posiciones omitidas.",
        dtype="DOUBLE", kind="continuous",
    ),
    "source_chunk": ColumnDoc(
        description="Identificador del bloque de petición de origen (rango pedido a la API), para trazabilidad.",
        source="Parámetro de la petición a la API de ENTSO-E.",
        dtype="VARCHAR", kind="identifier",
    ),
    "ingested_at": ColumnDoc(
        description="Marca temporal UTC del momento en que esta fila se cargó (o recargó) por última vez en la base de datos.",
        source="datetime.utcnow() en el momento de la inserción, dentro de la función load().",
        dtype="TIMESTAMP", kind="timestamp",
    ),
}


def document(con: duckdb.DuckDBPyConnection) -> None:
    catalog.refresh(con, "ntc_estimated", NTC_COLUMNS)


# ---------------------------------------------------------------------------
# Schema + load
# ---------------------------------------------------------------------------

def ensure_schema(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS ntc_estimated (
            interval_start_utc    TIMESTAMP NOT NULL,
            interval_start_local  TIMESTAMP NOT NULL,
            resolution_minutes    SMALLINT NOT NULL,
            border_key            VARCHAR NOT NULL,
            from_area              VARCHAR NOT NULL,
            to_area                VARCHAR NOT NULL,
            ntc_mw                  DOUBLE NOT NULL,
            source_chunk             VARCHAR NOT NULL,
            ingested_at               TIMESTAMP NOT NULL,
            PRIMARY KEY (interval_start_utc, from_area, to_area)
        )
        """
    )


# ---------------------------------------------------------------------------
# Orquestación de un rango + registro de auditoría
# ---------------------------------------------------------------------------

def _log_run(con, source, rows_loaded, status, message, started_at) -> None:
    con.execute(
        """
        INSERT INTO ingestion_log
            (source, target_table, delivery_date, rows_expected, rows_loaded, status, message, started_at, finished_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [source, "ntc_estimated", date.today(), None, rows_loaded, status, message, started_at, datetime.utcnow()],
    )


def run_direction_for_range(
    con: duckdb.DuckDBPyConnection, border_key_cfg: str, from_area: str, to_area: str,
    period_start: str, period_end: str, logger,
) -> dict:
    started_at = datetime.utcnow()
    ensure_schema(con)
    border = NTC_BORDERS[border_key_cfg]
    from_cfg = border["areas"][from_area]
    to_cfg = border["areas"][to_area]
    border_key = border["border_key"]
    source_chunk = f"{period_start}-{period_end}"
    source = f"entsoe_ntc:{border_key}:{from_area}->{to_area}"

    try:
        raw = fetch_raw(to_cfg["eic"], from_cfg["eic"], period_start, period_end)
        grouped = entsoe_xml.parse_periods(raw)
        rows, parse_stats = grouped.get("default", ([], {"n_exact_duplicates": 0, "n_revised": 0}))
    except Exception as exc:
        _log_run(con, source, 0, "ERROR", str(exc), started_at)
        logger.error("%s %s: %s", source, source_chunk, exc)
        return {"status": "ERROR", "message": str(exc)}

    messages: list[str] = []
    status = "OK"
    if parse_stats["n_revised"]:
        status = "WARN"
        messages.append(f"{parse_stats['n_revised']} intervalos revisados (valor distinto en dos TimeSeries)")
    n_negative = sum(1 for r in rows if r["value"] < 0)
    if n_negative:
        status = "WARN"
        messages.append(f"{n_negative} intervalos con NTC NEGATIVA (inesperado)")
    ntc_sanity_max_mw = border["ntc_sanity_max_mw"]
    n_out_of_band = sum(1 for r in rows if r["value"] > ntc_sanity_max_mw)
    if n_out_of_band:
        status = "WARN"
        messages.append(f"{n_out_of_band} intervalos por encima de la banda de plausibilidad ({ntc_sanity_max_mw:.0f} MW)")
    if not rows:
        status = "WARN"
        messages.append("sin filas en este bloque")

    now = datetime.utcnow()
    rows_loaded = 0
    if rows:
        tz = from_cfg["tz"]
        df = pd.DataFrame(
            [
                (
                    r["interval_start_utc"],
                    r["interval_start_utc"].replace(tzinfo=timezone.utc).astimezone(tz).replace(tzinfo=None),
                    r["resolution_minutes"], border_key, from_area, to_area, r["value"], source_chunk, now,
                )
                for r in rows
            ],
            columns=["interval_start_utc", "interval_start_local", "resolution_minutes", "border_key",
                     "from_area", "to_area", "ntc_mw", "source_chunk", "ingested_at"],
        )
        min_ts, max_ts = df["interval_start_utc"].min(), df["interval_start_utc"].max()
        con.execute("BEGIN TRANSACTION")
        try:
            con.execute(
                "DELETE FROM ntc_estimated WHERE from_area = ? AND to_area = ? "
                "AND interval_start_utc >= ? AND interval_start_utc <= ?",
                [from_area, to_area, min_ts, max_ts],
            )
            con.append("ntc_estimated", df)
            con.execute("COMMIT")
        except Exception:
            con.execute("ROLLBACK")
            raise
        rows_loaded = len(df)

    message = "; ".join(messages) if messages else "sin incidencias"
    _log_run(con, source, rows_loaded, status, message, started_at)
    log_fn = logger.info if status == "OK" else logger.warning
    log_fn("%s %s: %s filas cargadas — %s", source, source_chunk, rows_loaded, message)
    return {"status": status, "from_area": from_area, "to_area": to_area, "rows_loaded": rows_loaded, "message": message}


def run_border_for_range(con: duckdb.DuckDBPyConnection, border_key_cfg: str, period_start: str, period_end: str, logger) -> list[dict]:
    """Ejecuta ambos sentidos de una frontera."""
    a, b = list(NTC_BORDERS[border_key_cfg]["areas"])
    return [
        run_direction_for_range(con, border_key_cfg, a, b, period_start, period_end, logger),
        run_direction_for_range(con, border_key_cfg, b, a, period_start, period_end, logger),
    ]


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(__file__).rsplit("etl", 1)[0])
    from etl.common.db import connect, document_ingestion_log
    from etl.common.logging_config import get_logger

    border = sys.argv[1] if len(sys.argv) > 1 else "es_fr"
    log = get_logger("entsoe_ntc.manual", NTC_BORDERS[border]["db"])
    conn = connect(NTC_BORDERS[border]["db"])
    results = run_border_for_range(conn, border, "202301010000", "202301080000", log)
    print(results)
    document(conn)
    document_ingestion_log(conn)
    conn.close()
