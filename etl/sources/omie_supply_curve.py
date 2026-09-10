"""
Ingesta de la curva agregada de oferta del mercado diario (OMIE, fichero
curva_pbc) — dato requerido para investigar la petición del usuario del
2026-09-01: capturar cómo pujan los participantes (no solo qué tecnología
genera) en las horas de precio extremo, más allá de lo que explica la
mezcla tecnológica marginal (`tecnologia_marginal_final`, ver merit_order) —
verificado en la misma sesión: el sesgo del modelo en el régimen de escasez
extrema (bottom-2% `reserve_margin_mw`) es prácticamente el MISMO (~-9,5 a
-10 EUR/MWh) sea cual sea la tecnología marginal, lo que apunta a un markup
de escasez aplicado de forma más o menos uniforme — solo visible en la curva
de oferta real, no en la generación resultante.

Hechos verificados en vivo el 2026-09-01:
1. Fichero público, un `.1` por día natural, sin retraso de confidencialidad
   — descarga directa por
   `file-download?parents[0]=curva_pbc&filename=curva_pbc_YYYYMMDD.1`, mismo
   patrón que `omie_intraday.py`/`omie_spot.py`. Cabecera de 3 líneas
   (título, blanco, nombres de columna) antes de los datos. Codificación
   ISO-8859-1. La fecha de emisión observada es el día SIGUIENTE al de
   entrega (p.ej. curva del 30/08 emitida el 31/08) — el fichero solo existe
   una vez cerrada la sesión, con el flag de casación ya resuelto.
2. Cada fila es UN BLOQUE de oferta (no un resumen por periodo): columnas
   Periodo (formato "H{hora}Q{cuarto}", p.ej. "H1Q1".."H24Q4" — siempre con
   'Q' desde el cutover a cuartohorario, único rango cubierto por este
   módulo), Fecha, Pais, Unidad (VACÍA — no hay identidad de central a este
   nivel; eso solo está en los ficheros mensuales `curva_pbc_uof`, con
   retraso de confidencialidad, fuera de alcance por ahora), Tipo Oferta
   (C=compra/demanda, V=venta/generación — este módulo solo usa V), Potencia
   y Precio (formato numérico europeo, punto de millar + coma decimal),
   Ofertada(O)/Casada(C) (si ese bloque concreto casó), Tipología de Oferta
   (S=simple/C=compleja).
3. Volumen real: ~500.000 filas/día (~340.000 de venta) — guardar cada
   bloque crudo un año entero sería una tabla enorme y de más granularidad
   de la que hace falta (la FORMA de la curva, no cada bloque individual).
   Este módulo NO guarda las filas crudas: agrega la curva de venta (V) a un
   resumen por periodo (96/día) en el momento de la ingesta — precio de
   casación derivado de la propia curva (el volumen casado se calcula
   sumando los bloques V con flag Casada, sin depender de ninguna otra
   tabla) y el precio necesario para casar 250/500/1000/2000 MW EXTRA por
   encima del volumen realmente casado (proxy directo de la pendiente de la
   curva justo en el margen — cuán agresivamente puja la siguiente unidad).
4. Esta serie es EX-POST (ver punto 1: solo existe una vez cerrada la
   sesión) — NO es D-1-segura tal cual, no debe usarse directamente como
   feature de un modelo de predicción en vivo. Para eso haría falta un lag
   (p.ej. la pendiente de ayer o de hace una semana, mismo patrón que
   `_add_climatology`), todavía no construido — de momento es solo para el
   estudio explicativo del sesgo en precio extremo.
5. **Limitación verificada, no un bug**: reconstruir el precio de casación
   ordenando la curva 'O' por precio y leyendo el precio en el volumen
   casado (`matched_volume_mw`) da un valor CERCANO pero no idéntico al
   precio oficial de `omie_spot_prices` — contraste en un día de muestra
   (2026-08-30): diferencia media 13 EUR/MWh, máxima ~94 EUR/MWh en algún
   periodo. Motivo más probable: OMIE/Euphemia empareja con ofertas
   COMPLEJAS (bloques indivisibles, gradientes de carga, condiciones de
   parada — columna 'Tipología de Oferta' distinta de 'S' en ~4% de los
   bloques de la muestra), que pueden romper el orden de mérito simple que
   asume esta reconstrucción. `clearing_price_from_curve_eur_mwh` debe
   tratarse como una aproximación de la FORMA de la curva en esa zona, no
   como sustituto exacto del precio oficial — para el precio real, usar
   siempre `omie_spot_prices`.

Uso:
    .venv\\Scripts\\python.exe -m etl.sources.omie_supply_curve
"""

from __future__ import annotations

import io
import time
from datetime import date, datetime

import duckdb
import numpy as np
import pandas as pd
import requests

from etl.common import catalog
from etl.common.catalog import ColumnDoc
from etl.sources.omie_spot import expected_period_count, period_start_utc, resolution_minutes_for

BASE_URL = "https://www.omie.es/es/file-download"

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
PRICE_SANITY_MIN, PRICE_SANITY_MAX = -500.0, 4000.0
CURVE_START_DATE = date(2025, 10, 1)  # alcance pedido por el usuario: desde el cutover a cuartohoraria
EXTRA_MW_LEVELS = (250, 500, 1000, 2000)


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------

def fetch_raw(delivery_date: date, timeout: int = 60, retries: int = 3) -> str | None:
    filename = f"curva_pbc_{delivery_date.strftime('%Y%m%d')}.1"
    params = {"parents[0]": "curva_pbc", "filename": filename}
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            resp = _session.get(BASE_URL, params=params, timeout=timeout)
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            text = resp.content.decode("iso-8859-1")
            return text if text.strip().startswith("OMIE") else None
        except requests.RequestException as exc:
            last_exc = exc
            time.sleep(1.5 * attempt)
    raise last_exc  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Parse + agregación de la curva por periodo
# ---------------------------------------------------------------------------

def _parse_period_index(periodo: str) -> int | None:
    if not isinstance(periodo, str) or not periodo.startswith("H") or "Q" not in periodo:
        return None
    h_part, q_part = periodo[1:].split("Q", 1)
    try:
        h, q = int(h_part), int(q_part)
    except ValueError:
        return None
    return (h - 1) * 4 + q


def summarize(raw_text: str, delivery_date: date) -> list[dict]:
    df = pd.read_csv(
        io.StringIO(raw_text),
        sep=";",
        skiprows=2,
        header=0,
        decimal=",",
        thousands=".",
        engine="c",
        on_bad_lines="skip",
    )
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={
        "Periodo": "periodo", "Fecha": "fecha", "Tipo Oferta": "tipo_oferta",
        "Potencia Compra/Venta": "potencia", "Precio Compra/Venta": "precio",
        "Ofertada (O)/Casada (C)": "oc",
    })
    df = df.dropna(subset=["periodo", "tipo_oferta", "potencia", "precio"])

    fechas = sorted({f for f in pd.to_datetime(df["fecha"], format="%d/%m/%Y", errors="coerce").dt.date if f is not None})
    if fechas != [delivery_date]:
        raise ValueError(f"fecha(s) en fichero {fechas} no coincide con la solicitada ({delivery_date})")

    df["period_index"] = df["periodo"].map(_parse_period_index)
    df = df.dropna(subset=["period_index"])
    df["period_index"] = df["period_index"].astype(int)

    sell = df[df["tipo_oferta"] == "V"].copy()

    # El fichero trae DOS trazas superpuestas de la misma curva por periodo, no una
    # partición: 'O' es la curva COMPLETA de oferta (todo el rango de precio, se case
    # o no) y 'C' es una traza REDUNDANTE que solo repite los bloques que sí casaron
    # (mismo precio mínimo -500, precio máximo = precio de casación). Verificado en
    # vivo el 2026-09-01: sumar O+C como "oferta total" duplica volumen y busca el
    # volumen casado dentro de la curva equivocada (da un precio de casación absurdo,
    # p.ej. -5 EUR/MWh en vez de ~183). La curva real para buscar precios es SOLO 'O';
    # 'C' únicamente sirve para leer el volumen casado (su suma) y como contraste de
    # precio de casación (su precio máximo).
    summaries: list[dict] = []
    for period_index, g_all in sell.groupby("period_index"):
        g = g_all[g_all["oc"] == "O"].sort_values("precio")
        cum = g["potencia"].cumsum().to_numpy()
        price_sorted = g["precio"].to_numpy()
        matched_volume = float(g_all.loc[g_all["oc"] == "C", "potencia"].sum())
        total_offered = float(g["potencia"].sum())

        def price_at(target_cum: float) -> float | None:
            idx = int(np.searchsorted(cum, target_cum))
            if idx >= len(price_sorted):
                return None
            return float(price_sorted[idx])

        clearing_price = price_at(matched_volume)
        extra_prices = {mw: price_at(matched_volume + mw) for mw in EXTRA_MW_LEVELS}
        slope_500 = (
            (extra_prices[500] - clearing_price) / 500.0
            if clearing_price is not None and extra_prices[500] is not None
            else None
        )
        summaries.append({
            "period_index": int(period_index),
            "matched_volume_mw": matched_volume,
            "clearing_price_from_curve_eur_mwh": clearing_price,
            "total_sell_offered_mw": total_offered,
            "unused_sell_headroom_mw": total_offered - matched_volume,
            "price_at_plus250mw": extra_prices[250],
            "price_at_plus500mw": extra_prices[500],
            "price_at_plus1000mw": extra_prices[1000],
            "price_at_plus2000mw": extra_prices[2000],
            "slope_next500_eur_per_mw": slope_500,
            "n_sell_blocks": int(len(g)),
        })
    return sorted(summaries, key=lambda r: r["period_index"])


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

    n_null = sum(1 for r in rows if r["clearing_price_from_curve_eur_mwh"] is None)
    if n_null:
        status = "WARN"
        messages.append(f"{n_null} periodos sin precio de casación derivable de la curva")

    n_out_of_band = sum(
        1 for r in rows
        if r["clearing_price_from_curve_eur_mwh"] is not None
        and not (PRICE_SANITY_MIN <= r["clearing_price_from_curve_eur_mwh"] <= PRICE_SANITY_MAX)
    )
    if n_out_of_band:
        status = "WARN"
        messages.append(f"{n_out_of_band} periodos con precio de casación fuera de banda plausible")

    return {"status": status, "messages": messages, "expected_periods": expected, "actual_periods": len(rows)}


# ---------------------------------------------------------------------------
# Diccionario de datos
# ---------------------------------------------------------------------------

OMIE_SUPPLY_CURVE_COLUMNS = {
    "delivery_date": ColumnDoc(
        description="Día natural (hora local Europe/Madrid) al que corresponde la entrega.",
        source="Campo 'Fecha' del fichero curva_pbc de OMIE.",
        dtype="DATE", kind="identifier",
    ),
    "period_index": ColumnDoc(
        description="Índice secuencial (1-based) del cuarto de hora dentro del día natural (misma convención que omie_spot_prices.period_index).",
        source="Derivado del campo 'Periodo' (formato H{hora}Q{cuarto}) de cada bloque del fichero.",
        dtype="INTEGER", kind="identifier",
    ),
    "period_start_utc": ColumnDoc(
        description="Instante UTC de inicio del periodo.",
        source="Calculado por period_start_utc(), reutilizada de omie_spot.py (siempre 15 min, todo el rango cubierto por este módulo es posterior al cutover del 2025-10-01).",
        dtype="TIMESTAMP", kind="timestamp",
    ),
    "matched_volume_mw": ColumnDoc(
        description="Volumen de venta (generación) que realmente casó ese periodo. El fichero trae la curva completa (flag 'O') y una traza redundante solo con los bloques casados (flag 'C') — este campo es la suma de esa traza 'C', usada únicamente para saber CUÁNTO casó; el precio y la forma de la curva se leen siempre de la traza 'O' (ver clearing_price_from_curve_eur_mwh).",
        source="Suma de 'Potencia Compra/Venta' donde Tipo Oferta='V' y Ofertada/Casada='C'.",
        dtype="DOUBLE", kind="continuous",
    ),
    "clearing_price_from_curve_eur_mwh": ColumnDoc(
        description="Precio de casación derivado de la curva de venta completa (traza 'O', precio del bloque en el punto de volumen acumulado = matched_volume_mw). Debería coincidir de cerca con el precio oficial del mercado diario (omie_spot_prices) — sirve de contraste de calidad, no sustituye a esa serie.",
        source="Precio del bloque de venta (traza 'O', ordenada por precio) en el punto de volumen acumulado = matched_volume_mw.",
        dtype="DOUBLE", kind="continuous",
    ),
    "total_sell_offered_mw": ColumnDoc(
        description="Volumen TOTAL ofertado en el lado de venta ese periodo, a cualquier precio (casado + no casado) — techo de capacidad disponible a ofertar visto desde el lado de la puja, no desde la capacidad física instalada. Calculado SOLO sobre la traza 'O' (curva completa) — sumarla junto con la traza 'C' duplicaría el volumen ya incluido en 'O'.",
        source="Suma de 'Potencia Compra/Venta' donde Tipo Oferta='V' y Ofertada/Casada='O'.",
        dtype="DOUBLE", kind="continuous",
    ),
    "unused_sell_headroom_mw": ColumnDoc(
        description="total_sell_offered_mw - matched_volume_mw: volumen de venta ofertado que NO llegó a casar — proxy de holgura de oferta visto desde la puja (análogo, desde otro ángulo, a reserve_margin_mw).",
        source="Calculado.",
        dtype="DOUBLE", kind="continuous",
    ),
    "price_at_plus250mw": ColumnDoc(
        description="Precio que habría hecho falta pagar para casar 250 MW MÁS de los que realmente casaron ese periodo, leyendo la curva de venta real más allá del punto de casación.",
        source="Calculado: precio del bloque de venta en el punto de volumen acumulado = matched_volume_mw + 250.",
        dtype="DOUBLE", kind="continuous",
    ),
    "price_at_plus500mw": ColumnDoc(description="Igual que price_at_plus250mw, para 500 MW adicionales.", source="Calculado, ver price_at_plus250mw.", dtype="DOUBLE", kind="continuous"),
    "price_at_plus1000mw": ColumnDoc(description="Igual que price_at_plus250mw, para 1000 MW adicionales.", source="Calculado, ver price_at_plus250mw.", dtype="DOUBLE", kind="continuous"),
    "price_at_plus2000mw": ColumnDoc(description="Igual que price_at_plus250mw, para 2000 MW adicionales.", source="Calculado, ver price_at_plus250mw.", dtype="DOUBLE", kind="continuous"),
    "slope_next500_eur_per_mw": ColumnDoc(
        description="(price_at_plus500mw - clearing_price_from_curve_eur_mwh) / 500 — pendiente de la curva de oferta justo en el margen, en EUR/MWh por MW adicional. Feature principal candidata para explicar el markup de escasez (findings.md, sesión 2026-09-01): cuánto más caro se pone conseguir el siguiente bloque de energía, más allá de qué tecnología está fijando el precio.",
        source="Calculado.",
        dtype="DOUBLE", kind="continuous",
    ),
    "n_sell_blocks": ColumnDoc(
        description="Número de bloques de oferta de venta distintos ese periodo — proxy de fragmentación/complejidad de la curva.",
        source="Recuento de filas con Tipo Oferta='V' en ese periodo.",
        dtype="INTEGER", kind="continuous",
    ),
    "source_file": ColumnDoc(
        description="Nombre del fichero fuente descargado de OMIE, para trazabilidad.",
        source="Construido por el pipeline como curva_pbc_YYYYMMDD.1 a partir de delivery_date.",
        dtype="VARCHAR", kind="identifier",
    ),
    "ingested_at": ColumnDoc(
        description="Marca temporal UTC de la última carga/recarga de esta fila.",
        source="datetime.utcnow() en el momento de la inserción.",
        dtype="TIMESTAMP", kind="timestamp",
    ),
}


def document(con: duckdb.DuckDBPyConnection) -> None:
    catalog.refresh(con, "omie_supply_curve_summary", OMIE_SUPPLY_CURVE_COLUMNS)


# ---------------------------------------------------------------------------
# Schema + load
# ---------------------------------------------------------------------------

def ensure_schema(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS omie_supply_curve_summary (
            delivery_date                       DATE NOT NULL,
            period_index                        INTEGER NOT NULL,
            period_start_utc                    TIMESTAMP NOT NULL,
            matched_volume_mw                   DOUBLE,
            clearing_price_from_curve_eur_mwh    DOUBLE,
            total_sell_offered_mw               DOUBLE,
            unused_sell_headroom_mw             DOUBLE,
            price_at_plus250mw                  DOUBLE,
            price_at_plus500mw                  DOUBLE,
            price_at_plus1000mw                 DOUBLE,
            price_at_plus2000mw                 DOUBLE,
            slope_next500_eur_per_mw            DOUBLE,
            n_sell_blocks                       INTEGER,
            source_file                         VARCHAR NOT NULL,
            ingested_at                         TIMESTAMP NOT NULL,
            PRIMARY KEY (delivery_date, period_index)
        )
        """
    )


def load(con: duckdb.DuckDBPyConnection, delivery_date: date, rows: list[dict], source_file: str) -> int:
    ensure_schema(con)
    now = datetime.utcnow()
    df = pd.DataFrame(
        [
            (
                delivery_date, r["period_index"],
                period_start_utc(delivery_date, r["period_index"], 15).replace(tzinfo=None),
                r["matched_volume_mw"], r["clearing_price_from_curve_eur_mwh"],
                r["total_sell_offered_mw"], r["unused_sell_headroom_mw"],
                r["price_at_plus250mw"], r["price_at_plus500mw"],
                r["price_at_plus1000mw"], r["price_at_plus2000mw"],
                r["slope_next500_eur_per_mw"], r["n_sell_blocks"],
                source_file, now,
            )
            for r in rows
        ],
        columns=[
            "delivery_date", "period_index", "period_start_utc",
            "matched_volume_mw", "clearing_price_from_curve_eur_mwh",
            "total_sell_offered_mw", "unused_sell_headroom_mw",
            "price_at_plus250mw", "price_at_plus500mw", "price_at_plus1000mw", "price_at_plus2000mw",
            "slope_next500_eur_per_mw", "n_sell_blocks", "source_file", "ingested_at",
        ],
    )
    con.execute("BEGIN TRANSACTION")
    try:
        con.execute("DELETE FROM omie_supply_curve_summary WHERE delivery_date = ?", [delivery_date])
        con.append("omie_supply_curve_summary", df)
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
        ["omie_supply_curve", "omie_supply_curve_summary", delivery_date, rows_expected, rows_loaded, status, message,
         started_at, datetime.utcnow()],
    )


def run_for_date(con: duckdb.DuckDBPyConnection, delivery_date: date, logger) -> dict:
    started_at = datetime.utcnow()
    ensure_schema(con)
    source_file = f"curva_pbc_{delivery_date.strftime('%Y%m%d')}.1"

    if delivery_date < CURVE_START_DATE:
        message = f"fuera de alcance, anterior al inicio del backfill pedido ({CURVE_START_DATE.isoformat()})"
        _log_run(con, delivery_date, None, 0, "WARN", message, started_at)
        return {"status": "WARN", "message": message}

    try:
        raw = fetch_raw(delivery_date)
    except Exception as exc:
        _log_run(con, delivery_date, None, 0, "ERROR", str(exc), started_at)
        logger.error("omie_supply_curve %s: %s", delivery_date, exc)
        return {"status": "ERROR", "message": str(exc)}

    if raw is None:
        message = "fichero no encontrado (404) — puede que la sesión de ese día aún no se haya publicado"
        _log_run(con, delivery_date, None, 0, "WARN", message, started_at)
        logger.warning("omie_supply_curve %s: %s", delivery_date, message)
        return {"status": "WARN", "message": message}

    try:
        rows = summarize(raw, delivery_date)
        validation = validate(rows, delivery_date)
    except Exception as exc:
        _log_run(con, delivery_date, None, 0, "ERROR", str(exc), started_at)
        logger.error("omie_supply_curve %s: %s", delivery_date, exc)
        return {"status": "ERROR", "message": str(exc)}

    rows_loaded = load(con, delivery_date, rows, source_file)
    message = "; ".join(validation["messages"]) if validation["messages"] else "sin incidencias"
    _log_run(con, delivery_date, validation["expected_periods"], rows_loaded, validation["status"], message, started_at)
    log_fn = logger.info if validation["status"] == "OK" else logger.warning
    log_fn("omie_supply_curve %s: %s filas cargadas — %s", delivery_date, rows_loaded, message)
    return {"status": validation["status"], "rows_loaded": rows_loaded, "message": message}


if __name__ == "__main__":
    from etl.common.db import connect, document_ingestion_log
    from etl.common.logging_config import get_logger

    log = get_logger("omie_supply_curve.manual", "spain")
    conn = connect("spain")
    print(run_for_date(conn, date(2026, 8, 30), log))
    document(conn)
    document_ingestion_log(conn)
    conn.close()
