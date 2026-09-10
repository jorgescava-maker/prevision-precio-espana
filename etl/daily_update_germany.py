"""
Actualización diaria de la base de datos de Alemania (germany.duckdb).

SMARD sirve los datos en bloques fijos de 7 días; basta con re-descargar el/los
último(s) bloque(s) del índice cada día — la carga es idempotente por rango de
timestamps, así que recargar un bloque ya existente no duplica nada.

Uso manual:
    .venv\\Scripts\\python.exe -m etl.daily_update_germany
"""

from __future__ import annotations

import sys
from datetime import date

from etl.common.db import connect, document_ingestion_log
from etl.common.logging_config import get_logger
from etl.sources import smard_day_ahead

N_LATEST_CHUNKS = 2  # el actual (aún incompleto) + el anterior, por si hubo revisión


def main() -> int:
    logger = get_logger("daily_update.germany", "germany")
    logger.info("=== Actualización diaria Alemania: %s ===", date.today().isoformat())
    con = connect("germany")

    chunks = smard_day_ahead.fetch_index()[-N_LATEST_CHUNKS:]

    summary = {"OK": 0, "WARN": 0, "ERROR": 0}
    hard_failures: list[str] = []

    for chunk_ts in chunks:
        result = smard_day_ahead.run_for_chunk(con, chunk_ts, logger)
        summary[result["status"]] = summary.get(result["status"], 0) + 1
        if result["status"] == "ERROR":
            hard_failures.append(f"{chunk_ts}: {result['message']}")

    smard_day_ahead.document(con)
    document_ingestion_log(con)
    con.close()

    logger.info("Resumen: OK=%d WARN=%d ERROR=%d", summary["OK"], summary["WARN"], summary["ERROR"])
    if hard_failures:
        logger.error("Fallos: %s", hard_failures)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
