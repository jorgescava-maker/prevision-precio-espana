"""
Backfill histórico de precios day-ahead de Alemania (SMARD) -> germany.duckdb.

Uso:
    .venv\\Scripts\\python.exe -m etl.backfill_smard_day_ahead --start 2023-01-01 --end 2026-08-26
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import date, datetime, timezone

from etl.common.db import connect, document_ingestion_log
from etl.common.logging_config import get_logger
from etl.sources import smard_day_ahead


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill histórico SMARD day-ahead -> germany.duckdb")
    parser.add_argument("--start", required=True, type=date.fromisoformat, help="YYYY-MM-DD")
    parser.add_argument("--end", required=True, type=date.fromisoformat, help="YYYY-MM-DD")
    parser.add_argument("--sleep", type=float, default=0.2, help="segundos de espera entre peticiones")
    args = parser.parse_args()

    if args.start > args.end:
        sys.exit("--start no puede ser posterior a --end")

    logger = get_logger("smard_day_ahead.backfill", "germany")
    con = connect("germany")

    all_chunks = smard_day_ahead.fetch_index()
    # Selecciona todo bloque cuyo rango [ts, ts+7d) pueda solapar con [start, end],
    # con un margen de un bloque completo a cada lado por seguridad.
    start_ms = int(datetime(args.start.year, args.start.month, args.start.day, tzinfo=timezone.utc).timestamp() * 1000)
    end_ms = int(datetime(args.end.year, args.end.month, args.end.day, tzinfo=timezone.utc).timestamp() * 1000)
    start_ms -= smard_day_ahead.CHUNK_SPAN_MS
    end_ms += smard_day_ahead.CHUNK_SPAN_MS
    chunks = [c for c in all_chunks if start_ms <= c <= end_ms]

    logger.info("Backfill SMARD DE-LU: %s -> %s (%d bloques semanales)", args.start, args.end, len(chunks))

    summary = {"OK": 0, "WARN": 0, "ERROR": 0}
    errors: list[str] = []

    for i, chunk_ts in enumerate(chunks, start=1):
        result = smard_day_ahead.run_for_chunk(con, chunk_ts, logger)
        summary[result["status"]] = summary.get(result["status"], 0) + 1
        if result["status"] == "ERROR":
            errors.append(f"{chunk_ts}: {result['message']}")
        if i % 20 == 0 or i == len(chunks):
            logger.info("Progreso backfill: %d/%d bloques procesados", i, len(chunks))
        time.sleep(args.sleep)

    smard_day_ahead.document(con)
    document_ingestion_log(con)
    con.close()

    logger.info("Backfill finalizado. OK=%d WARN=%d ERROR=%d", summary["OK"], summary["WARN"], summary["ERROR"])
    print(f"\nResumen backfill: OK={summary['OK']} WARN={summary['WARN']} ERROR={summary['ERROR']}")
    if errors:
        print("Bloques con ERROR:")
        for e in errors:
            print(f"  - {e}")


if __name__ == "__main__":
    main()
