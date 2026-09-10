"""
Backfill histórico de precios spot OMIE para España.

Uso:
    .venv\\Scripts\\python.exe -m etl.backfill_omie_spot --start 2026-08-01 --end 2026-08-26
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import date, timedelta

from etl.common.db import connect, document_ingestion_log
from etl.common.logging_config import get_logger
from etl.sources import omie_spot


def daterange(start: date, end: date):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill histórico OMIE spot -> spain.duckdb")
    parser.add_argument("--start", required=True, type=date.fromisoformat, help="YYYY-MM-DD")
    parser.add_argument("--end", required=True, type=date.fromisoformat, help="YYYY-MM-DD")
    parser.add_argument("--sleep", type=float, default=0.3, help="segundos de espera entre peticiones (cortesía al servidor)")
    args = parser.parse_args()

    if args.start > args.end:
        sys.exit("--start no puede ser posterior a --end")

    logger = get_logger("omie_spot.backfill", "spain")
    con = connect("spain")

    summary = {"OK": 0, "WARN": 0, "ERROR": 0}
    errors: list[str] = []

    total_days = (args.end - args.start).days + 1
    logger.info("Backfill OMIE spot: %s -> %s (%d días)", args.start, args.end, total_days)

    for i, d in enumerate(daterange(args.start, args.end), start=1):
        result = omie_spot.run_for_date(con, d, logger)
        summary[result["status"]] = summary.get(result["status"], 0) + 1
        if result["status"] == "ERROR":
            errors.append(f"{d}: {result['message']}")
        if i % 25 == 0 or i == total_days:
            logger.info("Progreso backfill: %d/%d días procesados", i, total_days)
        time.sleep(args.sleep)

    omie_spot.document(con)
    document_ingestion_log(con)
    con.close()

    logger.info("Backfill finalizado. OK=%d WARN=%d ERROR=%d", summary["OK"], summary["WARN"], summary["ERROR"])
    if errors:
        logger.warning("Fechas con ERROR:\n  " + "\n  ".join(errors))
    print(f"\nResumen backfill: OK={summary['OK']} WARN={summary['WARN']} ERROR={summary['ERROR']}")
    if errors:
        print("Fechas con ERROR:")
        for e in errors:
            print(f"  - {e}")


if __name__ == "__main__":
    main()
