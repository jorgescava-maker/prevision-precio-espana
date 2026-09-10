"""
Backfill histórico de reservas hidráulicas (ENTSO-E A72), por año, para un área
soportada (solo España y Francia — ver docs/findings.md sobre por qué Alemania/
Países Bajos no tienen este dato).

Uso:
    .venv\\Scripts\\python.exe -m etl.backfill_entsoe_hydro_reservoir --area spain --start 2023 --end 2026
"""

from __future__ import annotations

import argparse
import sys
import time

from etl.common.db import connect, document_ingestion_log
from etl.common.logging_config import get_logger
from etl.sources import entsoe_hydro_reservoir as ehr
from etl.sources.entsoe_hydro_reservoir import AREAS


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill histórico de reservas hidráulicas ENTSO-E")
    parser.add_argument("--area", required=True, choices=list(AREAS))
    parser.add_argument("--start", required=True, type=int)
    parser.add_argument("--end", required=True, type=int)
    parser.add_argument("--sleep", type=float, default=0.5)
    args = parser.parse_args()

    if args.start > args.end:
        sys.exit("--start no puede ser posterior a --end")

    db_name = AREAS[args.area]["db"]
    logger = get_logger(f"entsoe_hydro_reservoir.backfill.{args.area}", db_name)
    con = connect(db_name)

    summary = {"OK": 0, "WARN": 0, "ERROR": 0}
    for year in range(args.start, args.end + 1):
        result = ehr.run_for_range(con, args.area, f"{year}01010000", f"{year + 1}01010000", logger)
        summary[result["status"]] = summary.get(result["status"], 0) + 1
        time.sleep(args.sleep)

    ehr.document(con, args.area)
    document_ingestion_log(con)
    con.close()

    logger.info("Backfill hydro %s finalizado: %s", args.area, summary)
    print(f"\nResumen backfill hydro ({args.area}): OK={summary['OK']} WARN={summary['WARN']} ERROR={summary['ERROR']}")


if __name__ == "__main__":
    main()
