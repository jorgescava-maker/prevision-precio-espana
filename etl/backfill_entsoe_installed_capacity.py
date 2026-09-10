"""
Backfill histórico de capacidad instalada por tecnología ENTSO-E (documentType A68),
por año, para un área soportada.

Uso:
    .venv\\Scripts\\python.exe -m etl.backfill_entsoe_installed_capacity --area france --start 2023 --end 2026
"""

from __future__ import annotations

import argparse
import sys
import time

from etl.common.db import connect, document_ingestion_log
from etl.common.logging_config import get_logger
from etl.sources import entsoe_installed_capacity as eic_mod
from etl.sources.entsoe_installed_capacity import AREAS


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill histórico ENTSO-E capacidad instalada por tecnología (A68)")
    parser.add_argument("--area", required=True, choices=list(AREAS))
    parser.add_argument("--start", required=True, type=int)
    parser.add_argument("--end", required=True, type=int)
    parser.add_argument("--sleep", type=float, default=0.5)
    args = parser.parse_args()

    if args.start > args.end:
        sys.exit("--start no puede ser posterior a --end")

    db_name = AREAS[args.area]["db"]
    logger = get_logger(f"entsoe_installed_capacity.backfill.{args.area}", db_name)
    con = connect(db_name)

    summary = {"OK": 0, "WARN": 0, "ERROR": 0}
    for year in range(args.start, args.end + 1):
        result = eic_mod.run_for_year(con, args.area, year, logger)
        summary[result["status"]] = summary.get(result["status"], 0) + 1
        time.sleep(args.sleep)
    logger.info("Backfill capacidad instalada %s finalizado: %s", args.area, summary)

    eic_mod.document(con)
    document_ingestion_log(con)
    con.close()

    print(f"\nResumen capacidad instalada ({args.area}): {summary}")


if __name__ == "__main__":
    main()
