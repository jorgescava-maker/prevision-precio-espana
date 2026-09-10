"""
Backfill histórico de capacidad instalada POR CENTRAL ENTSO-E (documentType A71),
por año, para España (único país cargado — ver docstring de
etl/sources/entsoe_generation_units.py).

Uso:
    .venv\\Scripts\\python.exe -m etl.backfill_entsoe_generation_units --start 2023 --end 2026
"""

from __future__ import annotations

import argparse
import sys
import time

from etl.common.db import connect, document_ingestion_log
from etl.common.logging_config import get_logger
from etl.sources import entsoe_generation_units as egu_mod
from etl.sources.entsoe_generation_units import AREAS


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill histórico ENTSO-E capacidad por central (A71)")
    parser.add_argument("--area", default="spain", choices=list(AREAS))
    parser.add_argument("--start", required=True, type=int)
    parser.add_argument("--end", required=True, type=int)
    parser.add_argument("--sleep", type=float, default=0.5)
    args = parser.parse_args()

    if args.start > args.end:
        sys.exit("--start no puede ser posterior a --end")

    db_name = AREAS[args.area]["db"]
    logger = get_logger(f"entsoe_generation_units.backfill.{args.area}", db_name)
    con = connect(db_name)

    summary = {"OK": 0, "WARN": 0, "ERROR": 0}
    for year in range(args.start, args.end + 1):
        result = egu_mod.run_for_year(con, args.area, year, logger)
        summary[result["status"]] = summary.get(result["status"], 0) + 1
        time.sleep(args.sleep)
    logger.info("Backfill capacidad por central %s finalizado: %s", args.area, summary)

    egu_mod.document(con)
    document_ingestion_log(con)
    con.close()

    print(f"\nResumen capacidad por central ({args.area}): {summary}")


if __name__ == "__main__":
    main()
