"""
Backfill histórico de precios day-ahead vía ENTSO-E, por año natural (UTC), para un
área soportada (ver AREAS en etl/sources/entsoe_day_ahead.py).

Uso:
    .venv\\Scripts\\python.exe -m etl.backfill_entsoe_day_ahead --area france --start 2023 --end 2026
"""

from __future__ import annotations

import argparse
import sys
import time

from etl.common.db import connect, document_ingestion_log
from etl.common.logging_config import get_logger
from etl.sources import entsoe_day_ahead


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill histórico ENTSO-E day-ahead")
    parser.add_argument("--area", required=True, choices=list(entsoe_day_ahead.AREAS))
    parser.add_argument("--start", required=True, type=int, help="año natural, p.ej. 2023")
    parser.add_argument("--end", required=True, type=int, help="año natural, p.ej. 2026")
    parser.add_argument("--sleep", type=float, default=0.5)
    args = parser.parse_args()

    if args.start > args.end:
        sys.exit("--start no puede ser posterior a --end")

    db_name = entsoe_day_ahead.AREAS[args.area]["db"]
    logger = get_logger(f"entsoe_day_ahead.backfill.{args.area}", db_name)
    con = connect(db_name)

    years = list(range(args.start, args.end + 1))
    logger.info("Backfill ENTSO-E %s: años %s", args.area, years)

    summary = {"OK": 0, "WARN": 0, "ERROR": 0}
    errors: list[str] = []

    for year in years:
        period_start = f"{year}01010000"
        period_end = f"{year + 1}01010000"
        result = entsoe_day_ahead.run_for_range(con, args.area, period_start, period_end, logger)
        summary[result["status"]] = summary.get(result["status"], 0) + 1
        if result["status"] == "ERROR":
            errors.append(f"{year}: {result['message']}")
        time.sleep(args.sleep)

    entsoe_day_ahead.document(con, args.area)
    document_ingestion_log(con)
    con.close()

    logger.info("Backfill finalizado. OK=%d WARN=%d ERROR=%d", summary["OK"], summary["WARN"], summary["ERROR"])
    print(f"\nResumen backfill ({args.area}): OK={summary['OK']} WARN={summary['WARN']} ERROR={summary['ERROR']}")
    if errors:
        print("Años con ERROR:")
        for e in errors:
            print(f"  - {e}")


if __name__ == "__main__":
    main()
