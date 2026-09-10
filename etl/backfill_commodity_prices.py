"""
Backfill histórico de materias primas (TTF, JKM, Brent) -> commodities.duckdb.

Uso:
    .venv\\Scripts\\python.exe -m etl.backfill_commodity_prices --start 2023-01-01 --end 2026-08-26
"""

from __future__ import annotations

import argparse
import time
from datetime import date, datetime

from etl.common.db import connect, document_ingestion_log
from etl.common.logging_config import get_logger
from etl.sources import commodity_prices


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill histórico de materias primas -> commodities.duckdb")
    parser.add_argument("--start", required=True, type=date.fromisoformat)
    parser.add_argument("--end", required=True, type=date.fromisoformat)
    parser.add_argument("--sleep", type=float, default=0.5)
    args = parser.parse_args()

    logger = get_logger("commodity_prices.backfill", "commodities")
    con = connect("commodities")

    p1 = int(datetime(args.start.year, args.start.month, args.start.day).timestamp())
    p2 = int(datetime(args.end.year, args.end.month, args.end.day, 23, 59).timestamp())

    summary = {"OK": 0, "WARN": 0, "ERROR": 0}
    for key in commodity_prices.COMMODITIES:
        result = commodity_prices.run_for_range(con, key, p1, p2, logger)
        summary[result["status"]] = summary.get(result["status"], 0) + 1
        time.sleep(args.sleep)

    commodity_prices.document(con)
    document_ingestion_log(con)
    con.close()

    logger.info("Backfill finalizado. OK=%d WARN=%d ERROR=%d", summary["OK"], summary["WARN"], summary["ERROR"])
    print(f"\nResumen backfill: OK={summary['OK']} WARN={summary['WARN']} ERROR={summary['ERROR']}")


if __name__ == "__main__":
    main()
