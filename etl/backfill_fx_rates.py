"""
Backfill histórico de tipos de cambio (EUR/USD, EUR/AUD) -> commodities.duckdb.

Uso:
    .venv\\Scripts\\python.exe -m etl.backfill_fx_rates --start 2023-01-01 --end 2026-08-29
"""

from __future__ import annotations

import argparse
import time
from datetime import date, datetime

from etl.common.db import connect, document_ingestion_log
from etl.common.logging_config import get_logger
from etl.sources import fx_rates


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill histórico de tipos de cambio -> commodities.duckdb")
    parser.add_argument("--start", required=True, type=date.fromisoformat)
    parser.add_argument("--end", required=True, type=date.fromisoformat)
    parser.add_argument("--sleep", type=float, default=0.5)
    args = parser.parse_args()

    logger = get_logger("fx_rates.backfill", "commodities")
    con = connect("commodities")

    p1 = int(datetime(args.start.year, args.start.month, args.start.day).timestamp())
    p2 = int(datetime(args.end.year, args.end.month, args.end.day, 23, 59).timestamp())

    summary = {"OK": 0, "WARN": 0, "ERROR": 0}
    for key in fx_rates.PAIRS:
        result = fx_rates.run_for_range(con, key, p1, p2, logger)
        summary[result["status"]] = summary.get(result["status"], 0) + 1
        time.sleep(args.sleep)

    fx_rates.document(con)
    document_ingestion_log(con)
    con.close()

    logger.info("Backfill FX finalizado. OK=%d WARN=%d ERROR=%d", summary["OK"], summary["WARN"], summary["ERROR"])
    print(f"\nResumen backfill FX: OK={summary['OK']} WARN={summary['WARN']} ERROR={summary['ERROR']}")


if __name__ == "__main__":
    main()
