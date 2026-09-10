"""
Backfill histórico de indisponibilidad de reactores nucleares franceses (ENTSO-E A80),
por mes (cada mes se subdivide automáticamente más si se supera el límite de 200
documentos de ENTSO-E — ver etl/sources/entsoe_nuclear_outages.py).

Uso:
    .venv\\Scripts\\python.exe -m etl.backfill_entsoe_nuclear_outages --start 2023 --end 2026
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime

from etl.common.db import connect, document_ingestion_log
from etl.common.logging_config import get_logger
from etl.sources import entsoe_nuclear_outages as eno


def month_range(start_year: int, end_year: int):
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            yield year, month


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill histórico de indisponibilidad nuclear francesa (ENTSO-E A80)")
    parser.add_argument("--start", required=True, type=int)
    parser.add_argument("--end", required=True, type=int)
    parser.add_argument("--sleep", type=float, default=0.5)
    args = parser.parse_args()

    logger = get_logger("entsoe_nuclear_outages.backfill", "france")
    con = connect("france")

    today = datetime.utcnow()
    summary = {"OK": 0, "ERROR": 0}
    months = list(month_range(args.start, args.end))
    for i, (year, month) in enumerate(months, start=1):
        start_dt = datetime(year, month, 1)
        if start_dt > today:
            break
        next_year, next_month = (year + 1, 1) if month == 12 else (year, month + 1)
        end_dt = min(datetime(next_year, next_month, 1), today)

        result = eno.run_for_range(con, start_dt, end_dt, logger)
        summary[result["status"]] = summary.get(result["status"], 0) + 1
        if i % 6 == 0 or i == len(months):
            logger.info("Progreso: %d/%d meses procesados", i, len(months))
        time.sleep(args.sleep)

    eno.document(con)
    document_ingestion_log(con)
    con.close()

    logger.info("Backfill nuclear francés finalizado: %s", summary)
    print(f"\nResumen backfill nuclear (Francia): OK={summary['OK']} ERROR={summary['ERROR']}")


if __name__ == "__main__":
    main()
