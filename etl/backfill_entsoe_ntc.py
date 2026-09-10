"""
Backfill histórico de NTC estimada (ENTSO-E, documentType A61/A01), por año, para una
frontera soportada (solo ES-FR y ES-PT — ver docs/findings.md sobre por qué DE-FR/DE-NL
quedan fuera de este pipeline).

Uso:
    .venv\\Scripts\\python.exe -m etl.backfill_entsoe_ntc --border es_fr --start 2023 --end 2026
"""

from __future__ import annotations

import argparse
import sys
import time

from etl.common.db import connect, document_ingestion_log
from etl.common.logging_config import get_logger
from etl.sources import entsoe_ntc
from etl.sources.entsoe_ntc import NTC_BORDERS


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill histórico de NTC estimada ENTSO-E")
    parser.add_argument("--border", required=True, choices=list(NTC_BORDERS))
    parser.add_argument("--start", required=True, type=int)
    parser.add_argument("--end", required=True, type=int)
    parser.add_argument("--sleep", type=float, default=1.0)
    args = parser.parse_args()

    if args.start > args.end:
        sys.exit("--start no puede ser posterior a --end")

    db_name = NTC_BORDERS[args.border]["db"]
    logger = get_logger(f"entsoe_ntc.backfill.{args.border}", db_name)
    con = connect(db_name)

    summary = {"OK": 0, "WARN": 0, "ERROR": 0}
    for year in range(args.start, args.end + 1):
        results = entsoe_ntc.run_border_for_range(con, args.border, f"{year}01010000", f"{year + 1}01010000", logger)
        for result in results:
            summary[result["status"]] = summary.get(result["status"], 0) + 1
        time.sleep(args.sleep)

    entsoe_ntc.document(con)
    document_ingestion_log(con)
    con.close()

    logger.info("Backfill NTC %s finalizado: %s", args.border, summary)
    print(f"\nResumen backfill NTC ({args.border}): OK={summary['OK']} WARN={summary['WARN']} ERROR={summary['ERROR']}")


if __name__ == "__main__":
    main()
