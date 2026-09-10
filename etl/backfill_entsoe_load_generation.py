"""
Backfill histórico de demanda (por año) y generación por tecnología (por mes) vía
ENTSO-E, para un área soportada.

Uso:
    .venv\\Scripts\\python.exe -m etl.backfill_entsoe_load_generation --area france --start 2023 --end 2026
"""

from __future__ import annotations

import argparse
import sys
import time

from etl.common.db import connect, document_ingestion_log
from etl.common.logging_config import get_logger
from etl.sources import entsoe_load_generation as elg
from etl.sources.entsoe_load_generation import AREAS


def month_range(start_year: int, end_year: int):
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            yield year, month


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill histórico ENTSO-E demanda + generación por tecnología")
    parser.add_argument("--area", required=True, choices=list(AREAS))
    parser.add_argument("--start", required=True, type=int)
    parser.add_argument("--end", required=True, type=int)
    parser.add_argument("--sleep", type=float, default=1.0)
    args = parser.parse_args()

    if args.start > args.end:
        sys.exit("--start no puede ser posterior a --end")

    db_name = AREAS[args.area]["db"]
    logger = get_logger(f"entsoe_load_generation.backfill.{args.area}", db_name)
    con = connect(db_name)

    # --- Demanda: por año ---
    summary_load = {"OK": 0, "WARN": 0, "ERROR": 0}
    for year in range(args.start, args.end + 1):
        result = elg.run_load_for_range(con, args.area, f"{year}01010000", f"{year + 1}01010000", logger)
        summary_load[result["status"]] = summary_load.get(result["status"], 0) + 1
        time.sleep(args.sleep)
    logger.info("Backfill demanda %s finalizado: %s", args.area, summary_load)

    # --- Generación: por mes (evita timeout del servidor con rangos de 1 año) ---
    summary_gen = {"OK": 0, "WARN": 0, "ERROR": 0}
    months = list(month_range(args.start, args.end))
    for i, (year, month) in enumerate(months, start=1):
        next_year, next_month = (year + 1, 1) if month == 12 else (year, month + 1)
        result = elg.run_generation_for_range(
            con, args.area, f"{year}{month:02d}010000", f"{next_year}{next_month:02d}010000", logger
        )
        summary_gen[result["status"]] = summary_gen.get(result["status"], 0) + 1
        if i % 10 == 0 or i == len(months):
            logger.info("Progreso generación: %d/%d meses procesados", i, len(months))
        time.sleep(args.sleep)
    logger.info("Backfill generación %s finalizado: %s", args.area, summary_gen)

    elg.document_load(con, args.area)
    elg.document_generation(con, args.area)
    document_ingestion_log(con)
    con.close()

    print(f"\nResumen demanda ({args.area}): {summary_load}")
    print(f"Resumen generación ({args.area}): {summary_gen}")


if __name__ == "__main__":
    main()
