"""
Backfill histórico de demanda y generación por tecnología de España (Península) vía
e·sios, por trimestre (un año completo da timeout en el servidor de e·sios — ~105.000
filas por indicador y rango, verificado en vivo; un trimestre, ~26.000, se resuelve en
~15s — ver docstring de etl/sources/esios_load_generation.py).

Uso:
    .venv\\Scripts\\python.exe -m etl.backfill_esios_load_generation --start 2023 --end 2026
"""

from __future__ import annotations

import argparse
import time
from datetime import date

from etl.common.db import connect, document_ingestion_log
from etl.common.logging_config import get_logger
from etl.sources import esios_load_generation as elg
from etl.sources.esios_load_generation import DB_NAME


def quarter_range(start_year: int, end_year: int):
    today = date.today()
    for year in range(start_year, end_year + 1):
        for q in range(4):
            start_month = q * 3 + 1
            end_month = start_month + 3
            end_year_ = year if end_month <= 12 else year + 1
            end_month_ = end_month if end_month <= 12 else 1
            start_date = date(year, start_month, 1)
            if start_date > today:
                return
            start_iso = f"{year}-{start_month:02d}-01T00:00:00"
            end_iso = f"{end_year_}-{end_month_:02d}-01T00:00:00"
            yield start_iso, end_iso


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill histórico e·sios demanda + generación por tecnología")
    parser.add_argument("--start", required=True, type=int)
    parser.add_argument("--end", required=True, type=int)
    parser.add_argument("--sleep", type=float, default=0.5)
    args = parser.parse_args()

    if args.start > args.end:
        raise SystemExit("--start no puede ser posterior a --end")

    logger = get_logger("esios_load_generation.backfill", DB_NAME)
    con = connect(DB_NAME)

    quarters = list(quarter_range(args.start, args.end))

    summary_demand = {"OK": 0, "WARN": 0, "ERROR": 0}
    for i, (start_iso, end_iso) in enumerate(quarters, start=1):
        result = elg.run_demand_for_range(con, start_iso, end_iso, logger)
        summary_demand[result["status"]] = summary_demand.get(result["status"], 0) + 1
        if i % 4 == 0 or i == len(quarters):
            logger.info("Progreso demanda: %d/%d trimestres procesados", i, len(quarters))
        time.sleep(args.sleep)
    logger.info("Backfill demanda finalizado: %s", summary_demand)

    summary_gen = {"OK": 0, "WARN": 0, "ERROR": 0}
    for i, (start_iso, end_iso) in enumerate(quarters, start=1):
        result = elg.run_generation_for_range(con, start_iso, end_iso, logger)
        summary_gen[result["status"]] = summary_gen.get(result["status"], 0) + 1
        logger.info("Progreso generación: %d/%d trimestres procesados", i, len(quarters))
        time.sleep(args.sleep)
    logger.info("Backfill generación finalizado: %s", summary_gen)

    elg.document_demand(con)
    elg.document_generation(con)
    document_ingestion_log(con)
    con.close()

    print(f"\nResumen demanda: {summary_demand}")
    print(f"Resumen generación: {summary_gen}")


if __name__ == "__main__":
    main()
