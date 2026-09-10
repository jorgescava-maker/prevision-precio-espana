"""
Backfill histórico de previsión D+1 de demanda, eólica y solar fotovoltaica de España
(Península) vía e·sios, por trimestre (mismo motivo de timeout que
etl/backfill_esios_load_generation.py).

Uso:
    .venv\\Scripts\\python.exe -m etl.backfill_esios_forecast --start 2023 --end 2026
"""

from __future__ import annotations

import argparse
import time
from datetime import date

from etl.common.db import connect, document_ingestion_log
from etl.common.logging_config import get_logger
from etl.sources import esios_forecast as ef
from etl.sources.esios_forecast import DB_NAME


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
    parser = argparse.ArgumentParser(description="Backfill histórico e·sios previsión D+1 demanda + eólica + solar fotovoltaica")
    parser.add_argument("--start", required=True, type=int)
    parser.add_argument("--end", required=True, type=int)
    parser.add_argument("--sleep", type=float, default=0.5)
    args = parser.parse_args()

    if args.start > args.end:
        raise SystemExit("--start no puede ser posterior a --end")

    logger = get_logger("esios_forecast.backfill", DB_NAME)
    con = connect(DB_NAME)

    quarters = list(quarter_range(args.start, args.end))

    summary = {"OK": 0, "WARN": 0, "ERROR": 0}
    for i, (start_iso, end_iso) in enumerate(quarters, start=1):
        result = ef.run_forecast_for_range(con, start_iso, end_iso, logger)
        summary[result["status"]] = summary.get(result["status"], 0) + 1
        if i % 4 == 0 or i == len(quarters):
            logger.info("Progreso previsión: %d/%d trimestres procesados", i, len(quarters))
        time.sleep(args.sleep)
    logger.info("Backfill previsión finalizado: %s", summary)

    ef.document(con)
    document_ingestion_log(con)
    con.close()

    print(f"\nResumen previsión: {summary}")


if __name__ == "__main__":
    main()
