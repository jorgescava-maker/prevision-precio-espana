"""
Backfill histórico de clima observado (ERA5) -> weather.duckdb, por año, para todas
las ubicaciones o una en concreto.

Uso:
    .venv\\Scripts\\python.exe -m etl.backfill_weather_era5 --start 2023 --end 2026
    .venv\\Scripts\\python.exe -m etl.backfill_weather_era5 --start 2023 --end 2026 --location spain
"""

from __future__ import annotations

import argparse
import time
from datetime import date, timedelta

from etl.common.db import connect, document_ingestion_log
from etl.common.logging_config import get_logger
from etl.sources import weather_era5
from etl.sources.weather_common import DB_NAME, LOCATIONS, document_actuals


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill histórico ERA5 -> weather.duckdb")
    parser.add_argument("--start", required=True, type=int)
    parser.add_argument("--end", required=True, type=int)
    parser.add_argument("--location", choices=list(LOCATIONS))
    parser.add_argument("--sleep", type=float, default=0.5)
    args = parser.parse_args()

    locations = [args.location] if args.location else list(LOCATIONS)
    # ERA5 tiene retraso de procesamiento: end_date=hoy devuelve 400 (verificado en
    # vivo), end_date=ayer funciona siempre. Tope en "ayer" para no fallar el último tramo.
    today = date.today() - timedelta(days=1)

    logger = get_logger("weather_era5.backfill", DB_NAME)
    con = connect(DB_NAME)

    summary = {"OK": 0, "WARN": 0, "ERROR": 0}
    for location_key in locations:
        for year in range(args.start, args.end + 1):
            start_date = date(year, 1, 1)
            end_date = min(date(year, 12, 31), today)
            if start_date > today:
                continue
            result = weather_era5.run_for_range(con, location_key, start_date.isoformat(), end_date.isoformat(), logger)
            summary[result["status"]] = summary.get(result["status"], 0) + 1
            time.sleep(args.sleep)

    document_actuals(con)
    document_ingestion_log(con)
    con.close()

    logger.info("Backfill ERA5 finalizado: %s", summary)
    print(f"\nResumen backfill ERA5: OK={summary['OK']} WARN={summary['WARN']} ERROR={summary['ERROR']}")


if __name__ == "__main__":
    main()
