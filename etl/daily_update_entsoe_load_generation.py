"""
Actualización diaria de demanda y generación por tecnología ENTSO-E (Alemania,
Francia, Países Bajos).

Uso manual:
    .venv\\Scripts\\python.exe -m etl.daily_update_entsoe_load_generation
"""

from __future__ import annotations

import sys
from datetime import date, timedelta

from etl.common.db import connect, document_ingestion_log
from etl.common.logging_config import get_logger
from etl.sources import entsoe_load_generation as elg
from etl.sources.entsoe_load_generation import AREAS

LOOKBACK_DAYS = 5
LOOKAHEAD_DAYS = 1


def main() -> int:
    today = date.today()
    start = today - timedelta(days=LOOKBACK_DAYS)
    end = today + timedelta(days=LOOKAHEAD_DAYS + 1)
    period_start = start.strftime("%Y%m%d0000")
    period_end = end.strftime("%Y%m%d0000")

    overall_ok = True
    for area_key, cfg in AREAS.items():
        logger = get_logger(f"daily_update.entsoe_lg.{area_key}", cfg["db"])
        logger.info("=== Actualización diaria demanda+generación ENTSO-E %s: %s ===", area_key, today.isoformat())
        con = connect(cfg["db"])

        r_load = elg.run_load_for_range(con, area_key, period_start, period_end, logger)
        r_gen = elg.run_generation_for_range(con, area_key, period_start, period_end, logger)

        elg.document_load(con, area_key)
        elg.document_generation(con, area_key)
        document_ingestion_log(con)
        con.close()

        if r_load["status"] == "ERROR" or r_gen["status"] == "ERROR":
            overall_ok = False

    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())
