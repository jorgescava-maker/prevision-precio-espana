"""
Actualización diaria de reservas hidráulicas ENTSO-E (España, Francia).

Re-descarga las últimas ~4 semanas — idempotente, y cubre el retraso de publicación
verificado en vivo (~4 días).

Uso manual:
    .venv\\Scripts\\python.exe -m etl.daily_update_hydro_reservoir
"""

from __future__ import annotations

import sys
from datetime import date, timedelta

from etl.common.db import connect, document_ingestion_log
from etl.common.logging_config import get_logger
from etl.sources import entsoe_hydro_reservoir as ehr
from etl.sources.entsoe_hydro_reservoir import AREAS

LOOKBACK_DAYS = 35


def main() -> int:
    today = date.today()
    start = today - timedelta(days=LOOKBACK_DAYS)
    period_start = start.strftime("%Y%m%d0000")
    period_end = today.strftime("%Y%m%d0000")

    overall_ok = True
    for area_key, cfg in AREAS.items():
        logger = get_logger(f"daily_update.hydro.{area_key}", cfg["db"])
        logger.info("=== Actualización diaria reservas hidráulicas %s: %s ===", area_key, today.isoformat())
        con = connect(cfg["db"])

        result = ehr.run_for_range(con, area_key, period_start, period_end, logger)

        ehr.document(con, area_key)
        document_ingestion_log(con)
        con.close()

        if result["status"] == "ERROR":
            overall_ok = False

    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())
