"""
Actualización diaria de las bases de datos ENTSO-E (Francia, Países Bajos).

Re-descarga una ventana móvil corta (últimos N días + mañana) para cada área
soportada. La carga es idempotente por rango real de timestamps.

Uso manual:
    .venv\\Scripts\\python.exe -m etl.daily_update_entsoe
"""

from __future__ import annotations

import sys
from datetime import date, timedelta

from etl.common.db import connect, document_ingestion_log
from etl.common.logging_config import get_logger
from etl.sources import entsoe_day_ahead

LOOKBACK_DAYS = 5
LOOKAHEAD_DAYS = 1


def main() -> int:
    today = date.today()
    start = today - timedelta(days=LOOKBACK_DAYS)
    end = today + timedelta(days=LOOKAHEAD_DAYS + 1)  # periodEnd es exclusivo del último día
    period_start = start.strftime("%Y%m%d0000")
    period_end = end.strftime("%Y%m%d0000")

    overall_ok = True
    for area_key, cfg in entsoe_day_ahead.AREAS.items():
        logger = get_logger(f"daily_update.entsoe.{area_key}", cfg["db"])
        logger.info("=== Actualización diaria ENTSO-E %s: %s ===", area_key, today.isoformat())
        con = connect(cfg["db"])

        result = entsoe_day_ahead.run_for_range(con, area_key, period_start, period_end, logger)

        entsoe_day_ahead.document(con, area_key)
        document_ingestion_log(con)
        con.close()

        logger.info("Resumen %s: %s", area_key, result["status"])
        if result["status"] == "ERROR":
            overall_ok = False

    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())
