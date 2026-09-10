"""
Actualización diaria de HDD/CDD. Recalcula los últimos LOOKBACK_DAYS días (no solo
"hoy") porque weather_actuals llega con 1 día de retraso desde origen (ERA5
archive-api, ver findings.md #38) y puede corregir valores de días recientes ya
cargados con cobertura horaria parcial.

Uso manual:
    .venv\\Scripts\\python.exe -m etl.daily_update_degree_days
"""

from __future__ import annotations

import sys
from datetime import date, timedelta

from etl.common.logging_config import get_logger
from etl.sources import degree_days

LOOKBACK_DAYS = 10


def main() -> int:
    logger = get_logger("daily_update.degree_days", "weather")
    today = date.today()
    start = today - timedelta(days=LOOKBACK_DAYS)

    logger.info("=== Actualización diaria HDD/CDD: %s ===", today.isoformat())
    summary = degree_days.run_all(logger, start_date=start, end_date=today)
    logger.info("HDD/CDD: %s", summary)

    return 0 if summary["ERROR"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
