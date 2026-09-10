"""
"Actualización diaria" del calendario de festivos. En la práctica recalcula el mismo
rango fijo cada vez (idempotente, sin red, coste despreciable) — se integra en el
orquestador solo por consistencia con el resto de pasos, no porque haya dato nuevo que
pueda llegar día a día.

Uso manual:
    .venv\\Scripts\\python.exe -m etl.daily_update_holiday_calendar
"""

from __future__ import annotations

import sys

from etl.common.logging_config import get_logger
from etl.sources import holiday_calendar


def main() -> int:
    logger = get_logger("daily_update.holiday_calendar", "orchestrator")
    logger.info("=== Actualización diaria calendario de festivos ===")
    summary = holiday_calendar.run_all(logger)
    logger.info("Calendario de festivos: %s", summary)
    return 0 if summary["ERROR"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
