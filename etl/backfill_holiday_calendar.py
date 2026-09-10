"""
Carga completa del calendario de festivos (España/Alemania/Francia/Países Bajos + 5
subregiones NEM). No admite rango por CLI: al ser cálculo local (no ingesta), siempre
recalcula el rango fijo YEAR_START-YEAR_END definido en el pipeline — coste marginal
nulo, no hay razón para trocearlo.

Uso:
    .venv\\Scripts\\python.exe -m etl.backfill_holiday_calendar
"""

from __future__ import annotations

from etl.common.logging_config import get_logger
from etl.sources import holiday_calendar


def main() -> None:
    logger = get_logger("holiday_calendar.backfill", "orchestrator")
    summary = holiday_calendar.run_all(logger)
    logger.info("Backfill calendario de festivos finalizado: %s", summary)
    print(f"\nResumen backfill calendario de festivos: {summary}")


if __name__ == "__main__":
    main()
