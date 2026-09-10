"""
Cálculo completo de HDD/CDD sobre todo el histórico disponible en weather_actuals, para
las 9 ubicaciones (4 países UE + 5 subregiones NEM). Sin rango por CLI: al ser agregación
local sobre datos ya cargados (no ingesta), siempre recalcula sobre el histórico completo
disponible — coste marginal nulo.

Uso:
    .venv\\Scripts\\python.exe -m etl.backfill_degree_days
"""

from __future__ import annotations

from etl.common.logging_config import get_logger
from etl.sources import degree_days


def main() -> None:
    logger = get_logger("degree_days.backfill", "weather")
    summary = degree_days.run_all(logger)
    logger.info("Backfill HDD/CDD finalizado: %s", summary)
    print(f"\nResumen backfill HDD/CDD: {summary}")


if __name__ == "__main__":
    main()
