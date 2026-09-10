"""
Actualización diaria de flujos físicos transfronterizos ENTSO-E — itera automáticamente
sobre todas las fronteras registradas en BORDERS (España-Francia, Alemania-Francia,
Alemania-Países Bajos, España-Portugal).

Sin lookahead: A11 es flujo físico REAL (medido), no una previsión, así que no tiene
sentido pedir días futuros como sí se hace en daily_update_entsoe_load_generation.py.

Uso manual:
    .venv\\Scripts\\python.exe -m etl.daily_update_entsoe_cross_border_flows
"""

from __future__ import annotations

import sys
from datetime import date, timedelta

from etl.common.db import connect, document_ingestion_log
from etl.common.logging_config import get_logger
from etl.sources import entsoe_cross_border_flows as ecbf
from etl.sources.entsoe_cross_border_flows import BORDERS

LOOKBACK_DAYS = 5


def main() -> int:
    today = date.today()
    start = today - timedelta(days=LOOKBACK_DAYS)
    end = today + timedelta(days=1)
    period_start = start.strftime("%Y%m%d0000")
    period_end = end.strftime("%Y%m%d0000")

    overall_ok = True
    for border_key, cfg in BORDERS.items():
        logger = get_logger(f"daily_update.entsoe_flows.{border_key}", cfg["db"])
        logger.info("=== Actualización diaria flujos físicos ENTSO-E %s: %s ===", border_key, today.isoformat())
        con = connect(cfg["db"])

        results = ecbf.run_border_for_range(con, border_key, period_start, period_end, logger)

        ecbf.document(con)
        document_ingestion_log(con)
        con.close()

        if any(r["status"] == "ERROR" for r in results):
            overall_ok = False

    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())
