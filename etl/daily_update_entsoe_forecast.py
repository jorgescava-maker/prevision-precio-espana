"""
Actualización diaria de previsión D-1 de demanda y generación eólica+solar ENTSO-E
(Alemania, Francia, Países Bajos).

Con lookahead: es una previsión día-adelantada, igual criterio que
etl/daily_update_entsoe_ntc.py — verificado en vivo que la previsión de "mañana" ya
está publicada.

Uso manual:
    .venv\\Scripts\\python.exe -m etl.daily_update_entsoe_forecast
"""

from __future__ import annotations

import sys
from datetime import date, timedelta

from etl.common.db import connect, document_ingestion_log
from etl.common.logging_config import get_logger
from etl.sources import entsoe_forecast as ef
from etl.sources.entsoe_forecast import AREAS

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
        logger = get_logger(f"daily_update.entsoe_forecast.{area_key}", cfg["db"])
        logger.info("=== Actualización diaria previsión D-1 ENTSO-E %s: %s ===", area_key, today.isoformat())
        con = connect(cfg["db"])

        r_load = ef.run_load_forecast_for_range(con, area_key, period_start, period_end, logger)
        r_gen = ef.run_generation_forecast_for_range(con, area_key, period_start, period_end, logger)

        ef.document_load_forecast(con, area_key)
        ef.document_generation_forecast(con, area_key)
        document_ingestion_log(con)
        con.close()

        if r_load["status"] == "ERROR" or r_gen["status"] == "ERROR":
            overall_ok = False

    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())
