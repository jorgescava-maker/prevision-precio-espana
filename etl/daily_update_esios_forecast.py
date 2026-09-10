"""
Actualización diaria de previsión D+1 de demanda, eólica y solar fotovoltaica de España
(Península) vía e·sios.

Con lookahead: es una previsión día-adelantada, igual criterio que
etl/daily_update_entsoe_ntc.py.

Uso manual:
    .venv\\Scripts\\python.exe -m etl.daily_update_esios_forecast
"""

from __future__ import annotations

import sys
from datetime import date, timedelta

from etl.common.db import connect, document_ingestion_log
from etl.common.logging_config import get_logger
from etl.sources import esios_forecast as ef
from etl.sources.esios_forecast import DB_NAME

LOOKBACK_DAYS = 5
LOOKAHEAD_DAYS = 1


def main() -> int:
    today = date.today()
    start = today - timedelta(days=LOOKBACK_DAYS)
    end = today + timedelta(days=LOOKAHEAD_DAYS + 1)
    start_iso = f"{start.isoformat()}T00:00:00"
    end_iso = f"{end.isoformat()}T00:00:00"

    logger = get_logger("daily_update.esios_forecast", DB_NAME)
    logger.info("=== Actualización diaria previsión D+1 e·sios: %s ===", today.isoformat())
    con = connect(DB_NAME)

    r = ef.run_forecast_for_range(con, start_iso, end_iso, logger)

    ef.document(con)
    document_ingestion_log(con)
    con.close()

    return 0 if r["status"] != "ERROR" else 1


if __name__ == "__main__":
    sys.exit(main())
