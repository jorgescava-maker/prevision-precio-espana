"""
Actualización diaria de commodities.duckdb (TTF, JKM, Brent).

Re-descarga los últimos 10 días naturales para cada materia prima — idempotente por
rango real de fechas devueltas.

Uso manual:
    .venv\\Scripts\\python.exe -m etl.daily_update_commodities
"""

from __future__ import annotations

import sys
from datetime import date, datetime, timedelta

from etl.common.db import connect, document_ingestion_log
from etl.common.logging_config import get_logger
from etl.sources import commodity_prices

LOOKBACK_DAYS = 10


def main() -> int:
    logger = get_logger("daily_update.commodities", "commodities")
    today = date.today()
    start = today - timedelta(days=LOOKBACK_DAYS)
    p1 = int(datetime(start.year, start.month, start.day).timestamp())
    p2 = int(datetime.utcnow().timestamp())

    logger.info("=== Actualización diaria commodities: %s ===", today.isoformat())
    con = connect("commodities")

    overall_ok = True
    for key in commodity_prices.COMMODITIES:
        result = commodity_prices.run_for_range(con, key, p1, p2, logger)
        if result["status"] == "ERROR":
            overall_ok = False

    commodity_prices.document(con)
    document_ingestion_log(con)
    con.close()

    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())
