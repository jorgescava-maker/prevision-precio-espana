"""
Actualización diaria de commodities.duckdb — tipos de cambio (EUR/USD, EUR/AUD).

Re-descarga los últimos 10 días naturales para cada par — idempotente por rango real
de fechas devueltas. Mismo patrón que daily_update_commodities.py.

Uso manual:
    .venv\\Scripts\\python.exe -m etl.daily_update_fx_rates
"""

from __future__ import annotations

import sys
from datetime import date, datetime, timedelta

from etl.common.db import connect, document_ingestion_log
from etl.common.logging_config import get_logger
from etl.sources import fx_rates

LOOKBACK_DAYS = 10


def main() -> int:
    logger = get_logger("daily_update.fx_rates", "commodities")
    today = date.today()
    start = today - timedelta(days=LOOKBACK_DAYS)
    p1 = int(datetime(start.year, start.month, start.day).timestamp())
    p2 = int(datetime.utcnow().timestamp())

    logger.info("=== Actualización diaria FX: %s ===", today.isoformat())
    con = connect("commodities")

    overall_ok = True
    for key in fx_rates.PAIRS:
        result = fx_rates.run_for_range(con, key, p1, p2, logger)
        if result["status"] == "ERROR":
            overall_ok = False

    fx_rates.document(con)
    document_ingestion_log(con)
    con.close()

    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())
