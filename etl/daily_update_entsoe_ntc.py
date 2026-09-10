"""
Actualización diaria de NTC estimada ENTSO-E (ES-FR, ES-PT).

Con lookahead: a diferencia de daily_update_entsoe_cross_border_flows.py (flujo
físico REAL, sin lookahead), la NTC es una previsión día-adelantada — verificado en
vivo que la NTC de "mañana" ya está publicada, igual que el precio day-ahead.

Uso manual:
    .venv\\Scripts\\python.exe -m etl.daily_update_entsoe_ntc
"""

from __future__ import annotations

import sys
from datetime import date, timedelta

from etl.common.db import connect, document_ingestion_log
from etl.common.logging_config import get_logger
from etl.sources import entsoe_ntc
from etl.sources.entsoe_ntc import NTC_BORDERS

LOOKBACK_DAYS = 5
LOOKAHEAD_DAYS = 1


def main() -> int:
    today = date.today()
    start = today - timedelta(days=LOOKBACK_DAYS)
    end = today + timedelta(days=LOOKAHEAD_DAYS + 1)
    period_start = start.strftime("%Y%m%d0000")
    period_end = end.strftime("%Y%m%d0000")

    overall_ok = True
    for border_key, cfg in NTC_BORDERS.items():
        logger = get_logger(f"daily_update.entsoe_ntc.{border_key}", cfg["db"])
        logger.info("=== Actualización diaria NTC ENTSO-E %s: %s ===", border_key, today.isoformat())
        con = connect(cfg["db"])

        results = entsoe_ntc.run_border_for_range(con, border_key, period_start, period_end, logger)

        entsoe_ntc.document(con)
        document_ingestion_log(con)
        con.close()

        if any(r["status"] == "ERROR" for r in results):
            overall_ok = False

    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())
