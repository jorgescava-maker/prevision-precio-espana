"""
Actualización diaria de indisponibilidad de reactores nucleares franceses (ENTSO-E A80).

Re-descarga los últimos ~45 días + los próximos ~90 (las paradas programadas se
anuncian con antelación) — idempotente por event_mrid.

Uso manual:
    .venv\\Scripts\\python.exe -m etl.daily_update_nuclear_outages
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta

from etl.common.db import connect, document_ingestion_log
from etl.common.logging_config import get_logger
from etl.sources import entsoe_nuclear_outages as eno

LOOKBACK_DAYS = 45
LOOKAHEAD_DAYS = 90


def main() -> int:
    now = datetime.utcnow()
    start_dt = now - timedelta(days=LOOKBACK_DAYS)
    end_dt = now + timedelta(days=LOOKAHEAD_DAYS)

    logger = get_logger("daily_update.nuclear_outages", "france")
    logger.info("=== Actualización diaria indisponibilidad nuclear Francia: %s ===", now.date().isoformat())
    con = connect("france")

    result = eno.run_for_range(con, start_dt, end_dt, logger)

    eno.document(con)
    document_ingestion_log(con)
    con.close()

    return 0 if result["status"] != "ERROR" else 1


if __name__ == "__main__":
    sys.exit(main())
