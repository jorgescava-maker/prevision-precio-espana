"""
Actualización diaria de demanda y generación por tecnología de España (Península) vía
e·sios. Sin lookahead (dato real/medido, no previsión día-adelantada) — mismo criterio
que etl/daily_update_entsoe_cross_border_flows.py.

Uso manual:
    .venv\\Scripts\\python.exe -m etl.daily_update_esios
"""

from __future__ import annotations

import sys
from datetime import date, timedelta

from etl.common.db import connect, document_ingestion_log
from etl.common.logging_config import get_logger
from etl.sources import esios_load_generation as elg
from etl.sources.esios_load_generation import DB_NAME

LOOKBACK_DAYS = 5


def main() -> int:
    today = date.today()
    start = today - timedelta(days=LOOKBACK_DAYS)
    end = today + timedelta(days=1)
    start_iso = f"{start.isoformat()}T00:00:00"
    end_iso = f"{end.isoformat()}T00:00:00"

    logger = get_logger("daily_update.esios", DB_NAME)
    logger.info("=== Actualización diaria demanda+generación e·sios: %s ===", today.isoformat())
    con = connect(DB_NAME)

    r_demand = elg.run_demand_for_range(con, start_iso, end_iso, logger)
    r_gen = elg.run_generation_for_range(con, start_iso, end_iso, logger)

    elg.document_demand(con)
    elg.document_generation(con)
    document_ingestion_log(con)
    con.close()

    return 0 if r_demand["status"] != "ERROR" and r_gen["status"] != "ERROR" else 1


if __name__ == "__main__":
    sys.exit(main())
