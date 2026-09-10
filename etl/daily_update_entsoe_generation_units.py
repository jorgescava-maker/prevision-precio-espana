"""
Actualización diaria de capacidad instalada por central ENTSO-E. Es un dato "year
ahead" que cambia poco a lo largo del año — se refresca solo el año en curso y el
siguiente, no todo el histórico. Mismo patrón que daily_update_entsoe_installed_capacity.py.

Uso manual:
    .venv\\Scripts\\python.exe -m etl.daily_update_entsoe_generation_units
"""

from __future__ import annotations

import sys
from datetime import date

from etl.common.db import connect, document_ingestion_log
from etl.common.logging_config import get_logger
from etl.sources import entsoe_generation_units as egu_mod
from etl.sources.entsoe_generation_units import AREAS


def main() -> int:
    today = date.today()
    overall_ok = True
    for area_key, cfg in AREAS.items():
        logger = get_logger(f"daily_update.entsoe_generation_units.{area_key}", cfg["db"])
        logger.info("=== Actualización diaria capacidad por central ENTSO-E %s: %s ===", area_key, today.isoformat())
        con = connect(cfg["db"])

        for year in (today.year, today.year + 1):
            result = egu_mod.run_for_year(con, area_key, year, logger)
            if result["status"] == "ERROR":
                overall_ok = False

        egu_mod.document(con)
        document_ingestion_log(con)
        con.close()

    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())
