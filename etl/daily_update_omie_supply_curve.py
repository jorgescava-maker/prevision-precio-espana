"""
Actualización diaria de la curva agregada de oferta del mercado diario
(OMIE, fichero curva_pbc). Serie EX-POST (ver docstring de
etl/sources/omie_supply_curve.py): el fichero de un día de entrega se
publica el día SIGUIENTE, no con un día de adelanto como el day-ahead —
por eso el lookback mira hacia atrás, no hacia delante.

Uso manual:
    .venv\\Scripts\\python.exe -m etl.daily_update_omie_supply_curve
"""

from __future__ import annotations

import sys
from datetime import date, timedelta

from etl.common.db import connect, document_ingestion_log
from etl.common.logging_config import get_logger
from etl.sources import omie_supply_curve

LOOKBACK_DAYS = 3


def main() -> int:
    logger = get_logger("daily_update.omie_supply_curve", "spain")
    today = date.today()
    dates_to_check = [today - timedelta(days=offset) for offset in range(LOOKBACK_DAYS, -1, -1)]

    logger.info("=== Actualización diaria OMIE curva de oferta: %s ===", today.isoformat())
    con = connect("spain")

    summary = {"OK": 0, "WARN": 0, "ERROR": 0}
    hard_failures: list[str] = []

    for d in dates_to_check:
        result = omie_supply_curve.run_for_date(con, d, logger)
        summary[result["status"]] = summary.get(result["status"], 0) + 1
        if result["status"] == "ERROR":
            hard_failures.append(f"{d}: {result['message']}")

    omie_supply_curve.document(con)
    document_ingestion_log(con)
    con.close()

    logger.info("Resumen: OK=%d WARN=%d ERROR=%d", summary["OK"], summary["WARN"], summary["ERROR"])
    if hard_failures:
        logger.error("Fallos: %s", hard_failures)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
