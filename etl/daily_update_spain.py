"""
Actualización diaria de la base de datos de España (spain.duckdb).

Pensado para ser lanzado una vez al día por el Programador de Tareas de Windows.
Cada ejecución vuelve a comprobar una ventana móvil de los últimos N días (por si
OMIE republica un fichero corregido) más el día siguiente (que es el que se acaba
de publicar). La carga es idempotente: recargar una fecha ya existente no duplica
filas, simplemente sustituye su contenido.

Uso manual:
    .venv\\Scripts\\python.exe -m etl.daily_update_spain
"""

from __future__ import annotations

import sys
from datetime import date, timedelta

from etl.common.db import connect, document_ingestion_log
from etl.common.logging_config import get_logger
from etl.sources import omie_spot

LOOKBACK_DAYS = 3  # días hacia atrás que se re-comprueban por si hay correcciones tardías
LOOKAHEAD_DAYS = 1  # el day-ahead de "mañana" ya está publicado cuando corre este script


def main() -> int:
    logger = get_logger("daily_update.spain", "spain")
    today = date.today()
    dates_to_check = [
        today + timedelta(days=offset)
        for offset in range(-LOOKBACK_DAYS, LOOKAHEAD_DAYS + 1)
    ]

    logger.info("=== Actualización diaria España: %s ===", today.isoformat())
    con = connect("spain")

    summary = {"OK": 0, "WARN": 0, "ERROR": 0}
    hard_failures: list[str] = []

    for d in dates_to_check:
        result = omie_spot.run_for_date(con, d, logger)
        summary[result["status"]] = summary.get(result["status"], 0) + 1
        # ERROR en el día de mañana o en el propio día de hoy es esperable si aún no se ha
        # publicado; solo lo tratamos como fallo duro si afecta a una fecha ya pasada.
        if result["status"] == "ERROR" and d < today:
            hard_failures.append(f"{d}: {result['message']}")

    omie_spot.document(con)
    document_ingestion_log(con)
    con.close()

    logger.info("Resumen: OK=%d WARN=%d ERROR=%d", summary["OK"], summary["WARN"], summary["ERROR"])
    if hard_failures:
        logger.error("Fallos en fechas ya pasadas (requieren revisión): %s", hard_failures)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
