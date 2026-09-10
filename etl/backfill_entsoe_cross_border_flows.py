"""
Backfill histórico de flujos físicos transfronterizos (ENTSO-E, documentType A11),
POR MES, para una frontera soportada.

Troceado mensual desde 2026-09-03 (findings.md #129): ENTSO-E ha endurecido el
límite de ventana de este export a P1M — una petición de un año, que es lo que
hacía este script y funcionaba hasta el 2026-08-28, hoy devuelve 400 con el texto
"Provided time interval ... is larger than maximum allowed period 'P1M' for
'NET_CROSS_BORDER_PHYSICAL_FLOWS_R3:XML' export". Verificado en vivo que afecta
también a fronteras ya cargadas (ES-FR, DE-NL), no solo a las nuevas: es un cambio
de la plataforma, no una peculiaridad de frontera. La actualización diaria NO se ve
afectada (pide 6 días).

Uso:
    .venv\\Scripts\\python.exe -m etl.backfill_entsoe_cross_border_flows --border es_fr --start 2023 --end 2026
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import date

from etl.common.db import connect, document_ingestion_log
from etl.common.logging_config import get_logger
from etl.sources import entsoe_cross_border_flows as ecbf
from etl.sources.entsoe_cross_border_flows import BORDERS


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill histórico de flujos físicos transfronterizos ENTSO-E")
    parser.add_argument("--border", required=True, choices=list(BORDERS))
    parser.add_argument("--start", required=True, type=int)
    parser.add_argument("--end", required=True, type=int)
    parser.add_argument("--sleep", type=float, default=0.2)
    args = parser.parse_args()

    if args.start > args.end:
        sys.exit("--start no puede ser posterior a --end")

    db_name = BORDERS[args.border]["db"]
    logger = get_logger(f"entsoe_cross_border_flows.backfill.{args.border}", db_name)
    con = connect(db_name)

    summary = {"OK": 0, "WARN": 0, "ERROR": 0}
    for year in range(args.start, args.end + 1):
        for month in range(1, 13):
            start = date(year, month, 1)
            end = date(year + (month == 12), month % 12 + 1, 1)
            if start > date.today():
                break
            results = ecbf.run_border_for_range(
                con, args.border, start.strftime("%Y%m%d0000"), end.strftime("%Y%m%d0000"), logger
            )
            for result in results:
                summary[result["status"]] = summary.get(result["status"], 0) + 1
            time.sleep(args.sleep)

    ecbf.document(con)
    document_ingestion_log(con)
    con.close()

    logger.info("Backfill %s finalizado: %s", args.border, summary)
    print(f"\nResumen backfill ({args.border}): OK={summary['OK']} WARN={summary['WARN']} ERROR={summary['ERROR']}")


if __name__ == "__main__":
    main()
