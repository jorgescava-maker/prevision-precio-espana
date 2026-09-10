"""
Actualización diaria de clima -> weather.duckdb: ERA5 (últimos días, el reanálisis se
consolida con unos días de retraso), previsión ECMWF (re-captura la ventana reciente
para tener los D-0..D-3 más nuevos) y previsión WeatherNext (acumulación diaria, sin
lookback — cada día es un run nuevo e independiente).

Uso manual:
    .venv\\Scripts\\python.exe -m etl.daily_update_weather
"""

from __future__ import annotations

import sys
from datetime import date, timedelta

from etl.common.db import connect, document_ingestion_log
from etl.common.logging_config import get_logger
from etl.sources import weather_era5, weather_forecast_ecmwf, weather_forecast_weathernext
from etl.sources.weather_common import DB_NAME, LOCATIONS, document_actuals, document_forecast

ERA5_LOOKBACK_DAYS = 10  # ERA5 se consolida con retraso — re-captura por si acaba de publicarse
ERA5_LAG_DAYS = 1  # verificado en vivo: end_date=hoy devuelve 400, end_date=ayer siempre funciona
ECMWF_LOOKBACK_DAYS = 5


def main() -> int:
    today = date.today()
    era5_start = (today - timedelta(days=ERA5_LOOKBACK_DAYS)).isoformat()
    era5_end = (today - timedelta(days=ERA5_LAG_DAYS)).isoformat()
    ecmwf_start = (today - timedelta(days=ECMWF_LOOKBACK_DAYS)).isoformat()
    ecmwf_end = today.isoformat()

    logger = get_logger("daily_update.weather", DB_NAME)
    logger.info("=== Actualización diaria clima: %s ===", today.isoformat())
    con = connect(DB_NAME)

    overall_ok = True
    for location_key in LOCATIONS:
        r1 = weather_era5.run_for_range(con, location_key, era5_start, era5_end, logger)
        r2 = weather_forecast_ecmwf.run_for_range(con, location_key, ecmwf_start, ecmwf_end, logger)
        r3 = weather_forecast_weathernext.run_today(con, location_key, logger)
        if any(r["status"] == "ERROR" for r in (r1, r2, r3)):
            overall_ok = False

    document_actuals(con)
    document_forecast(con)
    document_ingestion_log(con)
    con.close()

    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())
