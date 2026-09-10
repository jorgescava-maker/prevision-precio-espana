"""
Orquestador de la actualización diaria: solo las fuentes que necesita el modelo
de precio del mercado diario de España.

Es el `daily_update_all.py` del proyecto original recortado a los 21 pasos que
alimentan la cadena de España (incluida la parte de Francia, porque el modelo
se entrena con los dos países a la vez, y el precio alemán y los grados-día que
usa el dataset francés). El orden es el mismo que el original: los grados-día
dependen de que el clima ya se haya actualizado en la misma pasada.

Cada paso abre y cierra su propia conexión DuckDB, así que ejecutarlos en el
mismo proceso, uno tras otro, es seguro.

Uso:
    python -m etl.daily_update_espana            # los 21 pasos
    python -m etl.daily_update_espana --rapido   # solo lo necesario para predecir mañana
"""

from __future__ import annotations

import sys
from datetime import datetime

from etl.common.logging_config import get_logger
from etl import (
    daily_update_commodities,
    daily_update_degree_days,
    daily_update_entsoe,
    daily_update_entsoe_cross_border_flows,
    daily_update_entsoe_forecast,
    daily_update_entsoe_generation_outages_spain,
    daily_update_entsoe_generation_units,
    daily_update_entsoe_installed_capacity,
    daily_update_entsoe_load_generation,
    daily_update_entsoe_ntc,
    daily_update_esios,
    daily_update_esios_forecast,
    daily_update_fx_rates,
    daily_update_germany,
    daily_update_holiday_calendar,
    daily_update_hydro_reservoir,
    daily_update_nuclear_outages,
    daily_update_omie_supply_curve,
    daily_update_spain,
    daily_update_thermal_outages,
    daily_update_weather,
)

STEPS = [
    ("España — OMIE precio del mercado diario", daily_update_spain.main),
    ("España — OMIE curva agregada de oferta", daily_update_omie_supply_curve.main),
    ("España — e·sios demanda/generación por tecnología", daily_update_esios.main),
    ("España — e·sios previsión D+1 demanda/eólica/solar", daily_update_esios_forecast.main),
    ("Alemania — SMARD precio day-ahead (lo usa el dataset de Francia)", daily_update_germany.main),
    ("Francia — ENTSO-E precio day-ahead", daily_update_entsoe.main),
    ("Francia — ENTSO-E demanda/generación", daily_update_entsoe_load_generation.main),
    ("Francia — ENTSO-E previsión D-1 demanda/eólica/solar", daily_update_entsoe_forecast.main),
    ("España/Francia — ENTSO-E capacidad instalada por tecnología", daily_update_entsoe_installed_capacity.main),
    ("España — ENTSO-E capacidad instalada por central (gas/carbón)", daily_update_entsoe_generation_units.main),
    ("España — ENTSO-E indisponibilidad de centrales", daily_update_entsoe_generation_outages_spain.main),
    ("Interconexiones — ENTSO-E flujos físicos", daily_update_entsoe_cross_border_flows.main),
    ("Interconexiones — ENTSO-E NTC estimada", daily_update_entsoe_ntc.main),
    ("Materias primas — TTF/EUA/carbón (Yahoo Finance)", daily_update_commodities.main),
    ("Tipos de cambio — EUR/USD (Yahoo Finance)", daily_update_fx_rates.main),
    ("Clima — ERA5 (Open-Meteo)", daily_update_weather.main),
    ("Calendario de festivos", daily_update_holiday_calendar.main),
    ("Grados-día HDD/CDD (derivado del clima)", daily_update_degree_days.main),
    ("España/Francia — ENTSO-E reservas hidráulicas", daily_update_hydro_reservoir.main),
    ("Francia — ENTSO-E indisponibilidad nuclear", daily_update_nuclear_outages.main),
    ("Francia — ENTSO-E indisponibilidad térmica", daily_update_thermal_outages.main),
]


# Lo que cambia de un día para otro y hace falta para predecir España mañana
# ANTES del cierre de la subasta. El resto (sobre todo ENTSO-E de Francia,
# flujos e indisponibilidades francesas, que tardan varios minutos cada uno)
# solo alimenta filas de entrenamiento ya pasadas y puede ir después de publicar.
RAPIDOS = {
    daily_update_spain.main,
    daily_update_esios.main,
    daily_update_esios_forecast.main,
    daily_update_entsoe_generation_outages_spain.main,
    daily_update_entsoe_ntc.main,
    daily_update_commodities.main,
    daily_update_fx_rates.main,
    daily_update_holiday_calendar.main,
}


def main(rapido: bool = False) -> int:
    logger = get_logger("daily_update.espana", "orchestrator")
    logger.info("=== Actualización diaria (modelo de España%s): %s ===",
                ", solo lo rápido" if rapido else "", datetime.now().isoformat())

    results: dict[str, int] = {}
    for name, fn in STEPS:
        if rapido and fn not in RAPIDOS:
            continue
        logger.info("--- %s ---", name)
        try:
            results[name] = fn()
        except Exception:
            logger.exception("Fallo no controlado en: %s", name)
            results[name] = 1

    failed = [name for name, code in results.items() if code != 0]
    if failed:
        logger.error("Actualización diaria terminada CON FALLOS en: %s", failed)
    else:
        logger.info("Actualización diaria completa, sin fallos.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(rapido="--rapido" in sys.argv[1:]))
