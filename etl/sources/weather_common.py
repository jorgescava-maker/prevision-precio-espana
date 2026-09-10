"""
Configuración y esquema compartidos por los pipelines de clima (etl/sources/weather_*.py),
todos sobre weather.duckdb. Fuente: API de Open-Meteo (open-meteo.com), gratuita y sin
API key para uso no comercial — verificado en vivo el 2026-08-27 en cada endpoint usado.

Ubicaciones: un único punto representativo por zona de precio (capital/ciudad principal),
NO una media ponderada por población/capacidad instalada — simplificación deliberada de
la primera fase. Los 4 mercados europeos (España/Francia/Alemania/Países Bajos) son zona
de precio única, así que un punto = un país. NEM/Australia NO es zona única: son 5
subregiones de precio (NSW1/QLD1/SA1/TAS1/VIC1, ver etl/sources/aemo_spot.py) con climas
muy distintos entre sí, así que aquí un punto = una subregión, no el país — se añade un
location_key por región (nsw1/qld1/sa1/tas1/vic1), capital de cada estado como ciudad
representativa, mismo patrón y misma simplificación que Europa, solo que aplicada 5 veces
en vez de 1 (decisión 2026-08-27, ver docs/status.md).
"""

from __future__ import annotations

from zoneinfo import ZoneInfo

import duckdb

from etl.common import catalog
from etl.common.catalog import ColumnDoc

DB_NAME = "weather"

LOCATIONS = {
    "spain": {"lat": 40.4168, "lon": -3.7038, "tz": ZoneInfo("Europe/Madrid"), "label": "Madrid"},
    "france": {"lat": 48.8566, "lon": 2.3522, "tz": ZoneInfo("Europe/Paris"), "label": "Paris"},
    "germany": {"lat": 52.5200, "lon": 13.4050, "tz": ZoneInfo("Europe/Berlin"), "label": "Berlin"},
    "netherlands": {"lat": 52.3676, "lon": 4.9041, "tz": ZoneInfo("Europe/Amsterdam"), "label": "Amsterdam"},
    # NEM Australia: un punto por subregión de precio (ver nota del módulo), no por país.
    # Claves alineadas con REGIONS de etl/sources/aemo_spot.py (en minúsculas) para que
    # el estudio B1 pueda cruzar clima<->precio/generación sin tabla de mapeo aparte.
    "nsw1": {"lat": -33.8688, "lon": 151.2093, "tz": ZoneInfo("Australia/Sydney"), "label": "Sydney"},
    "qld1": {"lat": -27.4698, "lon": 153.0251, "tz": ZoneInfo("Australia/Brisbane"), "label": "Brisbane"},
    "sa1": {"lat": -34.9285, "lon": 138.6007, "tz": ZoneInfo("Australia/Adelaide"), "label": "Adelaide"},
    "tas1": {"lat": -42.8821, "lon": 147.3272, "tz": ZoneInfo("Australia/Hobart"), "label": "Hobart"},
    "vic1": {"lat": -37.8136, "lon": 144.9631, "tz": ZoneInfo("Australia/Melbourne"), "label": "Melbourne"},
}

# Variables meteorológicas comunes a actuals y forecast — mismos nombres/unidades en
# ambas tablas para poder comparar directamente (predicho vs. real) sin conversión.
WEATHER_VALUE_COLUMNS = ["temperature_2m_c", "wind_speed_10m_kmh", "wind_speed_100m_kmh",
                          "shortwave_radiation_wm2", "direct_radiation_wm2", "diffuse_radiation_wm2",
                          "precipitation_mm"]

_VALUE_COLUMN_DOCS = {
    "temperature_2m_c": ColumnDoc(
        description="Temperatura del aire a 2m de altura, en grados Celsius.",
        source="Campo temperature_2m de la API de Open-Meteo.",
        dtype="DOUBLE", kind="continuous",
    ),
    "wind_speed_10m_kmh": ColumnDoc(
        description="Velocidad del viento a 10m de altura (altura estándar de estación meteorológica), en km/h.",
        source="Campo wind_speed_10m de la API de Open-Meteo.",
        dtype="DOUBLE", kind="continuous",
    ),
    "wind_speed_100m_kmh": ColumnDoc(
        description="Velocidad del viento a 100m de altura (altura típica de buje de aerogenerador), en km/h. Más relevante que la de 10m para generación eólica.",
        source="Campo wind_speed_100m de la API de Open-Meteo.",
        dtype="DOUBLE", kind="continuous",
    ),
    "shortwave_radiation_wm2": ColumnDoc(
        description="Irradiancia solar global horizontal (GHI), en W/m². Variable principal para generación solar fotovoltaica. NULL para el modelo 'weathernext' (WeatherNext/DeepMind no calcula radiación solar, verificado en vivo — ver docs/findings.md).",
        source="Campo shortwave_radiation de la API de Open-Meteo.",
        dtype="DOUBLE", kind="continuous",
    ),
    "direct_radiation_wm2": ColumnDoc(
        description="Irradiancia solar directa (DNI-like, componente de haz directo), en W/m². Mismo caveat de NULL en 'weathernext' que shortwave_radiation_wm2.",
        source="Campo direct_radiation de la API de Open-Meteo.",
        dtype="DOUBLE", kind="continuous",
    ),
    "diffuse_radiation_wm2": ColumnDoc(
        description="Irradiancia solar difusa (DHI), en W/m². Mismo caveat de NULL en 'weathernext' que shortwave_radiation_wm2.",
        source="Campo diffuse_radiation de la API de Open-Meteo.",
        dtype="DOUBLE", kind="continuous",
    ),
    "precipitation_mm": ColumnDoc(
        description="Precipitación total de la hora, en mm (lluvia + nieve derretida equivalente). Añadida 2026-08-29 (estudio G, clima combinado) — columna al final de la tabla porque se incorporó vía ALTER TABLE sobre las tablas ya existentes, no en la posición 'lógica' junto al resto de variables (ver ensure_schema).",
        source="Campo precipitation de la API de Open-Meteo (todas las fuentes usadas: archive-api, previous-runs-api).",
        dtype="DOUBLE", kind="continuous",
    ),
}


def document_actuals(con: duckdb.DuckDBPyConnection) -> None:
    cols = {
        "location_key": ColumnDoc(
            description="Zona de precio representada: spain/france/germany/netherlands (país, zona única) o nsw1/qld1/sa1/tas1/vic1 (subregión NEM, ver LOCATIONS en el pipeline).",
            source="Constante de configuración, no viene por fila en la respuesta de Open-Meteo.",
            dtype="VARCHAR", kind="categorical",
        ),
        "interval_start_utc": ColumnDoc(
            description="Instante UTC de inicio de la hora.",
            source="Campo 'time' de la respuesta de Open-Meteo (pedido explícitamente en UTC).",
            dtype="TIMESTAMP", kind="identifier",
        ),
        "interval_start_local": ColumnDoc(
            description="Mismo instante en hora local de la ubicación.",
            source="Calculado por el pipeline a partir de interval_start_utc y el tz de LOCATIONS.",
            dtype="TIMESTAMP", kind="timestamp",
        ),
        **_VALUE_COLUMN_DOCS,
        "source_chunk": ColumnDoc(
            description="Identificador del bloque de petición de origen (rango de fechas pedido), para trazabilidad.",
            source="Parámetro de la petición a la API de Open-Meteo.",
            dtype="VARCHAR", kind="identifier",
        ),
        "ingested_at": ColumnDoc(
            description="Marca temporal UTC del momento en que esta fila se cargó (o recargó) por última vez en la base de datos.",
            source="datetime.utcnow() en el momento de la inserción, dentro de la función load().",
            dtype="TIMESTAMP", kind="timestamp",
        ),
    }
    catalog.refresh(con, "weather_actuals", cols)


def document_forecast(con: duckdb.DuckDBPyConnection) -> None:
    cols = {
        "location_key": ColumnDoc(
            description="Zona de precio representada: spain/france/germany/netherlands (país, zona única) o nsw1/qld1/sa1/tas1/vic1 (subregión NEM, ver LOCATIONS en el pipeline).",
            source="Constante de configuración, no viene por fila en la respuesta de Open-Meteo.",
            dtype="VARCHAR", kind="categorical",
        ),
        "model": ColumnDoc(
            description="Modelo de previsión: 'ecmwf_ifs025' (NWP tradicional, con radiación solar) o 'weathernext' (Google DeepMind, ensemble ~50 miembros, promedio; SIN radiación solar).",
            source="Constante de configuración del pipeline que generó la fila.",
            dtype="VARCHAR", kind="categorical",
        ),
        "run_date": ColumnDoc(
            description="Fecha (UTC) en que se emitió/consultó esta previsión concreta — el 'día D' desde el que se predice.",
            source="Para ecmwf_ifs025: derivado del sufijo _previous_dayN de la API (run_date = fecha_objetivo - N). Para weathernext: fecha de ejecución del pipeline diario.",
            dtype="DATE", kind="identifier",
        ),
        "interval_start_utc": ColumnDoc(
            description="Instante UTC de la hora OBJETIVO que se está prediciendo (no la hora de emisión).",
            source="Campo 'time' de la respuesta de Open-Meteo.",
            dtype="TIMESTAMP", kind="identifier",
        ),
        "interval_start_local": ColumnDoc(
            description="Mismo instante objetivo en hora local de la ubicación.",
            source="Calculado por el pipeline a partir de interval_start_utc y el tz de LOCATIONS.",
            dtype="TIMESTAMP", kind="timestamp",
        ),
        "lead_time_days": ColumnDoc(
            description="Horizonte de la previsión en días completos: interval_start_utc.date() - run_date. 0 = mismo día (nowcast/análisis), 1 = D-1 (horizonte día-adelantado, el más relevante para el estudio B1), etc.",
            source="Calculado por el pipeline — derivado de interval_start_utc y run_date, no vuelve a pedirse a la API.",
            dtype="SMALLINT", kind="categorical",
        ),
        **_VALUE_COLUMN_DOCS,
        "source_chunk": ColumnDoc(
            description="Identificador del bloque de petición de origen, para trazabilidad.",
            source="Parámetro de la petición a la API de Open-Meteo.",
            dtype="VARCHAR", kind="identifier",
        ),
        "ingested_at": ColumnDoc(
            description="Marca temporal UTC del momento en que esta fila se cargó (o recargó) por última vez en la base de datos.",
            source="datetime.utcnow() en el momento de la inserción, dentro de la función load().",
            dtype="TIMESTAMP", kind="timestamp",
        ),
    }
    catalog.refresh(con, "weather_forecast", cols)


def ensure_schema(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS weather_actuals (
            location_key           VARCHAR NOT NULL,
            interval_start_utc     TIMESTAMP NOT NULL,
            interval_start_local   TIMESTAMP NOT NULL,
            temperature_2m_c       DOUBLE,
            wind_speed_10m_kmh     DOUBLE,
            wind_speed_100m_kmh    DOUBLE,
            shortwave_radiation_wm2 DOUBLE,
            direct_radiation_wm2   DOUBLE,
            diffuse_radiation_wm2  DOUBLE,
            source_chunk           VARCHAR NOT NULL,
            ingested_at             TIMESTAMP NOT NULL,
            PRIMARY KEY (location_key, interval_start_utc)
        )
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS weather_forecast (
            location_key           VARCHAR NOT NULL,
            model                   VARCHAR NOT NULL,
            run_date                DATE NOT NULL,
            interval_start_utc     TIMESTAMP NOT NULL,
            interval_start_local   TIMESTAMP NOT NULL,
            lead_time_days          SMALLINT NOT NULL,
            temperature_2m_c       DOUBLE,
            wind_speed_10m_kmh     DOUBLE,
            wind_speed_100m_kmh    DOUBLE,
            shortwave_radiation_wm2 DOUBLE,
            direct_radiation_wm2   DOUBLE,
            diffuse_radiation_wm2  DOUBLE,
            source_chunk            VARCHAR NOT NULL,
            ingested_at              TIMESTAMP NOT NULL,
            PRIMARY KEY (location_key, model, run_date, interval_start_utc)
        )
        """
    )
    # precipitation_mm añadida 2026-08-29 sobre tablas ya existentes -> ALTER TABLE la
    # coloca siempre al FINAL del orden físico de columnas (verificado en vivo, no es
    # el comportamiento de un CREATE TABLE nuevo). con.append() de duckdb es POSICIONAL,
    # no por nombre (verificado en vivo) — todo el código que construye el DataFrame a
    # insertar debe poner precipitation_mm en esa misma posición final, después de
    # ingested_at, o los valores se desplazan en silencio a la columna equivocada.
    con.execute("ALTER TABLE weather_actuals ADD COLUMN IF NOT EXISTS precipitation_mm DOUBLE")
    con.execute("ALTER TABLE weather_forecast ADD COLUMN IF NOT EXISTS precipitation_mm DOUBLE")
