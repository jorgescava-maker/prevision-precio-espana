"""
Ingesta de flujos físicos transfronterizos (documentType A11, "Cross-Border Physical
Flow") vía ENTSO-E Transparency Platform. Fronteras cubiertas: España-Francia,
Alemania-Francia, Alemania-Países Bajos, España-Portugal, Alemania-Bélgica,
Alemania-Dinamarca (DK1 y DK2 por separado, ver nota 6), Alemania-Polonia,
Alemania-Austria, Alemania-Suiza (Fase 10 del roadmap, alimenta el estudio D1 de
interconexiones); desde 2026-09-03 también las cuatro de Países Bajos (NL-GB, NL-NO2,
NL-DK1, NL-BE) y las dos que faltaban de Alemania (DE-NO2, DE-SE4) — ver nota 7.

Requiere ENTSOE_API_TOKEN en .env (mismo token que day-ahead/demanda/generación) —
NO el token de e·sios/REE, que es específico de datos internos de España y no hace
falta para flujos transfronterizos publicados por ENTSO-E (verificado en vivo
2026-08-27 con ES-PT: responde con el token de ENTSO-E ya en uso, sin necesitar nada
adicional).

Francia y Países Bajos NO comparten frontera eléctrica física (Bélgica está en medio)
— verificado en vivo el 2026-08-26 pidiendo ese par a ENTSO-E, que devuelve
explícitamente "No matching data found" en vez de una serie vacía o un error. No hay
entrada "FR-NL" en BORDERS por eso, no por omisión.

Hechos verificados en vivo el 2026-08-26/27, críticos para un parseo correcto:

1. ENTSO-E publica el flujo físico de cada frontera como DOS series independientes,
   una por sentido (in_Domain=área destino, out_Domain=área origen en cada petición —
   el valor devuelto es el flujo origen->destino). El valor es siempre >= 0 (verificado:
   0 negativos en 28.920 puntos/sentido a lo largo de 2023 completo) — cuando el flujo
   neto va en un sentido, el sentido contrario reporta ~0, no un valor negativo. Por eso
   hay que pedir las dos direcciones por separado y guardarlas como dos series (from/to),
   no fusionarlas en una sola columna con signo.
2. Reutiliza el parseo común de etl/common/entsoe_xml.py (forward-fill de puntos
   omitidos + deduplicación de TimeSeries repetidos) — confirmado que aplica igual que
   en precio/demanda/generación (verificado en vivo: posiciones 2-10 omitidas en un
   bloque de ejemplo porque el valor no cambiaba respecto a la posición 1).
3. El corte de resolución horaria -> cuartohoraria es PROPIO DE CADA FRONTERA, no un
   corte único compartido: ES-FR pasó a cuartos de hora el 2023-03-27; DE-FR el
   2025-04-18; DE-NL ya estaba en cuartos de hora en enero de 2023 (desde el inicio
   del histórico cargado); ES-PT el 2025-05-23/24. Ninguna coincide con la reforma
   paneuropea del 2025-10-01 ni entre sí. Verificado en vivo pidiendo tramos de cada
   frontera antes de dar nada por hecho. El parseo no necesita saber la fecha de corte
   de antemano porque lee <resolution> por cada <Period>, pero ningún estudio debe
   asumir resolución homogénea dentro de un mismo año para estas series, ni inferir la
   fecha de corte de una frontera por analogía con otra.
4. Portugal usa huso horario WET/WEST (Europe/Lisbon, UTC+0/+1), distinto de
   CET/CEST (España/Francia/Alemania/Países Bajos, UTC+1/+2) — la primera frontera de
   este pipeline con áreas de huso distinto. interval_start_local usa el huso del área
   EXPORTADORA (from_area) de cada fila, así que las dos filas del mismo instante en
   ES-PT (spain->portugal y portugal->spain) pueden mostrar interval_start_local con
   una hora de diferencia entre sí — es el comportamiento esperado, documentado en el
   ColumnDoc de esa columna, no un bug. Bélgica/Dinamarca/Polonia/Austria/Suiza
   comparten CET/CEST con Alemania, sin este caveat.
5. Una petición de un año completo no da timeout (~6s, igual que el precio day-ahead;
   a diferencia de la generación por tecnología A75, que sí lo necesita trocear por
   mes — ver docs/findings.md #19).
6. Fronteras alemanas añadidas 2026-08-27 (verificado en vivo, ver docs/findings.md):
   flujo físico A11 responde con datos reales en las 6 (DE-BE, DE-DK1, DE-DK2, DE-PL,
   DE-AT, DE-CH). Dinamarca tiene DOS interconexiones eléctricas distintas con
   Alemania — DK1 (Jutlandia, AC clásica) y DK2 (Kriegers Flak, enlace híbrido con el
   parque eólico marino homónimo, operativo desde dic-2020) — por eso son dos
   entradas separadas en BORDERS, igual que ES-FR/ES-PT son entradas separadas y no
   una única "España".
7. Fronteras añadidas 2026-09-03 (findings.md #124 huecos nº1 y nº4, #129): las
   cuatro de Países Bajos —BritNed (NL-GB), NorNed (NL-NO2), COBRAcable (NL-DK1) y
   NL-BE— y las dos que faltaban de Alemania —NordLink (DE-NO2) y Baltic Cable
   (DE-SE4)—. Verificadas en vivo: las 6 devuelven flujo real en ambos sentidos.
   Dos avisos de esa verificación:
   - **NorNed (NL-NO2) está a cero desde ~junio de 2026**: 0 puntos distintos de
     cero en todo junio y 40 en agosto, contra ~2.700/mes antes. Es
     indisponibilidad real del enlace en origen, no un fallo de carga (los meses
     anteriores vienen completos y con valores normales). No silenciarlo ni
     "arreglarlo".
   - Ninguna de las 6 tiene NTC explícita (A61): la frontera nórdica-continental y
     NL-BE/NL-GB se asignan de forma implícita, mismo motivo ya documentado en
     etl/sources/entsoe_ntc.py para las fronteras de Core. No añadirlas a
     NTC_BORDERS.
8. **La ventana máxima de petición de este export bajó a P1M** (verificado el
   2026-09-03; hasta el 2026-08-28 se pedían años enteros sin problema). Una
   petición mayor devuelve 400 con "larger than maximum allowed period 'P1M' for
   'NET_CROSS_BORDER_PHYSICAL_FLOWS_R3:XML'". Afecta a TODAS las fronteras, también
   a las ya cargadas — por eso el backfill trocea por mes. La actualización diaria
   no se ve afectada (pide 6 días). findings.md #129.
"""

from __future__ import annotations

import time
from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo

import duckdb
import pandas as pd
import requests

from etl.common import catalog, entsoe_xml
from etl.common.catalog import ColumnDoc
from etl.common.config import require_env

API_URL = "https://web-api.tp.entsoe.eu/api"

# Una frontera = un par de áreas EIC. "db" es la base de datos DuckDB destino —
# separada por dominio (interconnections), no por país, porque el dato pertenece a la
# pareja de países, no a uno solo (patrón nuevo respecto a spain/france/... por país,
# pensado para poder añadir más fronteras sin recrear bases de países).
#
# flow_sanity_max_mw es una banda de plausibilidad POR FRONTERA (no global): la
# capacidad real varía mucho entre interconexiones, y usar un único umbral generaría
# WARN falsos en las fronteras de mayor capacidad. Cada valor tiene margen sobre el
# máximo observado en vivo al verificar la frontera (ver docs/findings.md).
#
# NOTA: Francia y Países Bajos NO comparten frontera eléctrica física (Bélgica está
# en medio) — verificado en vivo el 2026-08-26, ENTSO-E devuelve explícitamente "No
# matching data found" para ese par. No añadir "FR-NL" aquí.
BORDERS = {
    "es_fr": {
        "border_key": "ES-FR",
        "db": "interconnections",
        "flow_sanity_max_mw": 6000.0,  # máximo observado en 2023-2026 completo: 3968 MW
        "areas": {
            "spain": {"eic": "10YES-REE------0", "tz": ZoneInfo("Europe/Madrid")},
            "france": {"eic": "10YFR-RTE------C", "tz": ZoneInfo("Europe/Paris")},
        },
    },
    "de_fr": {
        "border_key": "DE-FR",
        "db": "interconnections",
        "flow_sanity_max_mw": 7000.0,  # máximo observado en enero 2026: 4918 MW
        "areas": {
            "germany": {"eic": "10Y1001A1001A82H", "tz": ZoneInfo("Europe/Berlin")},
            "france": {"eic": "10YFR-RTE------C", "tz": ZoneInfo("Europe/Paris")},
        },
    },
    "de_nl": {
        "border_key": "DE-NL",
        "db": "interconnections",
        "flow_sanity_max_mw": 8500.0,  # máximo observado en 2023 completo: 6232 MW
        "areas": {
            "germany": {"eic": "10Y1001A1001A82H", "tz": ZoneInfo("Europe/Berlin")},
            "netherlands": {"eic": "10YNL----------L", "tz": ZoneInfo("Europe/Amsterdam")},
        },
    },
    "es_pt": {
        "border_key": "ES-PT",
        "db": "interconnections",
        "flow_sanity_max_mw": 6000.0,  # máximo observado en 2023 completo: 4669 MW
        "areas": {
            # Portugal usa WET/WEST (Europe/Lisbon, UTC+0/+1), NO CET/CEST como
            # España/Francia/Alemania/Países Bajos — primera frontera con huso
            # horario distinto a cada lado (ver nota de interval_start_local).
            "spain": {"eic": "10YES-REE------0", "tz": ZoneInfo("Europe/Madrid")},
            "portugal": {"eic": "10YPT-REN------W", "tz": ZoneInfo("Europe/Lisbon")},
        },
    },
    "de_be": {
        "border_key": "DE-BE",
        "db": "interconnections",
        "flow_sanity_max_mw": 2000.0,  # máximo observado en 2023 completo: 1278 MW
        "areas": {
            "germany": {"eic": "10Y1001A1001A82H", "tz": ZoneInfo("Europe/Berlin")},
            "belgium": {"eic": "10YBE----------2", "tz": ZoneInfo("Europe/Brussels")},
        },
    },
    "de_dk1": {
        "border_key": "DE-DK1",
        "db": "interconnections",
        "flow_sanity_max_mw": 4000.0,  # máximo observado en 2023 completo: 2954 MW
        "areas": {
            # DK1 = Jutlandia/Fionia, interconexión AC clásica con Alemania desde
            # antes de 2023 (distinta de DK2/Kriegers Flak, ver nota 6 del módulo).
            "germany": {"eic": "10Y1001A1001A82H", "tz": ZoneInfo("Europe/Berlin")},
            "denmark_dk1": {"eic": "10YDK-1--------W", "tz": ZoneInfo("Europe/Copenhagen")},
        },
    },
    "de_dk2": {
        "border_key": "DE-DK2",
        "db": "interconnections",
        "flow_sanity_max_mw": 1500.0,  # máximo observado en 2023 completo: 1004 MW
        "areas": {
            # DK2 = Copenhague/Zealand, vía el enlace híbrido Kriegers Flak (combined
            # grid solution con el parque eólico marino homónimo, operativo desde
            # dic-2020) — sin NTC explícita (ver etl/sources/entsoe_ntc.py).
            "germany": {"eic": "10Y1001A1001A82H", "tz": ZoneInfo("Europe/Berlin")},
            "denmark_dk2": {"eic": "10YDK-2--------M", "tz": ZoneInfo("Europe/Copenhagen")},
        },
    },
    "de_pl": {
        "border_key": "DE-PL",
        "db": "interconnections",
        "flow_sanity_max_mw": 3000.0,  # máximo observado en 2023 completo: 2119 MW
        "areas": {
            "germany": {"eic": "10Y1001A1001A82H", "tz": ZoneInfo("Europe/Berlin")},
            "poland": {"eic": "10YPL-AREA-----S", "tz": ZoneInfo("Europe/Warsaw")},
        },
    },
    "de_at": {
        "border_key": "DE-AT",
        "db": "interconnections",
        "flow_sanity_max_mw": 5500.0,  # máximo observado en 2023 completo: 3861 MW
        "areas": {
            "germany": {"eic": "10Y1001A1001A82H", "tz": ZoneInfo("Europe/Berlin")},
            "austria": {"eic": "10YAT-APG------L", "tz": ZoneInfo("Europe/Vienna")},
        },
    },
    "de_ch": {
        "border_key": "DE-CH",
        "db": "interconnections",
        "flow_sanity_max_mw": 6500.0,  # máximo observado en 2023 completo: 4626 MW
        "areas": {
            "germany": {"eic": "10Y1001A1001A82H", "tz": ZoneInfo("Europe/Berlin")},
            "switzerland": {"eic": "10YCH-SWISSGRIDZ", "tz": ZoneInfo("Europe/Zurich")},
        },
    },
    # Fronteras menores añadidas 2026-08-28 (ver docs/findings.md #68): las 7
    # verificadas en vivo de la lista pendiente en status.md.
    "fr_be": {
        "border_key": "FR-BE",
        "db": "interconnections",
        "flow_sanity_max_mw": 6000.0,  # máximo observado en backfill 2023-2026: 5530 MW
        "areas": {
            "france": {"eic": "10YFR-RTE------C", "tz": ZoneInfo("Europe/Paris")},
            "belgium": {"eic": "10YBE----------2", "tz": ZoneInfo("Europe/Brussels")},
        },
    },
    "fr_ch": {
        "border_key": "FR-CH",
        "db": "interconnections",
        "flow_sanity_max_mw": 5500.0,  # máximo observado en backfill 2023-2026: 5061 MW
        "areas": {
            "france": {"eic": "10YFR-RTE------C", "tz": ZoneInfo("Europe/Paris")},
            "switzerland": {"eic": "10YCH-SWISSGRIDZ", "tz": ZoneInfo("Europe/Zurich")},
        },
    },
    "fr_it": {
        "border_key": "FR-IT",
        "db": "interconnections",
        # Banda deliberadamente MÁS BAJA que el máximo bruto observado (9.841 MW,
        # 2023-06-12): esa cifra y varias más de esa misma semana (5.500-9.841 MW)
        # superan casi 3× la capacidad física conocida de la interconexión (~3.200-
        # 3.400 MW combinados) — percentil 99,9% real de toda la serie: 4.680 MW. Se
        # trata como anomalía de origen de esa semana concreta (igual criterio que
        # `enlace_baleares` en esios_load_generation.py), no se sube la banda para
        # silenciarla: los WARN de esa semana quedan para quien analice la serie.
        "flow_sanity_max_mw": 5000.0,
        "areas": {
            # Italia representada por el EIC agregado nacional (10YIT-GRTN-----B),
            # no por una de sus zonas de precio internas — verificado en vivo que
            # ENTSO-E acepta este EIC para el flujo físico transfronterizo.
            "france": {"eic": "10YFR-RTE------C", "tz": ZoneInfo("Europe/Paris")},
            "italy": {"eic": "10YIT-GRTN-----B", "tz": ZoneInfo("Europe/Rome")},
        },
    },
    "at_ch": {
        "border_key": "AT-CH",
        "db": "interconnections",
        "flow_sanity_max_mw": 2800.0,  # máximo observado en backfill 2023-2026: 2244 MW
        "areas": {
            "austria": {"eic": "10YAT-APG------L", "tz": ZoneInfo("Europe/Vienna")},
            "switzerland": {"eic": "10YCH-SWISSGRIDZ", "tz": ZoneInfo("Europe/Zurich")},
        },
    },
    "at_it": {
        "border_key": "AT-IT",
        "db": "interconnections",
        "flow_sanity_max_mw": 800.0,  # máximo observado en backfill 2023-2026: 508 MW
        "areas": {
            "austria": {"eic": "10YAT-APG------L", "tz": ZoneInfo("Europe/Vienna")},
            "italy": {"eic": "10YIT-GRTN-----B", "tz": ZoneInfo("Europe/Rome")},
        },
    },
    "pl_cz": {
        "border_key": "PL-CZ",
        "db": "interconnections",
        "flow_sanity_max_mw": 3000.0,  # máximo observado en backfill 2023-2026: 2487 MW
        "areas": {
            "poland": {"eic": "10YPL-AREA-----S", "tz": ZoneInfo("Europe/Warsaw")},
            "czechia": {"eic": "10YCZ-CEPS-----N", "tz": ZoneInfo("Europe/Prague")},
        },
    },
    "pl_se4": {
        "border_key": "PL-SE4",
        "db": "interconnections",
        "flow_sanity_max_mw": 1000.0,  # máximo observado en backfill 2023-2026: 638 MW, coherente con la capacidad nominal conocida del SwePol Link (~600 MW)
        "areas": {
            # SwePol Link: enlace HVDC punto a punto entre Polonia y la zona de
            # precio sueca SE4 (Suecia no tiene un único EIC nacional — 4 zonas de
            # precio — este es el único de los 4 con interconexión física a Polonia).
            "poland": {"eic": "10YPL-AREA-----S", "tz": ZoneInfo("Europe/Warsaw")},
            "sweden_se4": {"eic": "10Y1001A1001A47J", "tz": ZoneInfo("Europe/Stockholm")},
        },
    },
    # Fronteras de Países Bajos y Alemania añadidas 2026-09-03 (findings.md #124,
    # huecos nº1 y nº4): Países Bajos es un mercado pequeño rodeado de interconectores
    # y solo teníamos DE-NL; Alemania no tenía su enlace con la hidráulica noruega.
    # Las 6 verificadas en vivo el 2026-09-03 (flujo A11 con datos reales en ambos
    # sentidos). Las tres holandesas con Reino Unido/Noruega/Dinamarca son enlaces
    # HVDC punto a punto con nombre propio; NL-BE es frontera AC clásica.
    "nl_gb": {
        "border_key": "NL-GB",
        "db": "interconnections",
        "flow_sanity_max_mw": 1400.0,  # BritNed, capacidad nominal 1.000 MW
        "areas": {
            # BritNed: HVDC Maasvlakte-Isle of Grain, operativo desde 2011. Reino
            # Unido salió del mercado único europeo pero sigue publicando el flujo
            # físico en ENTSO-E (verificado en vivo 2026-09-03). Huso Europe/London
            # (WET/WEST), una hora por detrás de CET — mismo caveat que Portugal en
            # ES-PT para interval_start_local (ver nota 4 del módulo).
            "netherlands": {"eic": "10YNL----------L", "tz": ZoneInfo("Europe/Amsterdam")},
            "great_britain": {"eic": "10YGB----------A", "tz": ZoneInfo("Europe/London")},
        },
    },
    "nl_no2": {
        "border_key": "NL-NO2",
        "db": "interconnections",
        "flow_sanity_max_mw": 900.0,  # NorNed, capacidad nominal 700 MW
        "areas": {
            # NorNed: HVDC Eemshaven-Feda, 580 km, el cable submarino más largo del
            # mundo cuando entró en 2008. Conecta con NO2, la zona de precio noruega
            # del suroeste (Noruega tiene 5 zonas; NO2 es la única con enlace a NL).
            # AVISO verificado en vivo 2026-09-03: el enlace está a cero desde
            # ~junio de 2026 (indisponibilidad real en origen, no fallo de carga).
            "netherlands": {"eic": "10YNL----------L", "tz": ZoneInfo("Europe/Amsterdam")},
            "norway_no2": {"eic": "10YNO-2--------T", "tz": ZoneInfo("Europe/Oslo")},
        },
    },
    "nl_dk1": {
        "border_key": "NL-DK1",
        "db": "interconnections",
        "flow_sanity_max_mw": 900.0,  # COBRAcable, capacidad nominal 700 MW
        "areas": {
            # COBRAcable: HVDC Eemshaven-Endrup, operativo desde 2019. DK1 =
            # Jutlandia, la misma zona de precio que en de_dk1.
            "netherlands": {"eic": "10YNL----------L", "tz": ZoneInfo("Europe/Amsterdam")},
            "denmark_dk1": {"eic": "10YDK-1--------W", "tz": ZoneInfo("Europe/Copenhagen")},
        },
    },
    "nl_be": {
        "border_key": "NL-BE",
        "db": "interconnections",
        # Banda subida de 4.000 a 5.000 tras el backfill (2026-09-03): el máximo
        # real es 4.425 MW y el percentil 99,9% 3.776, con solo 19 intervalos de
        # 72.520 por encima de 4.000. Es flujo físico plausible en una frontera AC
        # —por donde pasan flujos de tránsito, no solo el intercambio comercial de
        # ~2.400 MW— y no una anomalía de origen como la de FR-IT, que superaba 3×
        # la capacidad conocida. La banda se ajusta al dato observado, no al revés.
        "flow_sanity_max_mw": 5000.0,
        "areas": {
            "netherlands": {"eic": "10YNL----------L", "tz": ZoneInfo("Europe/Amsterdam")},
            "belgium": {"eic": "10YBE----------2", "tz": ZoneInfo("Europe/Brussels")},
        },
    },
    "de_no2": {
        "border_key": "DE-NO2",
        "db": "interconnections",
        "flow_sanity_max_mw": 1800.0,  # NordLink, capacidad nominal 1.400 MW
        "areas": {
            # NordLink: HVDC Wilster-Tonstad, operativo desde 2021. Es el enlace de
            # Alemania con la hidráulica noruega, la que amortigua sus horas tensas
            # (findings.md #124, hueco nº4).
            "germany": {"eic": "10Y1001A1001A82H", "tz": ZoneInfo("Europe/Berlin")},
            "norway_no2": {"eic": "10YNO-2--------T", "tz": ZoneInfo("Europe/Oslo")},
        },
    },
    "de_se4": {
        "border_key": "DE-SE4",
        "db": "interconnections",
        "flow_sanity_max_mw": 900.0,  # Baltic Cable, capacidad nominal 600 MW
        "areas": {
            # Baltic Cable: HVDC Lübeck-Kruseberg, operativo desde 1994. SE4 es la
            # zona de precio sueca del sur, la misma que en pl_se4.
            "germany": {"eic": "10Y1001A1001A82H", "tz": ZoneInfo("Europe/Berlin")},
            "sweden_se4": {"eic": "10Y1001A1001A47J", "tz": ZoneInfo("Europe/Stockholm")},
        },
    },
}

_session = requests.Session()


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------

def fetch_raw(in_eic: str, out_eic: str, period_start: str, period_end: str, timeout: int = 60, retries: int = 3) -> str:
    """period_start/period_end en formato ENTSO-E: 'AAAAMMDDHHmm' (UTC).
    in_eic = área destino, out_eic = área origen — el valor devuelto es el flujo
    origen->destino (verificado en vivo, ver nota 1 del módulo)."""
    params = {
        "securityToken": require_env("ENTSOE_API_TOKEN"),
        "documentType": "A11",
        "in_Domain": in_eic,
        "out_Domain": out_eic,
        "periodStart": period_start,
        "periodEnd": period_end,
    }
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            resp = _session.get(API_URL, params=params, timeout=timeout)
            resp.raise_for_status()
            return resp.text
        except requests.RequestException as exc:
            last_exc = exc
            time.sleep(2 * attempt)
    raise last_exc  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Diccionario de datos
# ---------------------------------------------------------------------------

FLOW_COLUMNS = {
    "interval_start_utc": ColumnDoc(
        description="Instante UTC de inicio del intervalo de medida del flujo físico.",
        source="timeInterval/start de cada <Period> + offset de (position-1) × resolución, del XML de ENTSO-E (documentType A11).",
        dtype="TIMESTAMP", kind="identifier",
    ),
    "interval_start_local": ColumnDoc(
        description=(
            "Mismo instante en hora local del área exportadora (from_area), NO del importador. "
            "España/Francia/Alemania/Países Bajos comparten huso CET/CEST, pero Portugal usa "
            "WET/WEST (Europe/Lisbon, una hora por detrás) — en la frontera ES-PT esta columna "
            "difiere según qué área sea from_area en cada fila, a diferencia del resto de "
            "fronteras donde es indistinto. No usar esta columna para comparar horas de reloj "
            "entre las dos filas (from/to) del mismo instante en ES-PT."
        ),
        source="Calculado por el pipeline a partir de interval_start_utc y el tz del área exportadora (from_area).",
        dtype="TIMESTAMP", kind="timestamp",
    ),
    "resolution_minutes": ColumnDoc(
        description=(
            "Duración real del intervalo de medida. El corte horario->cuartohorario es propio de "
            "cada frontera (border_key), no una fecha única compartida: ES-FR pasa el 2023-03-26/27, "
            "DE-FR sigue en 60 min en enero de 2023 (más cerca de la reforma paneuropea del "
            "2025-10-01), DE-NL ya está en 15 min desde el inicio del histórico cargado (2023-01). "
            "Ver docs/findings.md — no asumir homogeneidad entre fronteras."
        ),
        source="Elemento <resolution> del <Period> correspondiente del XML de ENTSO-E.",
        dtype="SMALLINT", kind="categorical",
    ),
    "border_key": ColumnDoc(
        description="Identificador de la frontera, códigos de país en orden alfabético ('ES-FR').",
        source="Constante de configuración (BORDERS en el pipeline).",
        dtype="VARCHAR", kind="categorical",
    ),
    "from_area": ColumnDoc(
        description="Área de origen físico del flujo en este intervalo (exportador).",
        source="Constante de configuración — corresponde al out_Domain de la petición a ENTSO-E.",
        dtype="VARCHAR", kind="categorical",
    ),
    "to_area": ColumnDoc(
        description="Área de destino físico del flujo en este intervalo (importador).",
        source="Constante de configuración — corresponde al in_Domain de la petición a ENTSO-E.",
        dtype="VARCHAR", kind="categorical",
    ),
    "flow_mw": ColumnDoc(
        description=(
            "Flujo físico medido en el sentido from_area -> to_area, en MW. Siempre >= 0 por "
            "construcción (cada sentido se pide y guarda como serie independiente, ver nota del "
            "módulo). El flujo neto de la frontera en un instante es flow(A->B) - flow(B->A)."
        ),
        source="Elemento <quantity> de cada <Point>, con forward-fill de posiciones omitidas.",
        dtype="DOUBLE", kind="continuous",
    ),
    "source_chunk": ColumnDoc(
        description="Identificador del bloque de petición de origen (rango pedido a la API), para trazabilidad.",
        source="Parámetro de la petición a la API de ENTSO-E.",
        dtype="VARCHAR", kind="identifier",
    ),
    "ingested_at": ColumnDoc(
        description="Marca temporal UTC del momento en que esta fila se cargó (o recargó) por última vez en la base de datos.",
        source="datetime.utcnow() en el momento de la inserción, dentro de la función load().",
        dtype="TIMESTAMP", kind="timestamp",
    ),
}


def document(con: duckdb.DuckDBPyConnection) -> None:
    catalog.refresh(con, "cross_border_flows", FLOW_COLUMNS)


# ---------------------------------------------------------------------------
# Schema + load
# ---------------------------------------------------------------------------

def ensure_schema(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS cross_border_flows (
            interval_start_utc    TIMESTAMP NOT NULL,
            interval_start_local  TIMESTAMP NOT NULL,
            resolution_minutes    SMALLINT NOT NULL,
            border_key            VARCHAR NOT NULL,
            from_area             VARCHAR NOT NULL,
            to_area                VARCHAR NOT NULL,
            flow_mw                DOUBLE NOT NULL,
            source_chunk            VARCHAR NOT NULL,
            ingested_at              TIMESTAMP NOT NULL,
            PRIMARY KEY (interval_start_utc, from_area, to_area)
        )
        """
    )


# ---------------------------------------------------------------------------
# Orquestación de un rango + registro de auditoría
# ---------------------------------------------------------------------------

def _log_run(con, source, rows_loaded, status, message, started_at) -> None:
    con.execute(
        """
        INSERT INTO ingestion_log
            (source, target_table, delivery_date, rows_expected, rows_loaded, status, message, started_at, finished_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [source, "cross_border_flows", date.today(), None, rows_loaded, status, message, started_at, datetime.utcnow()],
    )


def run_direction_for_range(
    con: duckdb.DuckDBPyConnection, border_key_cfg: str, from_area: str, to_area: str,
    period_start: str, period_end: str, logger,
) -> dict:
    started_at = datetime.utcnow()
    ensure_schema(con)
    border = BORDERS[border_key_cfg]
    from_cfg = border["areas"][from_area]
    to_cfg = border["areas"][to_area]
    border_key = border["border_key"]
    source_chunk = f"{period_start}-{period_end}"
    source = f"entsoe_flow:{border_key}:{from_area}->{to_area}"

    try:
        raw = fetch_raw(to_cfg["eic"], from_cfg["eic"], period_start, period_end)
        grouped = entsoe_xml.parse_periods(raw)
        rows, parse_stats = grouped.get("default", ([], {"n_exact_duplicates": 0, "n_revised": 0}))
    except Exception as exc:
        _log_run(con, source, 0, "ERROR", str(exc), started_at)
        logger.error("%s %s: %s", source, source_chunk, exc)
        return {"status": "ERROR", "message": str(exc)}

    messages: list[str] = []
    status = "OK"
    if parse_stats["n_revised"]:
        status = "WARN"
        messages.append(f"{parse_stats['n_revised']} intervalos revisados (valor distinto en dos TimeSeries)")
    n_negative = sum(1 for r in rows if r["value"] < 0)
    if n_negative:
        status = "WARN"
        messages.append(f"{n_negative} intervalos con flujo NEGATIVO (inesperado — ver nota 1 del módulo)")
    flow_sanity_max_mw = border["flow_sanity_max_mw"]
    n_out_of_band = sum(1 for r in rows if r["value"] > flow_sanity_max_mw)
    if n_out_of_band:
        status = "WARN"
        messages.append(f"{n_out_of_band} intervalos por encima de la banda de plausibilidad ({flow_sanity_max_mw:.0f} MW)")
    if not rows:
        status = "WARN"
        messages.append("sin filas en este bloque")

    now = datetime.utcnow()
    rows_loaded = 0
    if rows:
        tz = from_cfg["tz"]
        df = pd.DataFrame(
            [
                (
                    r["interval_start_utc"],
                    r["interval_start_utc"].replace(tzinfo=timezone.utc).astimezone(tz).replace(tzinfo=None),
                    r["resolution_minutes"], border_key, from_area, to_area, r["value"], source_chunk, now,
                )
                for r in rows
            ],
            columns=["interval_start_utc", "interval_start_local", "resolution_minutes", "border_key",
                     "from_area", "to_area", "flow_mw", "source_chunk", "ingested_at"],
        )
        min_ts, max_ts = df["interval_start_utc"].min(), df["interval_start_utc"].max()
        con.execute("BEGIN TRANSACTION")
        try:
            con.execute(
                "DELETE FROM cross_border_flows WHERE from_area = ? AND to_area = ? "
                "AND interval_start_utc >= ? AND interval_start_utc <= ?",
                [from_area, to_area, min_ts, max_ts],
            )
            con.append("cross_border_flows", df)
            con.execute("COMMIT")
        except Exception:
            con.execute("ROLLBACK")
            raise
        rows_loaded = len(df)

    message = "; ".join(messages) if messages else "sin incidencias"
    _log_run(con, source, rows_loaded, status, message, started_at)
    log_fn = logger.info if status == "OK" else logger.warning
    log_fn("%s %s: %s filas cargadas — %s", source, source_chunk, rows_loaded, message)
    return {"status": status, "from_area": from_area, "to_area": to_area, "rows_loaded": rows_loaded, "message": message}


def run_border_for_range(con: duckdb.DuckDBPyConnection, border_key_cfg: str, period_start: str, period_end: str, logger) -> list[dict]:
    """Ejecuta ambos sentidos de una frontera."""
    a, b = list(BORDERS[border_key_cfg]["areas"])
    return [
        run_direction_for_range(con, border_key_cfg, a, b, period_start, period_end, logger),
        run_direction_for_range(con, border_key_cfg, b, a, period_start, period_end, logger),
    ]


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(__file__).rsplit("etl", 1)[0])
    from etl.common.db import connect, document_ingestion_log
    from etl.common.logging_config import get_logger

    border = sys.argv[1] if len(sys.argv) > 1 else "es_fr"
    log = get_logger("entsoe_cross_border_flows.manual", BORDERS[border]["db"])
    conn = connect(BORDERS[border]["db"])
    results = run_border_for_range(conn, border, "202301010000", "202301080000", log)
    print(results)
    document(conn)
    document_ingestion_log(conn)
    conn.close()
