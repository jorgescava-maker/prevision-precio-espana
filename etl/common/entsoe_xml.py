"""
Parseo XML común para documentos ENTSO-E con forma <TimeSeries>/<Period>/<Point>
(A44 precio day-ahead, A65 carga real, A75 generación real por tecnología...).

Encapsula dos comportamientos verificados en vivo y documentados en docs/findings.md:
- #11: los <Point> cuyo valor no cambia respecto al anterior se OMITEN del XML — hay
  que rellenar hacia adelante (forward-fill), no asumir que faltan datos.
- #12: puede haber más de un <TimeSeries> cubriendo el mismo periodo con contenido
  idéntico (duplicado real de ENTSO-E) — se deduplica quedándose con la última
  ocurrencia, y se distingue de una revisión real (mismo instante, valor distinto).
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from typing import Callable

RESOLUTIONS = {"PT60M": 60, "PT30M": 30, "PT15M": 15}


def strip_ns(elem: ET.Element) -> ET.Element:
    for el in elem.iter():
        if "}" in el.tag:
            el.tag = el.tag.split("}", 1)[1]
    return elem


def parse_periods(
    xml_text: str,
    series_key_fn: Callable[[ET.Element], str] = lambda ts: "default",
) -> dict[str, tuple[list[dict], dict]]:
    """Devuelve {clave_de_serie: (filas, stats)}. `series_key_fn` extrae la clave de
    agrupación de cada <TimeSeries> (p.ej. el <psrType> para generación por tecnología;
    una constante para series de una sola dimensión como carga o precio)."""
    root = strip_ns(ET.fromstring(xml_text))

    if root.tag == "Acknowledgement_MarketDocument":
        reason = root.findtext(".//Reason/text") or "sin detalle"
        raise ValueError(f"ENTSO-E devolvió un Acknowledgement (sin datos): {reason}")

    grouped: dict[str, dict[datetime, dict]] = {}
    stats: dict[str, dict[str, int]] = {}

    for ts in root.findall("TimeSeries"):
        key = series_key_fn(ts)
        by_ts = grouped.setdefault(key, {})
        s = stats.setdefault(key, {"n_exact_duplicates": 0, "n_revised": 0})

        for period in ts.findall("Period"):
            start_str = period.findtext("timeInterval/start")
            end_str = period.findtext("timeInterval/end")
            resolution_str = period.findtext("resolution")
            resolution_minutes = RESOLUTIONS.get(resolution_str)
            if resolution_minutes is None:
                raise ValueError(f"resolución no soportada: {resolution_str}")

            start_dt = datetime.strptime(start_str, "%Y-%m-%dT%H:%MZ").replace(tzinfo=timezone.utc)
            end_dt = datetime.strptime(end_str, "%Y-%m-%dT%H:%MZ").replace(tzinfo=timezone.utc)
            n_positions = round((end_dt - start_dt).total_seconds() / 60 / resolution_minutes)

            points: dict[int, float] = {}
            for pt in period.findall("Point"):
                pos = int(pt.findtext("position"))
                value_el = pt.find("quantity")
                if value_el is None:
                    value_el = pt.find("price.amount")
                points[pos] = float(value_el.text)

            last_val = None
            for pos in range(1, n_positions + 1):
                if pos in points:
                    last_val = points[pos]
                if last_val is None:
                    continue
                interval_start = (start_dt + timedelta(minutes=(pos - 1) * resolution_minutes)).replace(tzinfo=None)

                existing = by_ts.get(interval_start)
                if existing is not None:
                    if existing["value"] == last_val:
                        s["n_exact_duplicates"] += 1
                    else:
                        s["n_revised"] += 1
                by_ts[interval_start] = {
                    "interval_start_utc": interval_start,
                    "resolution_minutes": resolution_minutes,
                    "value": last_val,
                }

    result = {}
    for key, by_ts in grouped.items():
        rows = [by_ts[k] for k in sorted(by_ts)]
        result[key] = (rows, stats[key])
    return result
