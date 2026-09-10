"""
Construye desde cero las bases DuckDB de `data/` que necesita el modelo de
España: lanza cada `etl/backfill_*.py` con las áreas, fronteras y fechas que
lee la cadena.

Tarda horas, casi todo por ENTSO-E (limita el tamaño de cada petición y a
veces va lento o se cae). Se puede interrumpir y relanzar: cada backfill borra
y reescribe su propio rango, así que repetir un paso no duplica datos.
Después, `etl/daily_update_espana.py` las mantiene al día.

Necesita `.env` con los dos tokens (ver `.env.example`).

Uso:
    python -m scripts.construir_bases              # todo, desde 2023
    python -m scripts.construir_bases --mostrar    # solo imprime los comandos
    python -m scripts.construir_bases --desde 12   # retomar desde el paso 12
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

from etl.common import logging_config  # noqa: F401 - fuerza stdout a UTF-8

ROOT = Path(__file__).resolve().parents[1]
INICIO = 2023

# Qué lee la cadena de cada fuente (si se añade una variable que use otra
# frontera u otra área, hay que añadirla aquí):
#   flujos   ES-FR y ES-PT (import neto de España, motor de despacho) y las
#            cinco fronteras de Francia (motor de despacho francés)
#   NTC      solo ES-FR (variable `ntc_es_fr_mw`)
#   clima    solo Francia (grados-día del dataset francés)
FRONTERAS_FLUJO = ["es_fr", "es_pt", "de_fr", "fr_be", "fr_ch", "fr_it"]
FRONTERAS_NTC = ["es_fr"]


def pasos(hoy: date) -> list[list[str]]:
    a, b = str(INICIO), str(hoy.year)
    d0, d1 = f"{INICIO}-01-01", hoy.isoformat()
    # Las materias primas empiezan un mes antes: el modelo usa el cierre de
    # dos días antes de la entrega y arrastra hasta 5 días el último válido.
    m0 = f"{INICIO - 1}-12-01"
    p = [
        ["etl.backfill_omie_spot", "--start", d0, "--end", d1],
        # OMIE solo publica la curva agregada de oferta desde el cambio a 15 min.
        ["etl.backfill_omie_supply_curve", "--start", "2025-10-01", "--end", d1],
        ["etl.backfill_esios_load_generation", "--start", a, "--end", b],
        ["etl.backfill_esios_forecast", "--start", a, "--end", b],
        ["etl.backfill_smard_day_ahead", "--start", d0, "--end", d1],
        ["etl.backfill_entsoe_day_ahead", "--area", "france", "--start", a, "--end", b],
        ["etl.backfill_entsoe_load_generation", "--area", "france", "--start", a, "--end", b],
        ["etl.backfill_entsoe_forecast", "--area", "france", "--start", a, "--end", b],
        ["etl.backfill_entsoe_installed_capacity", "--area", "spain", "--start", a, "--end", b],
        ["etl.backfill_entsoe_installed_capacity", "--area", "france", "--start", a, "--end", b],
        ["etl.backfill_entsoe_generation_units", "--area", "spain", "--start", a, "--end", b],
        ["etl.backfill_entsoe_generation_outages_spain", "--start", a, "--end", b],
        ["etl.backfill_entsoe_nuclear_outages", "--start", a, "--end", b],
        ["etl.backfill_entsoe_thermal_outages", "--start", a, "--end", b],
        ["etl.backfill_entsoe_hydro_reservoir", "--area", "spain", "--start", a, "--end", b],
        ["etl.backfill_entsoe_hydro_reservoir", "--area", "france", "--start", a, "--end", b],
    ]
    p += [["etl.backfill_entsoe_cross_border_flows", "--border", f, "--start", a, "--end", b] for f in FRONTERAS_FLUJO]
    p += [["etl.backfill_entsoe_ntc", "--border", f, "--start", a, "--end", b] for f in FRONTERAS_NTC]
    p += [
        ["etl.backfill_commodity_prices", "--start", m0, "--end", d1],
        ["etl.backfill_fx_rates", "--start", m0, "--end", d1],
        ["etl.backfill_weather_era5", "--location", "france", "--start", a, "--end", b],
        ["etl.backfill_holiday_calendar"],
        # Los grados-día se derivan del clima: tiene que ir después.
        ["etl.backfill_degree_days"],
    ]
    return p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mostrar", action="store_true", help="imprime los comandos sin ejecutarlos")
    ap.add_argument("--desde", type=int, default=1, help="número de paso (1-based) desde el que empezar")
    args = ap.parse_args()

    lista = pasos(date.today())
    (ROOT / "data").mkdir(exist_ok=True)
    fallos = []
    for i, cmd in enumerate(lista, start=1):
        if i < args.desde:
            continue
        linea = "python -m " + " ".join(cmd)
        if args.mostrar:
            print(f"{i:2d}. {linea}")
            continue
        print(f"[{i}/{len(lista)}] {linea}", flush=True)
        t0 = time.perf_counter()
        r = subprocess.run([sys.executable, "-m", *cmd], cwd=ROOT)
        print(f"      {'ok' if r.returncode == 0 else 'CON INCIDENCIAS'} en {time.perf_counter() - t0:.0f}s", flush=True)
        if r.returncode != 0:
            fallos.append(i)
    if fallos:
        print(f"\nPasos con incidencias: {fallos}. Revisa logs/ y relanza con --desde. "
              "Algunas son esperables (p. ej. ENTSO-E aún no publica la capacidad del año siguiente).")
    return 1 if fallos else 0


if __name__ == "__main__":
    raise SystemExit(main())
