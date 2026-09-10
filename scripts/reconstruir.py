"""
Reconstruye la cadena completa del modelo de España a partir de las bases
DuckDB de `data/`, en el orden en que cada paso necesita la salida del anterior.

Dos tramos:

  datasets     variables D-1 de España y de Francia (unos minutos). Es lo que
               hay que rehacer cada día antes de predecir.
  walkforward  el backtest mensual: modelos agrupado y por hora, redes, pesos
               de la mezcla, calibración de la cola y filtro de precio<=0. Deja
               la predicción de referencia en
               studies/precio_da_conjunto/output/predicciones_fase27b.parquet
               (columna `pred_final`) y los pesos que usa la predicción diaria.

Uso:
    python -m scripts.reconstruir                 # todo
    python -m scripts.reconstruir datasets
    python -m scripts.reconstruir walkforward
    python -m scripts.reconstruir todo --desde 5  # retomar desde el paso 5
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

from etl.common import logging_config  # noqa: F401 - fuerza stdout a UTF-8

ROOT =Path(__file__).resolve().parents[1]
LOGS = ROOT / "logs"

DATASETS = [
    ("valor del agua — serie semanal (C3)", "studies.c3_valor_agua.build_dataset"),
    ("motor de despacho — dataset", "studies.merit_order.build_dataset"),
    ("motor de despacho — hidráulica como coste + indisponibilidad", "studies.merit_order.model"),
    ("variables D-1 de España", "studies.precio_da_mejor_modelo.build_dataset"),
    ("motor de despacho de Francia", "studies.merit_order_francia.build_dataset"),
    ("variables D-1 de Francia", "studies.precio_da_francia.build_dataset"),
]
# El walk-forward se hace con el valor del agua CAUSAL (findings.md #146): un
# dataset por mes de corte, con la superficie del agua estimada solo con las
# semanas ya publicadas, y cada mes se entrena y predice con el suyo. Los
# scripts `ensemble_pooled_hora`, `redes_espana` y `ensemble_tres_miembros`
# siguen en el repositorio, pero usan una superficie estimada con toda la
# historia y su backtest no vale como referencia.
WALKFORWARD = [
    ("walk-forward causal: un dataset por mes de corte", "studies.precio_da_conjunto.agua_causal datasets --rehacer"),
    ("walk-forward causal: agrupado, por hora, redes, filtro, mezcla y calibración",
     "studies.precio_da_conjunto.agua_causal walkforward"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("tramo", nargs="?", default="todo", choices=["todo", "datasets", "walkforward"])
    ap.add_argument("--desde", type=int, default=1, help="número de paso (1-based) desde el que empezar")
    args = ap.parse_args()

    pasos = {"todo": DATASETS + WALKFORWARD, "datasets": DATASETS, "walkforward": WALKFORWARD}[args.tramo]
    LOGS.mkdir(exist_ok=True)
    # Algunos pasos escriben cachés en la carpeta output/ de OTRO estudio (el
    # motor de Francia guarda la serie semanal del agua en precio_da_francia/),
    # que en un clon recién hecho todavía no existe.
    for estudio in (ROOT / "studies").iterdir():
        if estudio.is_dir():
            (estudio / "output").mkdir(exist_ok=True)
    total = time.perf_counter()
    for i, (nombre, modulo) in enumerate(pasos, start=1):
        if i < args.desde:
            continue
        print(f"[{i}/{len(pasos)}] {nombre} ({modulo}) ...", flush=True)
        t0 = time.perf_counter()
        log = LOGS / f"reconstruir_{re.sub(r'[^A-Za-z0-9]+', '_', modulo.split('.', 1)[1]).strip('_')}.log"
        with open(log, "w", encoding="utf-8") as f:
            r = subprocess.run([sys.executable, "-m", *modulo.split()], cwd=ROOT, stdout=f, stderr=subprocess.STDOUT,
                               env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"})
        dt = time.perf_counter() - t0
        if r.returncode != 0:
            print(f"      FALLÓ tras {dt:.0f}s — ver {log.relative_to(ROOT)}", flush=True)
            return r.returncode
        print(f"      ok en {dt:.0f}s", flush=True)
    print(f"Terminado en {(time.perf_counter() - total) / 60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
