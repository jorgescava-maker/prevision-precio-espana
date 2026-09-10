"""
Ciclo diario: actualiza los datos, reconstruye las variables, predice el día
siguiente, lo guarda en `previsiones/diarias/`, evalúa lo ya publicado y, con
`--publicar`, hace commit y push.

Pensado para lanzarse cada mañana ANTES del cierre de la subasta (12:00, hora
peninsular). El commit en GitHub es la prueba de que la previsión se hizo sin
conocer el precio.

Uso:
    python -m scripts.prevision_diaria                 # predice mañana, no publica
    python -m scripts.prevision_diaria --publicar
    python -m scripts.prevision_diaria --sin-etl 2026-09-11
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from etl.common import logging_config  # noqa: F401 - fuerza stdout a UTF-8

ROOT =Path(__file__).resolve().parents[1]
SALIDA_LIVE = ROOT / "studies" / "prediccion_live" / "output"
DIARIAS = ROOT / "previsiones" / "diarias"
TZ = "Europe/Madrid"


def _paso(nombre: str, args: list[str]) -> None:
    print(f"== {nombre} ...", flush=True)
    t0 = time.perf_counter()
    r = subprocess.run([sys.executable, "-m", *args], cwd=ROOT)
    if r.returncode != 0:
        raise SystemExit(f"Falló: {nombre} (código {r.returncode})")
    print(f"   ok en {time.perf_counter() - t0:.0f}s", flush=True)


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dia", nargs="?", help="día de entrega (YYYY-MM-DD); por defecto, mañana")
    ap.add_argument("--sin-etl", action="store_true", help="no actualizar las bases antes de predecir")
    ap.add_argument("--publicar", action="store_true", help="commit + push al terminar")
    args = ap.parse_args()
    dia = date.fromisoformat(args.dia) if args.dia else date.today() + timedelta(days=1)
    t0 = time.perf_counter()

    if not args.sin_etl:
        # Solo lo que cambia de un día a otro: el ETL completo tarda ~15 min por
        # ENTSO-E y no cabe antes del cierre; va al final, después de publicar.
        # Un fallo aislado de una fuente no debe impedir predecir: el ETL deja
        # la tabla como estaba y el modelo trabaja con el último dato conocido.
        print("== actualización rápida de datos ...", flush=True)
        subprocess.run([sys.executable, "-m", "etl.daily_update_espana", "--rapido"], cwd=ROOT)
    _paso("variables D-1", ["scripts.reconstruir", "datasets"])
    _paso(f"predicción de {dia}", ["studies.prediccion_live.predict_manana_espana", str(dia)])

    p = pd.read_parquet(SALIDA_LIVE / f"prediccion_espana_{dia}.parquet").sort_values("period_start_utc")
    emitida = datetime.now(timezone.utc).replace(microsecond=0, tzinfo=None)
    out = pd.DataFrame({
        "dia_entrega": str(dia),
        "inicio_utc": p["period_start_utc"],
        "inicio_local": p["period_start_utc"].dt.tz_localize("UTC").dt.tz_convert(TZ).dt.strftime("%Y-%m-%d %H:%M"),
        "precio_previsto": p["pred_final"].round(2),
        "prob_precio_cero": p["proba_precio_cero"].round(3),
        "emitida_utc": emitida,
    })
    DIARIAS.mkdir(parents=True, exist_ok=True)
    destino = DIARIAS / f"{dia}.csv"
    out.to_csv(destino, index=False)
    print(f"== {len(out)} periodos guardados en {destino.relative_to(ROOT)} "
          f"(media {out['precio_previsto'].mean():.2f} EUR/MWh)")

    _paso("evaluación de lo publicado", ["scripts.evaluar"])
    print(f"== ciclo completo en {(time.perf_counter() - t0) / 60:.1f} min")

    if args.publicar:
        _git("add", "previsiones/diarias", "resultados")
        r = _git("commit", "-m", f"Previsión para el {dia} (emitida {emitida:%Y-%m-%d %H:%M} UTC)")
        if r.returncode != 0 and "nothing to commit" not in r.stdout:
            raise SystemExit(f"git commit falló:\n{r.stdout}{r.stderr}")
        r = _git("push")
        if r.returncode != 0:
            raise SystemExit(f"git push falló:\n{r.stderr}")
        print("== publicado")

    if not args.sin_etl:
        # Ya publicado: ahora sin prisa, el resto de fuentes para mañana.
        print("== actualización completa de datos (para el próximo ciclo) ...", flush=True)
        subprocess.run([sys.executable, "-m", "etl.daily_update_espana"], cwd=ROOT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
