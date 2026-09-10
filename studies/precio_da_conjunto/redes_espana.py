"""
Fase 27 (2026-09-02) — TERCER miembro del ensemble de España: un ENSEMBLE DE
REDES NEURONALES, el siguiente paso anotado en findings.md #110 tras agotar la
vía de los árboles en la campaña Fable (#109, España 10,53).

Diseño (por qué así, ver #110):
  * Un ensemble de K redes con distinta inicialización, NO una red suelta: en
    el benchmark de referencia de EPF una red individual pierde contra un
    lineal bien hecho, y el promedio de 5-10 copias gana a todo. La
    inestabilidad de las redes es justo lo que hace que sus errores se
    cancelen. (Contraste medido aquí: promediar 5 semillas de LightGBM no
    movió el número —C3 de #109—, porque los GBDT ya son estables.)
  * Miembro NUEVO, no sustituto: LightGBM corta rectangularmente y no
    extrapola; una red aprende funciones continuas. Se equivocan distinto, que
    es la condición para que un ensemble aporte.
  * Mismo walk-forward mensual causal (L8) y mismas features que el campeón,
    para que la predicción sea combinable periodo a periodo con `pred_pooled` y
    `pred_por_hora` de `ensemble_pooled_hora.py`.

Diferencia inevitable con LightGBM: las redes NO manejan NaN ni escalas
dispares. Todo el preprocesado (mediana + estandarizado, features y objetivo)
se ajusta SOLO con el train de cada mes y se aplica al test — nunca al revés.

Uso:
    .venv\Scripts\python.exe -m studies.precio_da_conjunto.redes_espana [--probe N] [--seeds K] [--jobs J]

Corre siempre las dos variantes (N1 pooled ES+FR y N2 solo España, ~20 min en
total con 5 semillas): la comparación entre ambas es justamente el resultado.

Escribe output/predicciones_redes.parquet (una columna por variante) y
output/redes_espana_summary.txt.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from etl.common import logging_config  # noqa: F401
from studies.precio_da_mejor_modelo.model import INITIAL_TRAIN_MONTHS, report
from studies.precio_da_conjunto.model import COMMON_FEATURES
from studies.precio_da_conjunto.model_full import (
    SPAIN_ONLY, FRANCE_ONLY, load_spain_full, load_france_full,
)

OUTPUT_DIR = Path(__file__).resolve().parent / "output"

# Configuración de partida: MLP de 2 capas, la alternativa "más simple y
# también validada en el benchmark" de #110. Early stopping con una fracción
# del PROPIO train (nunca del test) para no fijar el nº de épocas a mano.
MLP_KWARGS = dict(
    hidden_layer_sizes=(128, 64),
    activation="relu",
    solver="adam",
    alpha=1e-4,
    batch_size=256,
    learning_rate_init=1e-3,
    max_iter=200,
    early_stopping=True,
    n_iter_no_change=10,
    validation_fraction=0.1,
    shuffle=True,
)


def _fit_one(X_tr: pd.DataFrame, y_scaled: np.ndarray, X_te: pd.DataFrame, seed: int,
             mlp_kwargs: dict) -> np.ndarray:
    # `mlp_kwargs` viaja como ARGUMENTO, no como global del módulo: con
    # n_jobs>1 los ajustes corren en procesos hijos que reimportan este
    # módulo, así que cualquier configuración fijada mutando MLP_KWARGS desde
    # fuera (p.ej. una ronda de variantes) se perdería en silencio.
    pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("mlp", MLPRegressor(random_state=seed, **mlp_kwargs)),
    ])
    pipe.fit(X_tr, y_scaled)
    return pipe.predict(X_te)


def _fit_predict_ensemble(X_tr: pd.DataFrame, y_tr: np.ndarray, X_te: pd.DataFrame,
                          seeds: list[int], n_jobs: int = 1,
                          mlp_kwargs: dict | None = None) -> np.ndarray:
    """Promedio de K redes idénticas con distinta semilla. El objetivo también
    se estandariza con la media/desviación del train (adam converge mal con un
    objetivo en EUR/MWh sin escalar) y se deshace la transformación al predecir.
    Las K redes son independientes entre sí: se ajustan en paralelo (procesos,
    con los hilos de BLAS limitados dentro de cada uno para no sobresuscribir)."""
    mu, sigma = float(np.mean(y_tr)), float(np.std(y_tr))
    if sigma <= 0:
        sigma = 1.0
    y_scaled = (y_tr - mu) / sigma
    kw = dict(MLP_KWARGS if mlp_kwargs is None else mlp_kwargs)

    if n_jobs > 1 and len(seeds) > 1:
        preds = Parallel(n_jobs=min(n_jobs, len(seeds)))(
            delayed(_fit_one)(X_tr, y_scaled, X_te, s, kw) for s in seeds
        )
    else:
        preds = [_fit_one(X_tr, y_scaled, X_te, s, kw) for s in seeds]
    return np.mean(preds, axis=0) * sigma + mu


def walk_forward_redes(df: pd.DataFrame, features: list[str], required: list[str],
                       target_col: str, tag: str, seeds: list[int], log,
                       max_months: int | None = None, n_jobs: int = 1,
                       mlp_kwargs: dict | None = None) -> pd.Series:
    d = df.dropna(subset=required + [target_col]).copy()
    d["year_month"] = d["period_start_utc"].dt.to_period("M")
    months = sorted(d["year_month"].unique())
    last = len(months) if max_months is None else min(len(months), INITIAL_TRAIN_MONTHS + max_months)

    preds = pd.Series(np.nan, index=df.index)
    t0 = time.time()
    for i in range(INITIAL_TRAIN_MONTHS, last):
        train = d[d["year_month"].isin(months[:i])]
        test = d[d["year_month"] == months[i]]
        if test.empty or train.empty:
            continue
        t1 = time.time()
        preds.loc[test.index] = _fit_predict_ensemble(
            train[features], train[target_col].to_numpy(dtype=float), test[features], seeds,
            n_jobs, mlp_kwargs,
        )
        log(f"    {tag} {months[i]}: train={len(train):,} test={len(test):,} "
            f"({time.time() - t1:.0f}s, acumulado {(time.time() - t0)/60:.1f} min)")
    log(f"  {tag}: walk-forward completado ({last - INITIAL_TRAIN_MONTHS} meses de test, "
        f"{(time.time() - t0)/60:.1f} min)")
    return preds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", type=int, default=None,
                    help="limita el walk-forward a N meses de test (sondeo de coste)")
    ap.add_argument("--seeds", type=int, default=5, help="tamaño del ensemble de redes")
    ap.add_argument("--jobs", type=int, default=5, help="redes ajustadas en paralelo (procesos)")
    args = ap.parse_args()
    seeds = list(range(42, 42 + args.seeds))

    lines: list[str] = []

    def log(s: str = "") -> None:
        print(s, flush=True)
        lines.append(s)

    log("=" * 72)
    log(f"Fase 27 — ensemble de {len(seeds)} redes (MLP 2 capas) como tercer miembro")
    log("=" * 72)

    es = load_spain_full()
    fr = load_france_full()
    pooled = pd.concat([es, fr], ignore_index=True)
    pooled["is_spain"] = (pooled["country"] == "ES").astype(int)
    all_features = list(dict.fromkeys(COMMON_FEATURES + SPAIN_ONLY + FRANCE_ONLY + ["is_spain"]))
    log(f"Pooled ES+FR: {len(pooled):,} periodos, {len(all_features)} features. "
        f"Semillas: {seeds}")

    log("\n-- (N1) redes sobre el pooled ES+FR, features completas (imputadas) --")
    pooled["pred_red_pooled"] = walk_forward_redes(
        pooled, all_features, COMMON_FEATURES, "price_real", "N1 pooled", seeds, log,
        max_months=args.probe, n_jobs=args.jobs,
    )

    log("\n-- (N2) redes solo con las filas de España --")
    es_rows = pooled[pooled["country"] == "ES"].copy()
    feats_es = [f for f in all_features if f not in FRANCE_ONLY + ["is_spain"]]
    es_rows["pred_red_es"] = walk_forward_redes(
        es_rows, feats_es, COMMON_FEATURES, "price_real", "N2 solo ES", seeds, log,
        max_months=args.probe, n_jobs=args.jobs,
    )
    pooled = pooled.merge(
        es_rows[["period_start_utc", "country", "pred_red_es"]],
        on=["period_start_utc", "country"], how="left",
    )

    log("\n-- Resultados (España, ventana propia de las redes) --")
    sub = pooled[pooled["country"] == "ES"]
    mask = sub["pred_red_pooled"].notna() & sub["pred_red_es"].notna()
    report(sub, "pred_red_pooled", "(N1) redes pooled ES+FR", log, mask=mask)
    report(sub, "pred_red_es", "(N2) redes solo España", log, mask=mask)

    log("\n-- Resultados (Francia, referencia) --")
    subfr = pooled[pooled["country"] == "FR"]
    report(subfr, "pred_red_pooled", "(N1) redes pooled ES+FR", log)

    cols = ["period_start_utc", "country", "price_real", "pred_red_pooled", "pred_red_es"]
    pooled[cols].to_parquet(OUTPUT_DIR / "predicciones_redes.parquet", index=False)
    (OUTPUT_DIR / "redes_espana_summary.txt").write_text("\n".join(lines), encoding="utf-8")
    log("\nGuardado.")


if __name__ == "__main__":
    main()
