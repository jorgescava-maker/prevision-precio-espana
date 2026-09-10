"""
Ensemble pooled+por-hora sobre el mejor modelo conjunto de España (pooled
ES+FR con features completas, NaN nativo — ver model_full.py, 10,99 EUR/MWh
de MAE, el mejor número de España en toda esta línea). Mismo patrón causal
ya validado en `precio_da_mejor_modelo` (Fase 17) y `precio_da_francia`
(Fase 4): peso del promedio elegido mes a mes usando solo el histórico de
test ya observado.

Uso:
    .venv\\Scripts\\python.exe -m studies.precio_da_conjunto.ensemble_pooled_hora
"""

from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from etl.common import logging_config  # noqa: F401
from studies.precio_da_mejor_modelo.model import LGB_PARAMS, INITIAL_TRAIN_MONTHS, report
from studies.precio_da_conjunto.model import COMMON_FEATURES
from studies.precio_da_conjunto.model_full import SPAIN_ONLY, FRANCE_ONLY, load_spain_full, load_france_full

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
WARMUP_MONTHS = 6
PESOS_CANDIDATOS = np.arange(0.0, 1.01, 0.1)
MIN_ROWS_PER_GROUP = 30


def walk_forward_by_group_nan_ok(df: pd.DataFrame, features: list[str], required: list[str],
                                  target_col: str, group_col: str, tag: str, log) -> pd.Series:
    d = df.dropna(subset=required + [target_col, group_col]).copy()
    d["year_month"] = d["period_start_utc"].dt.to_period("M")
    months = sorted(d["year_month"].unique())
    preds = pd.Series(np.nan, index=df.index)
    n_skipped = 0
    for i in range(INITIAL_TRAIN_MONTHS, len(months)):
        train_all = d[d["year_month"].isin(months[:i])]
        test_all = d[d["year_month"] == months[i]]
        if test_all.empty or train_all.empty:
            continue
        for g, test_g in test_all.groupby(group_col):
            train_g = train_all[train_all[group_col] == g]
            if len(train_g) < MIN_ROWS_PER_GROUP:
                n_skipped += len(test_g)
                continue
            reg = lgb.LGBMRegressor(**LGB_PARAMS)
            reg.fit(train_g[features], train_g[target_col])
            preds.loc[test_g.index] = reg.predict(test_g[features])
    log(f"  {tag}: walk-forward por grupo completado, {n_skipped:,} filas sin predicción por falta de histórico")
    return preds


def walk_forward_nan_ok(df: pd.DataFrame, features: list[str], required: list[str], target_col: str, tag: str, log) -> pd.Series:
    d = df.dropna(subset=required + [target_col]).copy()
    d["year_month"] = d["period_start_utc"].dt.to_period("M")
    months = sorted(d["year_month"].unique())
    preds = pd.Series(np.nan, index=df.index)
    for i in range(INITIAL_TRAIN_MONTHS, len(months)):
        train = d[d["year_month"].isin(months[:i])]
        test = d[d["year_month"] == months[i]]
        if test.empty or train.empty:
            continue
        reg = lgb.LGBMRegressor(**LGB_PARAMS)
        reg.fit(train[features], train[target_col])
        preds.loc[test.index] = reg.predict(test[features])
    log(f"  {tag}: walk-forward completado")
    return preds


def main() -> None:
    lines: list[str] = []

    def log(s: str = "") -> None:
        print(s)
        lines.append(s)

    log("=" * 72)
    log("Ensemble pooled+por-hora sobre el modelo conjunto ES+FR (features completas)")
    log("=" * 72)

    es = load_spain_full()
    fr = load_france_full()
    pooled = pd.concat([es, fr], ignore_index=True)
    pooled["is_spain"] = (pooled["country"] == "ES").astype(int)
    # dict.fromkeys: SPAIN_ONLY y FRANCE_ONLY comparten el nombre
    # `price_simulado_final` desde 2026-08-31 (mismo motivo que model_full.py).
    all_features = list(dict.fromkeys(COMMON_FEATURES + SPAIN_ONLY + FRANCE_ONLY + ["is_spain"]))
    feats_hora = [f for f in all_features if f not in ("hour_of_day", "period_of_day")]

    log("\n-- (Z2) pooled, arquitectura agrupada (ya conocida) --")
    pooled["pred_pooled"] = walk_forward_nan_ok(pooled, all_features, COMMON_FEATURES, "price_real", "pooled", log)

    log("\n-- (Z2-h) pooled, por hora del día --")
    pooled["pred_por_hora"] = walk_forward_by_group_nan_ok(
        pooled, feats_hora, COMMON_FEATURES, "price_real", "hour_of_day", "pooled por hora", log
    )

    mask = pooled["pred_pooled"].notna() & pooled["pred_por_hora"].notna()
    d = pooled[mask].sort_values("period_start_utc").copy()
    d["year_month"] = d["period_start_utc"].dt.to_period("M")
    months = sorted(d["year_month"].unique())

    res_pooled = d["pred_pooled"] - d["price_real"]
    res_hora = d["pred_por_hora"] - d["price_real"]
    log(f"\nCorrelación entre residuos pooled/por-hora (todo el conjunto ES+FR): {res_pooled.corr(res_hora):.4f}")

    d["pred_ensemble"] = np.nan
    d["w_usado"] = np.nan
    for i, m in enumerate(months):
        if i < WARMUP_MONTHS:
            continue
        hist = d[d["year_month"].isin(months[:i])]
        maes = [
            (hist["price_real"] - (w * hist["pred_pooled"] + (1 - w) * hist["pred_por_hora"])).abs().mean()
            for w in PESOS_CANDIDATOS
        ]
        w_best = PESOS_CANDIDATOS[int(np.argmin(maes))]
        cur = d[d["year_month"] == m]
        d.loc[cur.index, "pred_ensemble"] = w_best * cur["pred_pooled"] + (1 - w_best) * cur["pred_por_hora"]
        d.loc[cur.index, "w_usado"] = w_best

    valid = d.dropna(subset=["pred_ensemble"]).copy()
    log(f"\nPeriodo evaluable tras calentamiento de {WARMUP_MONTHS} meses: {len(valid):,} periodos")

    for pais in ("ES", "FR"):
        sub = valid[valid["country"] == pais]
        log(f"\n-- {pais}: comparación --")
        report(sub, "pred_pooled", f"  pooled (agrupado)", log)
        report(sub, "pred_por_hora", f"  pooled (por hora)", log)
        report(sub, "pred_ensemble", f"  ensemble (peso causal)", log)

    valid.to_parquet(OUTPUT_DIR / "predicciones_ensemble_conjunto.parquet", index=False)
    (OUTPUT_DIR / "ensemble_conjunto_summary.txt").write_text("\n".join(lines), encoding="utf-8")
    log("\nGuardado.")


if __name__ == "__main__":
    main()
