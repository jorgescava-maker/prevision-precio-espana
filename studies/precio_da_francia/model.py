"""
Fase 1 de "el mejor modelo posible de precio DA" para Francia — ver
DESIGN.md. Desde la Fase 10 (2026-08-31) SÍ hay un motor mecanicista propio
que comparar (`studies/merit_order_francia/`, correlación 0,708 en
solitario), reintegrado como feature ancla `price_simulado_final` (L6):
  (a) baseline naive — `price_lag_24h` solo (misma disciplina A2/D1: probar
      lo simple antes de construir algo complejo).
  (a') el análogo solo — segunda referencia barata, ya sabíamos en España
      que suele acercarse mucho al modelo completo por sí solo.
  (a''') el motor de despacho solo.
  (b) ML flexible (LightGBM) sobre todas las features D-1-seguras,
      incluido el motor.

Uso:
    .venv\\Scripts\\python.exe -m studies.precio_da_francia.model

Requiere haber corrido antes build_dataset.py.
"""

from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from etl.common import logging_config  # noqa: F401
from studies.precio_da_mejor_modelo.model import walk_forward, report, INITIAL_TRAIN_MONTHS, LGB_PARAMS

OUTPUT_DIR = Path(__file__).resolve().parent / "output"

# findings.md #103 (2026-09-01): `coal_eur_mwh_th` sin actualizar desde
# 2025-12-26 — excluida de producción, mismo motivo que en
# precio_da_mejor_modelo/model.py (recupera 2026 completo en la evaluación).
# HDD/CDD (findings.md #106, 2026-09-01): grados-día reales de ayer (lag 1
# día, D-1-seguro), nunca probados antes en Francia — correlación univariada
# comprobada en vivo: HDD +0,36 con el precio (Francia tiene la mayor
# dependencia de calefacción eléctrica de los 4 mercados).
FEATURES_BASE = [
    "forecast_demanda_mw", "forecast_eolica_mw", "forecast_solar_mw",
    "ttf_eur_mwh", "eua_eur_t",
    "nuclear_capacity_avail_mw", "gas_capacity_avail_mw", "coal_capacity_avail_mw",
    "embalse_gwh",
    "hdd_lag1", "cdd_lag1",
    "price_lag_24h_de", "price_lag_168h_de",
    "analog_price_mean",
    "price_lag_24h", "price_lag_48h", "price_lag_168h", "price_ema_7d", "price_ema_28d",
    "price_boundary_prev_day",
    "hour_of_day", "period_in_hour", "period_of_day", "day_of_week", "month",
    "is_holiday", "is_weekend",
    "price_simulado_final",
]
# Fase 10 (2026-08-31): `price_simulado_final` (motor de despacho propio,
# `merit_order_francia`) reintegrado como ancla mecanicista L6 — ADOPTADO,
# mejora real y creciente con cada mejora del motor (13,62->13,42 en Fase
# 10; 13,62->13,37 en Fase 10e, tras el ajuste de curtailment nuclear).
# FEATURES_SIN_MOTOR sirve de referencia "antes" para comparación honesta
# (L9), no es candidato vivo.
FEATURES_SIN_MOTOR = [f for f in FEATURES_BASE if f != "price_simulado_final"]

# Fase 9/10/10d/10e (2026-08-31/2026-09-01): valor del agua propio de
# Francia (`water_value.py`) probado CUATRO veces según iba mejorando el
# motor — sin motor (Fase 9, plano), con motor v1/SMA (Fase 10, peor),
# con motor v2/EWM (Fase 10d, ligeramente mejor), con motor v3/curtailment
# nuclear (Fase 10e, peor otra vez: 0,907/13,43 vs. 0,908/13,37 sin ella).
# Patrón consistente: cuanto más preciso es el motor en capturar el
# mecanismo real (nuclear como curtailment condicionado a la sobra, no una
# climatología ciega), menos aporta el agua por separado — su información
# ya queda mayormente absorbida en `price_simulado_final`. Se mantiene
# como columna disponible por trazabilidad, NO en producción (motivo
# opuesto al de Fase 10d, mismo resultado neto: sin water_value).
FEATURES_FULL = FEATURES_BASE


def main() -> None:
    lines: list[str] = []

    def log(s: str = "") -> None:
        print(s)
        lines.append(s)

    log("=" * 72)
    log("Francia — Fase 1: baseline naive vs. análogo solo vs. ML flexible")
    log("=" * 72)

    df = pd.read_parquet(OUTPUT_DIR / "dataset.parquet")
    df["is_holiday"] = df["is_holiday"].astype(int)
    df["is_weekend"] = df["is_weekend"].astype(int)

    log("\n-- (b) ML flexible SIN motor (baseline pre-Fase 10) --")
    df["pred_sin_motor"] = walk_forward(df, FEATURES_SIN_MOTOR, "price_real", "ML flexible sin motor", log)

    log("\n-- (b2) ML flexible CON motor (Fase 0d, curtailment nuclear) — producción actual --")
    df["pred_ml_flexible"] = walk_forward(df, FEATURES_FULL, "price_real", "ML flexible", log)

    log("\n-- (b3) ML flexible + motor + water_value (reprobado por 4ª vez, Fase 10e) --")
    features_con_wv = FEATURES_FULL + ["water_value_eur_mwh", "analog_price_mean_wv"]
    df["pred_con_water_value"] = walk_forward(df, features_con_wv, "price_real", "ML flexible +water_value", log)

    test_mask = df["pred_sin_motor"].notna() & df["pred_ml_flexible"].notna() & df["pred_con_water_value"].notna()
    log(f"\n-- Comparación en el MISMO periodo de test ({test_mask.sum():,} periodos, L9) --")
    results = []
    results.append(report(df, "price_lag_24h", "(a) Baseline naive: price_lag_24h solo", log, mask=test_mask))
    results.append(report(df, "analog_price_mean", "(a') Análogo solo (con embalse)", log, mask=test_mask))
    results.append(report(df, "analog_price_mean_wv", "(a'') Análogo solo (con water_value)", log, mask=test_mask))
    results.append(report(df, "price_simulado_final", "(a''') Motor solo (merit_order_francia, Fase 0d)", log, mask=test_mask))
    results.append(report(df, "pred_sin_motor", "(b) ML flexible sin motor (antes de Fase 10)", log, mask=test_mask))
    results.append(report(df, "pred_ml_flexible", "(b2) ML flexible + motor — producción", log, mask=test_mask))
    results.append(report(df, "pred_con_water_value", "(b3) ML flexible + motor + water_value — descartado", log, mask=test_mask))

    summary = pd.DataFrame(results)
    summary.to_parquet(OUTPUT_DIR / "arquitecturas_summary.parquet", index=False)

    log("\n-- SHAP del modelo ML flexible --")
    d = df.dropna(subset=FEATURES_FULL + ["price_real"]).copy()
    d["year_month"] = d["period_start_utc"].dt.to_period("M")
    months = sorted(d["year_month"].unique())
    train_final = d[d["year_month"].isin(months[:-1])]
    reg_final = lgb.LGBMRegressor(**LGB_PARAMS)
    reg_final.fit(train_final[FEATURES_FULL], train_final["price_real"])
    explainer = shap.TreeExplainer(reg_final)
    sample = train_final[FEATURES_FULL].sample(min(3000, len(train_final)), random_state=42)
    shap_values = explainer.shap_values(sample)
    mean_abs_shap = pd.Series(np.abs(shap_values).mean(axis=0), index=FEATURES_FULL).sort_values(ascending=False)
    log("  Importancia SHAP (valor absoluto medio):")
    for f, v in mean_abs_shap.items():
        log(f"    {f:30s} {v:6.2f}")

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(mean_abs_shap.index[::-1], mean_abs_shap.values[::-1], color="#2e6f95")
    ax.set_xlabel("|SHAP| medio")
    ax.set_title("Francia — importancia SHAP, ML flexible")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "shap_importance.png", dpi=110)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(summary["tag"], summary["corr"], color=["#b85c2e", "#8a5ea3", "#2e6f95"])
    ax.set_ylabel("Correlación con precio real (mismo periodo de test)")
    ax.set_title("Francia — comparación de arquitecturas")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "comparacion_arquitecturas.png", dpi=110)
    plt.close(fig)

    df.to_parquet(OUTPUT_DIR / "predicciones_fase1.parquet", index=False)
    (OUTPUT_DIR / "model_summary.txt").write_text("\n".join(lines), encoding="utf-8")
    log(f"\nGuardado. Gráficos: comparacion_arquitecturas.png, shap_importance.png")


if __name__ == "__main__":
    main()
