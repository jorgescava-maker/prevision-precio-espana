"""
Fase 1 de "el mejor modelo posible de precio DA" — ver DESIGN.md §4. Compara,
con walk-forward mensual (L8) y sobre el MISMO periodo de test para que la
comparación sea justa (L9), tres arquitecturas candidatas:

  (a) el motor de merit_order solo (`price_simulado_final`) — el baseline a batir.
  (b) ML flexible (LightGBM) sobre todas las features D-1, incluido el propio
      motor como una feature más (L6).
  (c) híbrido — ML sobre el RESIDUO del motor (price_real - price_simulado_final),
      usando el resto de features (sin el motor como insumo, para no predecir
      su propio residuo circularmente) — predicción final = motor + residuo predicho.

Uso:
    .venv\\Scripts\\python.exe -m studies.precio_da_mejor_modelo.model

Requiere haber corrido antes build_dataset.py. Escribe en output/:
    model_summary.txt, comparacion_arquitecturas.png, shap_importance.png
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

from etl.common import logging_config  # noqa: F401 - fuerza stdout a UTF-8

OUTPUT_DIR = Path(__file__).resolve().parent / "output"

# findings.md #103 (2026-09-01): `coal_eur_mwh_th` (API2/Yahoo) no se
# actualiza desde 2025-12-26 — mantenerla en FEATURES_FULL con el dropna
# estricto de walk_forward() descartaba en silencio TODO 2026 (8 meses,
# incluida la ola de calor) de la evaluación/entrenamiento. Excluida aquí
# (mismo criterio que bandera_extremo.py/modelo_diario.py ya habían adoptado
# por separado, nunca generalizado) para recuperar esa ventana. La columna
# sigue en el dataset por trazabilidad, solo se quita de la lista de features.
FEATURES_FULL = [
    "forecast_demanda_mw", "forecast_eolica_mw", "forecast_solar_mw",
    "water_value_eur_mwh", "ttf_eur_mwh", "eua_eur_t",
    "cogeneracion_resto_clim", "solar_termica_clim", "termica_renovable_clim",
    "net_import_mw_clim", "demand_to_cover_mw", "must_run_final_mw",
    "nuclear_capacity_avail_mw", "gas_capacity_avail_mw", "coal_capacity_avail_mw",
    "analog_price_mean_v2", "analog_price_mean_v3", "analog_price_mean_v4", "analog_price_mean_v5", "analog_price_dev",
    "price_lag_24h", "price_lag_48h", "price_lag_168h", "price_ema_7d", "price_ema_28d",
    "price_boundary_prev_day",
    "price_lag_24h_fr", "price_lag_168h_fr", "price_lag_24h_pt", "price_lag_168h_pt",
    "ntc_es_fr_mw", "nuclear_capacity_fr_avail_mw",
    "hour_of_day", "period_in_hour", "period_of_day", "day_of_week", "month",
    "is_holiday", "is_weekend",
]
FEATURES_HYBRID = FEATURES_FULL  # sin price_simulado_final — se usa como ancla, no como insumo del residuo

INITIAL_TRAIN_MONTHS = 12
# findings.md #109 (campaña Fable, 2026-09-01/02): los parámetros anteriores
# (300 árboles, lr 0.05, max_depth=5, num_leaves por defecto=31) resultaron ser
# EL TECHO REAL del modelo, no las features. `max_depth=5` estaba aquí desde el
# inicio del proyecto sin cuestionarse y va contra el diseño de LightGBM, que
# crece leaf-wise (elige la hoja de mayor ganancia, árboles asimétricos): el
# control natural de capacidad es `num_leaves`, no un tope de profundidad bajo.
# Liberar la profundidad dio el mayor salto individual de 6 rondas de
# experimentos (10,61->10,53 en España).
#
# Verificado en los 4 países ANTES de adoptar (este dict lo importan POR
# REFERENCIA Francia dedicado, Alemania y Países Bajos — no se degrada un
# modelo para mejorar otro, ver feedback_no_degradar_modelo_compartido y
# `verificacion_multipais_fable.py`). Todos mejoran, ninguno empeora:
#   España         10,82 -> 10,53   (pooled ES+FR + ensemble + gate de hurdle)
#                  (medido de nuevo con el pipeline ya adoptado: 10,50 — el
#                   clasificador del gate también usa este dict, así que
#                   mejoró de rebote; ver findings.md #111. Y 9,96 desde que
#                   hay un tercer miembro de redes, findings.md #112.)
#   Francia        15,86 -> 15,71   (pooled dedicado + ensemble)
#   Alemania       15,21 -> 15,17   (solo + ensemble)
#   Países Bajos   17,27 -> 17,21   (solo + ensemble)
# Meseta acotada por ambos lados: 63 hojas pierde (10,63), 191 no aporta (10,54).
LGB_PARAMS = dict(
    n_estimators=800, max_depth=-1, num_leaves=127, learning_rate=0.02,
    subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=-1,
)


def walk_forward(df: pd.DataFrame, features: list[str], target_col: str, tag: str, log) -> pd.Series:
    d = df.dropna(subset=features + [target_col]).copy()
    d["year_month"] = d["period_start_utc"].dt.to_period("M")
    months = sorted(d["year_month"].unique())
    if len(months) <= INITIAL_TRAIN_MONTHS:
        raise SystemExit(f"{tag}: solo {len(months)} meses, se necesitan más de {INITIAL_TRAIN_MONTHS}")

    preds = pd.Series(np.nan, index=df.index)
    for i in range(INITIAL_TRAIN_MONTHS, len(months)):
        train = d[d["year_month"].isin(months[:i])]
        test = d[d["year_month"] == months[i]]
        if test.empty or train.empty:
            continue
        reg = lgb.LGBMRegressor(**LGB_PARAMS)
        reg.fit(train[features], train[target_col])
        preds.loc[test.index] = reg.predict(test[features])

    log(f"  {tag}: walk-forward completado, {len(months) - INITIAL_TRAIN_MONTHS} meses de test")
    return preds


MIN_ROWS_PER_GROUP = 30


def walk_forward_by_group(df: pd.DataFrame, features: list[str], target_col: str, group_col: str, tag: str, log) -> pd.Series:
    """Fase 4 — un modelo INDEPENDIENTE por valor de `group_col` (p.ej. una
    hora del día distinta) en vez de un único modelo agrupado con esa
    variable como feature más — pregunta abierta de DESIGN.md §6: la forma
    de la relación entre features y precio podría no ser la misma a las 4 AM
    que a las 20h (patrón de "doble joroba" de A2)."""
    d = df.dropna(subset=features + [target_col, group_col]).copy()
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

    log(f"  {tag}: walk-forward por grupo completado, {len(months) - INITIAL_TRAIN_MONTHS} meses de test, "
        f"{n_skipped:,} filas sin predicción por falta de histórico del grupo")
    return preds


def report(df: pd.DataFrame, price_col: str, tag: str, log, mask: pd.Series | None = None) -> dict:
    valid = df.dropna(subset=[price_col, "price_real"])
    if mask is not None:
        valid = valid[mask.loc[valid.index]]
    corr = valid["price_real"].corr(valid[price_col])
    mae = (valid["price_real"] - valid[price_col]).abs().mean()
    bias = (valid[price_col] - valid["price_real"]).mean()
    log(f"  {tag}: n={len(valid):,}  correlación={corr:.3f}  MAE={mae:.2f}  sesgo={bias:+.2f}")
    return {"tag": tag, "n": len(valid), "corr": corr, "mae": mae, "bias": bias}


def main() -> None:
    lines: list[str] = []

    def log(s: str = "") -> None:
        print(s)
        lines.append(s)

    log("=" * 72)
    log("Mejor modelo de precio DA — Fase 3: resolución nativa (15 min / horaria)")
    log("=" * 72)

    df = pd.read_parquet(OUTPUT_DIR / "dataset.parquet")
    df["is_holiday"] = df["is_holiday"].astype(int)
    df["is_weekend"] = df["is_weekend"].astype(int)

    # -----------------------------------------------------------------
    log("\n-- (b) ML flexible sobre todas las features (incluido el motor) --")
    features_b = FEATURES_FULL + ["price_simulado_final"]
    df["pred_ml_flexible"] = walk_forward(df, features_b, "price_real", "ML flexible", log)

    # -----------------------------------------------------------------
    log("\n-- (c) Híbrido: ML sobre el residuo del motor --")
    df["residuo_motor"] = df["price_real"] - df["price_simulado_final"]
    df["pred_residuo"] = walk_forward(df, FEATURES_HYBRID, "residuo_motor", "Híbrido (residuo)", log)
    df["pred_hibrido"] = df["price_simulado_final"] + df["pred_residuo"]

    # -----------------------------------------------------------------
    log("\n-- (d) ML por hora del día: un modelo independiente por hour_of_day --")
    features_d = [f for f in features_b if f not in ("hour_of_day", "period_of_day")]
    df["pred_por_hora"] = walk_forward_by_group(df, features_d, "price_real", "hour_of_day", "ML por hora", log)

    # -----------------------------------------------------------------
    # Periodo de test común (después del arranque walk-forward) para que
    # las 4 arquitecturas se comparen sobre EXACTAMENTE las mismas horas (L9).
    test_mask = df["pred_ml_flexible"].notna() & df["pred_hibrido"].notna() & df["pred_por_hora"].notna()
    log(f"\n-- Comparación en el MISMO periodo de test ({test_mask.sum():,} periodos) --")
    results = []
    results.append(report(df, "price_simulado_final", "(a) Motor solo (baseline)", log, mask=test_mask))
    results.append(report(df, "pred_ml_flexible", "(b) ML flexible (pooled)", log, mask=test_mask))
    results.append(report(df, "pred_hibrido", "(c) Híbrido (motor + residuo ML)", log, mask=test_mask))
    results.append(report(df, "pred_por_hora", "(d) ML por hora del día", log, mask=test_mask))

    summary = pd.DataFrame(results)
    summary.to_parquet(OUTPUT_DIR / "arquitecturas_summary.parquet", index=False)

    # -----------------------------------------------------------------
    log("\n-- SHAP de la arquitectura ganadora (por correlación) --")
    winner = summary.loc[summary["corr"].idxmax(), "tag"]
    log(f"  Ganadora: {winner}")
    if "hora" in winner:
        log("  (d) son 24 modelos independientes — el SHAP de abajo explica en su lugar el modelo")
        log("  (b) agrupado, como proxy razonable de qué features importan en general.")

    d = df.dropna(subset=FEATURES_FULL + ["price_real"]).copy()
    d["year_month"] = d["period_start_utc"].dt.to_period("M")
    months = sorted(d["year_month"].unique())
    train_final = d[d["year_month"].isin(months[:-1])]  # último mes fuera, para un ajuste final "actual"
    reg_final = lgb.LGBMRegressor(**LGB_PARAMS)
    if "Híbrido" in winner:
        reg_final.fit(train_final[FEATURES_HYBRID], train_final["price_real"] - train_final["price_simulado_final"])
        explainer = shap.TreeExplainer(reg_final)
        shap_values = explainer.shap_values(train_final[FEATURES_HYBRID].sample(min(3000, len(train_final)), random_state=42))
        feat_names = FEATURES_HYBRID
    else:
        reg_final.fit(train_final[features_b], train_final["price_real"])
        explainer = shap.TreeExplainer(reg_final)
        sample = train_final[features_b].sample(min(3000, len(train_final)), random_state=42)
        shap_values = explainer.shap_values(sample)
        feat_names = features_b

    mean_abs_shap = pd.Series(np.abs(shap_values).mean(axis=0), index=feat_names).sort_values(ascending=False)
    log("  Importancia SHAP (valor absoluto medio):")
    for f, v in mean_abs_shap.items():
        log(f"    {f:30s} {v:6.2f}")

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.barh(mean_abs_shap.index[::-1], mean_abs_shap.values[::-1], color="#4a7c59")
    ax.set_xlabel("|SHAP| medio")
    ax.set_title(f"Fase 3 — importancia SHAP de la arquitectura ganadora ({winner})")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "shap_importance.png", dpi=110)
    plt.close(fig)

    # -----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(summary["tag"], summary["corr"], color=["#2e6f95", "#b85c2e", "#4a7c59", "#8a5ea3"])
    ax.set_ylabel("Correlación con precio real (mismo periodo de test)")
    ax.set_title("Fase 3 — comparación de arquitecturas (resolución nativa)")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "comparacion_arquitecturas.png", dpi=110)
    plt.close(fig)

    df.to_parquet(OUTPUT_DIR / "predicciones_fase1.parquet", index=False)
    summary_path = OUTPUT_DIR / "model_summary.txt"
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    log(f"\nResumen guardado en {summary_path}")
    log("Gráficos: comparacion_arquitecturas.png, shap_importance.png")


if __name__ == "__main__":
    main()
