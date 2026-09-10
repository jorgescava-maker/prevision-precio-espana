"""
Fase 7 de "el mejor modelo posible de precio DA" — dos formas DISTINTAS de
plantear el problema, en vez de más features sobre la misma arquitectura de
regresión puntual (pedido explícito del usuario, 2026-08-31):

  (1) Hurdle / dos etapas: separar "¿va a ser precio cero/negativo?" (un
      mecanismo casi determinista, ver A2) del resto — un clasificador
      decide, y SOLO para las horas que no caen ahí se entrena un regresor
      "condicional a precio positivo", sin que la masa de ceros tire de su
      ajuste.
  (2) Predicción por cuantiles: en vez de un único número, un intervalo
      (p10/p50/p90) — da una medida de incertidumbre que el punto único no
      tiene, relevante dado lo que E1 ya encontró sobre las colas del precio.

Ambos comparados honestamente (L9) contra el modelo (b) ML flexible ya
validado, en el MISMO periodo de test.

Uso:
    .venv\\Scripts\\python.exe -m studies.precio_da_mejor_modelo.hurdle_quantile

Requiere haber corrido antes build_dataset.py y model.py (usa
predicciones_fase1.parquet para comparar contra pred_ml_flexible). Escribe en
output/: hurdle_quantile_summary.txt, quantile_calibration.png,
hurdle_vs_pooled.png
"""

from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from etl.common import logging_config  # noqa: F401 - fuerza stdout a UTF-8
from studies.precio_da_mejor_modelo.model import FEATURES_FULL, INITIAL_TRAIN_MONTHS, LGB_PARAMS

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
FEATURES = FEATURES_FULL + ["price_simulado_final"]

ZERO_THRESHOLD = 0.0  # mismo criterio que A2 (price_real<=0)


# ---------------------------------------------------------------------------
# (1) Hurdle: clasificador (¿precio<=0?) + regresor condicional a precio>0
# ---------------------------------------------------------------------------

def best_threshold_f1(proba: np.ndarray, y: np.ndarray) -> float:
    """Umbral que maximiza F1 SOLO en train (L9 — mismo criterio que A2/D1,
    nunca se mira test para elegir el umbral)."""
    best_thr, best_f1 = 0.5, -1.0
    for thr in np.arange(0.1, 0.95, 0.05):
        pred = (proba >= thr).astype(int)
        tp = ((pred == 1) & (y == 1)).sum()
        fp = ((pred == 1) & (y == 0)).sum()
        fn = ((pred == 0) & (y == 1)).sum()
        precision = tp / (tp + fp) if (tp + fp) else 0
        recall = tp / (tp + fn) if (tp + fn) else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
        if f1 > best_f1:
            best_thr, best_f1 = thr, f1
    return best_thr


def walk_forward_hurdle(df: pd.DataFrame, log) -> pd.Series:
    d = df.dropna(subset=FEATURES + ["price_real"]).copy()
    d["is_zero"] = (d["price_real"] <= ZERO_THRESHOLD).astype(int)
    d["year_month"] = d["period_start_utc"].dt.to_period("M")
    months = sorted(d["year_month"].unique())

    preds = pd.Series(np.nan, index=df.index)
    n_predicted_zero = 0
    for i in range(INITIAL_TRAIN_MONTHS, len(months)):
        train = d[d["year_month"].isin(months[:i])]
        test = d[d["year_month"] == months[i]]
        if test.empty or train.empty or train["is_zero"].nunique() < 2:
            continue

        clf = lgb.LGBMClassifier(**LGB_PARAMS)
        clf.fit(train[FEATURES], train["is_zero"])
        proba_train = clf.predict_proba(train[FEATURES])[:, 1]
        thr = best_threshold_f1(proba_train, train["is_zero"].to_numpy())
        proba_test = clf.predict_proba(test[FEATURES])[:, 1]
        pred_zero = proba_test >= thr

        train_positive = train[train["price_real"] > ZERO_THRESHOLD]
        reg = lgb.LGBMRegressor(**LGB_PARAMS)
        reg.fit(train_positive[FEATURES], train_positive["price_real"])
        reg_pred = reg.predict(test[FEATURES])

        zero_fill = train.loc[train["is_zero"] == 1, "price_real"].mean()
        zero_fill = 0.0 if pd.isna(zero_fill) else zero_fill
        combined = np.where(pred_zero, zero_fill, reg_pred)
        preds.loc[test.index] = combined
        n_predicted_zero += pred_zero.sum()

    log(f"  Hurdle: walk-forward completado, {len(months) - INITIAL_TRAIN_MONTHS} meses de test, "
        f"{n_predicted_zero:,} periodos clasificados como 'precio<=0'")
    return preds


# ---------------------------------------------------------------------------
# (2) Cuantiles: p10/p50/p90 walk-forward
# ---------------------------------------------------------------------------

def walk_forward_quantile(df: pd.DataFrame, alpha: float, log) -> pd.Series:
    d = df.dropna(subset=FEATURES + ["price_real"]).copy()
    d["year_month"] = d["period_start_utc"].dt.to_period("M")
    months = sorted(d["year_month"].unique())

    params = dict(LGB_PARAMS)
    params["objective"] = "quantile"
    params["alpha"] = alpha

    preds = pd.Series(np.nan, index=df.index)
    for i in range(INITIAL_TRAIN_MONTHS, len(months)):
        train = d[d["year_month"].isin(months[:i])]
        test = d[d["year_month"] == months[i]]
        if test.empty or train.empty:
            continue
        reg = lgb.LGBMRegressor(**params)
        reg.fit(train[FEATURES], train["price_real"])
        preds.loc[test.index] = reg.predict(test[FEATURES])

    log(f"  Cuantil q{int(alpha*100)}: walk-forward completado")
    return preds


def main() -> None:
    lines: list[str] = []

    def log(s: str = "") -> None:
        print(s)
        lines.append(s)

    log("=" * 72)
    log("Mejor modelo de precio DA — Fase 7: hurdle (cero/no-cero) + cuantiles")
    log("=" * 72)

    df = pd.read_parquet(OUTPUT_DIR / "dataset.parquet")
    df["is_holiday"] = df["is_holiday"].astype(int)
    df["is_weekend"] = df["is_weekend"].astype(int)
    pooled = pd.read_parquet(OUTPUT_DIR / "predicciones_fase1.parquet")[["period_start_utc", "pred_ml_flexible"]]
    df = df.merge(pooled, on="period_start_utc", how="left")

    # -----------------------------------------------------------------
    log("\n-- (1) Hurdle: clasificador precio<=0 + regresor condicional a precio>0 --")
    df["pred_hurdle"] = walk_forward_hurdle(df, log)

    test_mask = df["pred_hurdle"].notna() & df["pred_ml_flexible"].notna()
    valid = df[test_mask]
    for tag, col in [("(b) ML flexible pooled (referencia, ya validado)", "pred_ml_flexible"),
                      ("(e) Hurdle (cero/no-cero)", "pred_hurdle")]:
        corr = valid["price_real"].corr(valid[col])
        mae = (valid["price_real"] - valid[col]).abs().mean()
        bias = (valid[col] - valid["price_real"]).mean()
        log(f"  {tag}: n={len(valid):,}  correlación={corr:.3f}  MAE={mae:.2f}  sesgo={bias:+.2f}")

    # Desglose específico en la zona de precio<=0 real, donde el hurdle debería notarse
    near_zero_real = valid[valid["price_real"] <= ZERO_THRESHOLD]
    log(f"\n  Solo en las {len(near_zero_real):,} horas con precio real<=0:")
    for tag, col in [("(b) ML flexible pooled", "pred_ml_flexible"), ("(e) Hurdle", "pred_hurdle")]:
        mae_zone = (near_zero_real["price_real"] - near_zero_real[col]).abs().mean()
        log(f"    {tag}: MAE en esa zona={mae_zone:.2f}")

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.bar(["(b) ML flexible\n(pooled)", "(e) Hurdle\n(cero/no-cero)"],
           [valid["price_real"].corr(valid["pred_ml_flexible"]), valid["price_real"].corr(valid["pred_hurdle"])],
           color=["#2e6f95", "#c0392b"])
    ax.set_ylabel("Correlación con precio real (mismo periodo de test)")
    ax.set_title("Fase 7 — hurdle vs. modelo pooled")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "hurdle_vs_pooled.png", dpi=110)
    plt.close(fig)

    # -----------------------------------------------------------------
    log("\n-- (2) Predicción por cuantiles (p10/p50/p90) --")
    for alpha in (0.1, 0.5, 0.9):
        df[f"pred_q{int(alpha*100)}"] = walk_forward_quantile(df, alpha, log)

    qvalid = df.dropna(subset=["pred_q10", "pred_q50", "pred_q90", "price_real"])
    below_q10 = (qvalid["price_real"] < qvalid["pred_q10"]).mean()
    below_q90 = (qvalid["price_real"] < qvalid["pred_q90"]).mean()
    coverage_80 = ((qvalid["price_real"] >= qvalid["pred_q10"]) & (qvalid["price_real"] <= qvalid["pred_q90"])).mean()
    mae_q50 = (qvalid["price_real"] - qvalid["pred_q50"]).abs().mean()
    corr_q50 = qvalid["price_real"].corr(qvalid["pred_q50"])
    avg_width = (qvalid["pred_q90"] - qvalid["pred_q10"]).mean()

    log(f"  n={len(qvalid):,} periodos")
    log(f"  Calibración: % real < q10 = {100*below_q10:.1f}% (objetivo ~10%)  |  "
        f"% real < q90 = {100*below_q90:.1f}% (objetivo ~90%)")
    log(f"  Cobertura del intervalo [q10,q90] = {100*coverage_80:.1f}% (objetivo ~80%)")
    log(f"  Ancho medio del intervalo [q10,q90] = {avg_width:.2f} EUR/MWh")
    log(f"  q50 (mediana) como predicción puntual: correlación={corr_q50:.3f}  MAE={mae_q50:.2f} "
        f"(comparar con {valid['price_real'].corr(valid['pred_ml_flexible']):.3f}/"
        f"{(valid['price_real']-valid['pred_ml_flexible']).abs().mean():.2f} del modelo (b))")

    fig, ax = plt.subplots(figsize=(11, 4.5))
    sample = qvalid[(qvalid["period_start_utc"] >= "2026-06-01") & (qvalid["period_start_utc"] < "2026-06-08")].sort_values("period_start_utc")
    ax.plot(sample["period_start_utc"], sample["price_real"], label="real", color="#2e6f95", linewidth=1.3)
    ax.plot(sample["period_start_utc"], sample["pred_q50"], label="mediana (q50)", color="#b85c2e", linewidth=1.1)
    ax.fill_between(sample["period_start_utc"], sample["pred_q10"], sample["pred_q90"], alpha=0.25, color="#b85c2e", label="intervalo [q10,q90]")
    ax.set_ylabel("EUR/MWh")
    ax.set_title("Fase 7 — predicción por cuantiles, muestra 2026-06-01 a 2026-06-08")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "quantile_calibration.png", dpi=110)
    plt.close(fig)

    df.to_parquet(OUTPUT_DIR / "predicciones_fase7.parquet", index=False)
    summary_path = OUTPUT_DIR / "hurdle_quantile_summary.txt"
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    log(f"\nResumen guardado en {summary_path}")
    log("Gráficos: hurdle_vs_pooled.png, quantile_calibration.png")


if __name__ == "__main__":
    main()
