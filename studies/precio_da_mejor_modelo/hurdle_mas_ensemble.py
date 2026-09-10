"""
Fase 24 (2026-08-31, nueva sesión) — pedida directamente por la auditoría del
modelo final: el hurdle de precio<=0 (Fase 7, -27% MAE en esa zona) y el
ensemble pooled+por-hora de producción (Fase 17, 0,942/10,80) se validaron
CADA UNO por separado contra el modelo (b) pooled solo — nunca se combinaron
entre sí. Este script cierra ese hueco.

Diseño (L7/L9): en vez de reentrenar un hurdle nuevo con su propio regresor
condicional (duplicaría Fase 7), se reutiliza el ENSEMBLE de Fase 17 ya
validado como predicción base, y se le aplica como GATE el mismo clasificador
causal de "¿precio<=0?" de Fase 7 (mismas features, mismo umbral elegido solo
en train) — donde el clasificador dice "sí", se sustituye la predicción del
ensemble por el precio medio de la clase cero observado en train (zero_fill,
igual criterio que Fase 7); donde dice "no", se deja el ensemble tal cual.
Esto aísla la pregunta real: ¿ayuda el GATING del clasificador sobre el mejor
modelo ya conocido, o el ensemble ya captura esa zona igual de bien?

Uso:
    .venv\\Scripts\\python.exe -m studies.precio_da_mejor_modelo.hurdle_mas_ensemble

Requiere haber corrido antes build_dataset.py, model.py y ensemble_pooled_hora.py.
Escribe en output/: hurdle_mas_ensemble_summary.txt, hurdle_mas_ensemble.png
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
from studies.precio_da_mejor_modelo.model import INITIAL_TRAIN_MONTHS, LGB_PARAMS
from studies.precio_da_mejor_modelo.hurdle_quantile import FEATURES, ZERO_THRESHOLD, best_threshold_f1

OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def walk_forward_zero_gate(df: pd.DataFrame, log) -> pd.DataFrame:
    """Mismo bucle walk-forward y mismo clasificador que Fase 7
    (`hurdle_quantile.walk_forward_hurdle`), pero exponiendo por separado la
    bandera de clasificación (`pred_zero`) y el relleno de la clase cero
    (`zero_fill`) por periodo, en vez de devolver ya un precio combinado con
    un regresor propio — aquí el regresor de la zona positiva es el ENSEMBLE
    de Fase 17, no uno nuevo."""
    d = df.dropna(subset=FEATURES + ["price_real"]).copy()
    d["is_zero"] = (d["price_real"] <= ZERO_THRESHOLD).astype(int)
    d["year_month"] = d["period_start_utc"].dt.to_period("M")
    months = sorted(d["year_month"].unique())

    pred_zero = pd.Series(False, index=df.index)
    zero_fill_col = pd.Series(np.nan, index=df.index)
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
        flag = proba_test >= thr

        zero_fill = train.loc[train["is_zero"] == 1, "price_real"].mean()
        zero_fill = 0.0 if pd.isna(zero_fill) else zero_fill

        pred_zero.loc[test.index] = flag
        zero_fill_col.loc[test.index] = zero_fill
        n_predicted_zero += flag.sum()

    log(f"  Clasificador cero (idéntico a Fase 7): walk-forward completado, "
        f"{len(months) - INITIAL_TRAIN_MONTHS} meses de test, "
        f"{n_predicted_zero:,} periodos marcados 'precio<=0'")
    out = df[["period_start_utc"]].copy()
    out["pred_zero"] = pred_zero
    out["zero_fill"] = zero_fill_col
    return out


def report(df: pd.DataFrame, col: str, tag: str, log, mask: pd.Series | None = None) -> dict:
    valid = df.dropna(subset=[col, "price_real"])
    if mask is not None:
        valid = valid[mask.loc[valid.index]]
    corr = valid["price_real"].corr(valid[col])
    mae = (valid["price_real"] - valid[col]).abs().mean()
    bias = (valid[col] - valid["price_real"]).mean()
    log(f"  {tag}: n={len(valid):,}  correlación={corr:.4f}  MAE={mae:.2f}  sesgo={bias:+.2f}")
    return {"tag": tag, "n": len(valid), "corr": corr, "mae": mae, "bias": bias}


def main() -> None:
    lines: list[str] = []

    def log(s: str = "") -> None:
        print(s)
        lines.append(s)

    log("=" * 72)
    log("Fase 24 — ¿ayuda combinar el hurdle (Fase 7) con el ensemble de producción (Fase 17)?")
    log("=" * 72)

    dataset = pd.read_parquet(OUTPUT_DIR / "dataset.parquet")
    dataset["is_holiday"] = dataset["is_holiday"].astype(int)
    dataset["is_weekend"] = dataset["is_weekend"].astype(int)

    log("\n-- Clasificador causal de precio<=0 (mismo criterio que Fase 7) --")
    gate = walk_forward_zero_gate(dataset, log)

    ensemble = pd.read_parquet(OUTPUT_DIR / "predicciones_fase17.parquet")[
        ["period_start_utc", "price_real", "pred_ensemble"]
    ]
    df = ensemble.merge(gate, on="period_start_utc", how="inner")
    df["pred_ensemble_hurdle"] = np.where(df["pred_zero"], df["zero_fill"], df["pred_ensemble"])

    test_mask = df["pred_ensemble"].notna() & df["pred_zero"].notna()
    valid = df[test_mask].copy()
    log(f"\n-- Comparación en el periodo común evaluable ({len(valid):,} periodos) --")
    results = [
        report(valid, "pred_ensemble", "(g) Ensemble solo (Fase 17, producción actual)", log),
        report(valid, "pred_ensemble_hurdle", "(h) Ensemble + gate de hurdle (Fase 24)", log),
    ]

    real_zero = valid[valid["price_real"] <= ZERO_THRESHOLD]
    real_pos = valid[valid["price_real"] > ZERO_THRESHOLD]
    log(f"\n-- Desglose por zona real --")
    log(f"  Zona precio<=0 real (n={len(real_zero):,}):")
    for tag, col in [("  (g) Ensemble solo", "pred_ensemble"), ("  (h) Ensemble+gate", "pred_ensemble_hurdle")]:
        mae_zone = (real_zero["price_real"] - real_zero[col]).abs().mean()
        log(f"  {tag}: MAE={mae_zone:.2f}")
    log(f"  Zona precio>0 real (n={len(real_pos):,}):")
    for tag, col in [("  (g) Ensemble solo", "pred_ensemble"), ("  (h) Ensemble+gate", "pred_ensemble_hurdle")]:
        mae_zone = (real_pos["price_real"] - real_pos[col]).abs().mean()
        log(f"  {tag}: MAE={mae_zone:.2f}")

    # Coste de los falsos positivos del gate: periodos marcados "precio<=0" que en
    # realidad NO lo fueron — ahí el gate tira una predicción del ensemble que
    # podía ser buena y la sustituye por zero_fill, potencialmente empeorando.
    flagged = valid[valid["pred_zero"]]
    false_pos = flagged[flagged["price_real"] > ZERO_THRESHOLD]
    true_pos = flagged[flagged["price_real"] <= ZERO_THRESHOLD]
    log(f"\n-- Coste/beneficio del gate sobre los {len(flagged):,} periodos marcados 'precio<=0' --")
    log(f"  Aciertos reales (precio<=0 de verdad): {len(true_pos):,} "
        f"({100*len(true_pos)/len(flagged):.1f}% de lo marcado)" if len(flagged) else "  (nada marcado)")
    log(f"  Falsos positivos (precio en realidad >0): {len(false_pos):,} "
        f"({100*len(false_pos)/len(flagged):.1f}% de lo marcado)" if len(flagged) else "")
    if len(false_pos):
        mae_fp_ensemble = (false_pos["price_real"] - false_pos["pred_ensemble"]).abs().mean()
        mae_fp_gated = (false_pos["price_real"] - false_pos["pred_ensemble_hurdle"]).abs().mean()
        log(f"  Dentro de los falsos positivos: MAE ensemble solo={mae_fp_ensemble:.2f} "
            f"vs. MAE tras el gate={mae_fp_gated:.2f} "
            f"({'empeora' if mae_fp_gated > mae_fp_ensemble else 'mejora'})")
    if len(true_pos):
        mae_tp_ensemble = (true_pos["price_real"] - true_pos["pred_ensemble"]).abs().mean()
        mae_tp_gated = (true_pos["price_real"] - true_pos["pred_ensemble_hurdle"]).abs().mean()
        log(f"  Dentro de los aciertos reales: MAE ensemble solo={mae_tp_ensemble:.2f} "
            f"vs. MAE tras el gate={mae_tp_gated:.2f} "
            f"({'empeora' if mae_tp_gated > mae_tp_ensemble else 'mejora'})")

    summary = pd.DataFrame(results)
    summary.to_parquet(OUTPUT_DIR / "hurdle_mas_ensemble_summary.parquet", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].bar(summary["tag"], summary["mae"], color=["#2e6f95", "#c0392b"])
    axes[0].set_ylabel("MAE agregado (EUR/MWh)")
    axes[0].set_title("Agregado")
    axes[0].tick_params(axis="x", rotation=15)

    zone_labels = ["precio<=0 real", "precio>0 real"]
    mae_g = [
        (real_zero["price_real"] - real_zero["pred_ensemble"]).abs().mean(),
        (real_pos["price_real"] - real_pos["pred_ensemble"]).abs().mean(),
    ]
    mae_h = [
        (real_zero["price_real"] - real_zero["pred_ensemble_hurdle"]).abs().mean(),
        (real_pos["price_real"] - real_pos["pred_ensemble_hurdle"]).abs().mean(),
    ]
    x = np.arange(len(zone_labels))
    width = 0.35
    axes[1].bar(x - width / 2, mae_g, width, label="(g) Ensemble solo", color="#2e6f95")
    axes[1].bar(x + width / 2, mae_h, width, label="(h) Ensemble+gate", color="#c0392b")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(zone_labels)
    axes[1].set_ylabel("MAE (EUR/MWh)")
    axes[1].set_title("Por zona real")
    axes[1].legend()
    fig.suptitle("Fase 24 — hurdle como gate del ensemble de producción")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "hurdle_mas_ensemble.png", dpi=110)
    plt.close(fig)

    df.to_parquet(OUTPUT_DIR / "predicciones_fase24.parquet", index=False)
    summary_path = OUTPUT_DIR / "hurdle_mas_ensemble_summary.txt"
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    log(f"\nResumen guardado en {summary_path}")
    log("Gráfico: hurdle_mas_ensemble.png")


if __name__ == "__main__":
    main()
