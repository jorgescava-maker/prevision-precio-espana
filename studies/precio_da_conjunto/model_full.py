"""
Segunda vuelta del modelo conjunto (ver RESULTS.md, "próximos pasos" #1):
en vez de reducir a la intersección de features comunes, se entrena el
modelo pooled con las features PROPIAS de cada país también incluidas
(water_value/must_run/demand_to_cover/motor de merit-order/análogo v5/
FR-PT lags/NTC de España; embalse de Francia) — las columnas exclusivas de
un país quedan NaN en las filas del otro, y se dejan así a propósito:
LightGBM maneja NaN de forma nativa en cada split (aprende una dirección
por defecto), así que una columna irrelevante para un país simplemente no
se usa en sus filas sin necesidad de descartarlas. Distinto del resto del
proyecto (que usa `dropna` estricto antes de entrenar) — aquí se evita
deliberadamente para no tirar la mitad del dataset.

Uso:
    .venv\\Scripts\\python.exe -m studies.precio_da_conjunto.model_full
"""

from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from etl.common import logging_config  # noqa: F401
from studies.precio_da_mejor_modelo.model import LGB_PARAMS, INITIAL_TRAIN_MONTHS, report
from studies.precio_da_conjunto.model import COMMON_FEATURES, load_spain, load_france

OUTPUT_DIR = Path(__file__).resolve().parent / "output"

SPAIN_ONLY = [
    "water_value_eur_mwh", "must_run_final_mw", "demand_to_cover_mw",
    "price_simulado_final", "net_import_mw_clim",
    "cogeneracion_resto_clim", "solar_termica_clim", "termica_renovable_clim",
    "price_lag_24h_fr", "price_lag_168h_fr", "price_lag_24h_pt", "price_lag_168h_pt",
    "ntc_es_fr_mw", "nuclear_capacity_fr_avail_mw",
    # --- revisión del 2026-09-13: diez variables, ningún dato nuevo ---
    # `reserve_margin_mw` y `ratio_renovable_periodo` ya se construían en
    # `build_dataset.py` y nunca se habían conectado aquí. Las otras ocho las
    # construye `precio_da_mejor_modelo/features_v3.py`, que documenta de dónde
    # sale cada una y por qué es D-1-segura.
    "reserve_margin_mw", "ratio_renovable_periodo", "tension_fr",
    "ltsc_90d", "ltsc_365d", "ltsc_pendiente", "desvio_ltsc",
    "prev_dem_var24h", "prev_eol_var24h", "prev_sol_var24h",
    # --- 2026-09-14: margen_neto = reserve_margin_mw - tension_fr, ADOPTADA
    # en el repo privado (findings.md #189). Cero dato nuevo, resta de dos
    # columnas que ya estaban aquí arriba.
    "margen_neto",
]
# findings.md #107/#109 (2026-09-01): curva real de oferta de OMIE, versión
# D-1-segura (Fase 15 de precio_da_mejor_modelo/build_dataset.py) — PROBADA Y
# DESCARTADA contra el campeón REAL (pooled+ensemble+hurdle), no un proxy
# simplificado. Baseline limpio recalculado primero (el dataset base cambió
# en esta misma sesión, findings.md #108, recuperados feb/marzo de 2026):
# campeón sin curva = 10,82 (mejora respecto al 10,85 antiguo, por el fix de
# #108, nada que ver con la curva). CON estas dos features en SPAIN_ONLY:
# 10,84 — empeora, dentro del ruido pero sin ganancia real. Contradice el
# resultado prometedor de un test simplificado previo (un solo LightGBM sin
# ensemble/hurdle, MAE~16 de baseline) — con una baseline mucho más débil que
# el campeón real, cualquier feature nueva parece ayudar más de lo que ayuda
# de verdad; el test que importa es siempre contra el pipeline de producción
# completo. NO se añaden a SPAIN_ONLY. Se deja aquí sin usar por si en el
# futuro aparece una variante de la curva con más señal (ver findings.md
# #109, pendiente: price_at_plus1000/2000mw sin probar todavía).
SPAIN_ONLY_CURVE = ["slope_next500_eur_per_mw_clim", "unused_sell_headroom_mw_clim"]  # NO USAR — descartado, ver arriba
# price_simulado_final de Francia (motor de merit_order_francia, Fase 10 de
# precio_da_francia) se probó AQUÍ 2026-08-31 y se REVIRTIÓ el mismo día,
# decisión explícita del usuario: "vamos a mantener las cosas de españa sin
# que se toquen, no se puede empeorar un modelo para mejorar el otro, es
# mejor duplicar las cosas para reusarlas en estos casos". El modelo
# conjunto es un ÚNICO LightGBM compartido entre ambos países — añadir esa
# feature mejoraba a Francia (12,89->12,74) pero empeoraba a España como
# efecto secundario del árbol compartido (10,63->10,71), sin tocar ninguna
# feature propia de España. Ver precio_da_francia/RESULTS.md Fase 10b y
# findings.md para el detalle completo. La mejora de Francia sigue vigente
# en su propia línea en solitario (precio_da_francia/model.py::FEATURES_FULL,
# que SÍ la tiene — pipeline separado, no comparte modelo con España).
FRANCE_ONLY = ["embalse_gwh"]


def load_spain_full() -> pd.DataFrame:
    df = pd.read_parquet(Path(__file__).resolve().parents[1] / "precio_da_mejor_modelo" / "output" / "dataset.parquet")
    df = df.rename(columns={"analog_price_mean_v5": "analog_price_mean"})
    df["is_holiday"] = df["is_holiday"].astype(int)
    df["is_weekend"] = df["is_weekend"].astype(int)
    df["country"] = "ES"
    keep = ["period_start_utc", "price_real", "country"] + COMMON_FEATURES + SPAIN_ONLY
    return df[keep].copy()


def load_france_full() -> pd.DataFrame:
    df = pd.read_parquet(Path(__file__).resolve().parents[1] / "precio_da_francia" / "output" / "dataset.parquet")
    df["is_holiday"] = df["is_holiday"].astype(int)
    df["is_weekend"] = df["is_weekend"].astype(int)
    df["country"] = "FR"
    keep = ["period_start_utc", "price_real", "country"] + COMMON_FEATURES + FRANCE_ONLY
    return df[keep].copy()


def walk_forward_nan_ok(df: pd.DataFrame, features: list[str], required: list[str], target_col: str, tag: str, log) -> pd.Series:
    """Igual que `walk_forward` del resto del proyecto, pero solo descarta
    filas con NaN en `required` (features comunes + target) — las columnas
    exclusivas de un país pueden quedar NaN, LightGBM las maneja de forma
    nativa (aprende una dirección por defecto en cada split)."""
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
    log(f"  {tag}: walk-forward completado, {len(months) - INITIAL_TRAIN_MONTHS} meses de test")
    return preds


def main() -> None:
    lines: list[str] = []

    def log(s: str = "") -> None:
        print(s)
        lines.append(s)

    log("=" * 72)
    log("Modelo conjunto España+Francia — features completas (NaN nativo)")
    log("=" * 72)

    es = load_spain_full()
    fr = load_france_full()
    log(f"España: {len(es):,} periodos ({len(SPAIN_ONLY)} features propias). "
        f"Francia: {len(fr):,} periodos ({len(FRANCE_ONLY)} features propias).")

    pooled = pd.concat([es, fr], ignore_index=True)
    pooled["is_spain"] = (pooled["country"] == "ES").astype(int)
    # dict.fromkeys en vez de + directo: SPAIN_ONLY y FRANCE_ONLY comparten
    # el nombre `price_simulado_final` desde 2026-08-31 (cada motor calcula
    # el de su propio país, pero el NOMBRE de columna es el mismo) — sin
    # dedupe, LightGBM recibiría esa columna duplicada en la lista de
    # features.
    all_features = list(dict.fromkeys(COMMON_FEATURES + SPAIN_ONLY + FRANCE_ONLY + ["is_spain"]))

    log(f"\nTotal features del modelo conjunto completo: {len(all_features)}")
    log("\n-- (Z2) Pooled con features completas (NaN nativo) --")
    pooled["pred_pooled_full"] = walk_forward_nan_ok(
        pooled, all_features, required=COMMON_FEATURES, target_col="price_real", tag="pooled completo", log=log
    )

    pooled_es = pooled[pooled["country"] == "ES"].set_index("period_start_utc")["pred_pooled_full"]
    pooled_fr = pooled[pooled["country"] == "FR"].set_index("period_start_utc")["pred_pooled_full"]
    es["pred_pooled_full"] = es["period_start_utc"].map(pooled_es)
    fr["pred_pooled_full"] = fr["period_start_utc"].map(pooled_fr)

    # Referencias ya conocidas (ver RESULTS.md): (X)/(Y) solo-pais con
    # features comunes, (Z) pooled con features comunes.
    ref_es = pd.read_parquet(OUTPUT_DIR / "predicciones_es.parquet")[["period_start_utc", "pred_solo", "pred_pooled"]]
    ref_fr = pd.read_parquet(OUTPUT_DIR / "predicciones_fr.parquet")[["period_start_utc", "pred_solo", "pred_pooled"]]
    es = es.merge(ref_es, on="period_start_utc", how="left", suffixes=("", "_ref"))
    fr = fr.merge(ref_fr, on="period_start_utc", how="left", suffixes=("", "_ref"))

    log("\n-- España: comparación completa, mismo periodo --")
    mask_es = es["pred_solo"].notna() & es["pred_pooled"].notna() & es["pred_pooled_full"].notna()
    report(es, "pred_solo", "(X) España sola, features comunes", log, mask=mask_es)
    report(es, "pred_pooled", "(Z) España, pooled features comunes", log, mask=mask_es)
    report(es, "pred_pooled_full", "(Z2) España, pooled features COMPLETAS", log, mask=mask_es)

    log("\n-- Francia: comparación completa, mismo periodo --")
    mask_fr = fr["pred_solo"].notna() & fr["pred_pooled"].notna() & fr["pred_pooled_full"].notna()
    report(fr, "pred_solo", "(Y) Francia sola, features comunes", log, mask=mask_fr)
    report(fr, "pred_pooled", "(Z) Francia, pooled features comunes", log, mask=mask_fr)
    report(fr, "pred_pooled_full", "(Z2) Francia, pooled features COMPLETAS", log, mask=mask_fr)

    es.to_parquet(OUTPUT_DIR / "predicciones_es_full.parquet", index=False)
    fr.to_parquet(OUTPUT_DIR / "predicciones_fr_full.parquet", index=False)
    (OUTPUT_DIR / "model_full_summary.txt").write_text("\n".join(lines), encoding="utf-8")
    log("\nGuardado.")


if __name__ == "__main__":
    main()
