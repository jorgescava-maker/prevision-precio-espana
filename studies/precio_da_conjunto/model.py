"""
Experimento "más creativo" tras cerrar las líneas de España y Francia por
separado: un modelo POOLED entrenado con el histórico de AMBOS países a la
vez (con un indicador de país), comparado contra cada país entrenado por
separado con el MISMO conjunto de features (reducido a la intersección
común — sin `water_value`/`must_run`/motor de España, sin las versiones
v2-v4 del análogo, para que la comparación sea justa: aísla el efecto de
compartir datos, no el de tener más features).

Hipótesis: los dos mercados están acoplados en la misma subasta EUPHEMIA y
comparten meteorología continental/precio del gas — un modelo con más
historia total (los dos países) podría generalizar mejor, especialmente
para Francia (línea más nueva, menos madura) y para eventos raros
compartidos (p.ej. un Dunkelflaute europeo golpea a los dos a la vez).

Uso:
    .venv\\Scripts\\python.exe -m studies.precio_da_conjunto.model
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from etl.common import logging_config  # noqa: F401
from studies.precio_da_mejor_modelo.model import walk_forward, report, INITIAL_TRAIN_MONTHS

SPAIN_DIR = Path(__file__).resolve().parents[1] / "precio_da_mejor_modelo" / "output"
FRANCE_DIR = Path(__file__).resolve().parents[1] / "precio_da_francia" / "output"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"

# Features comunes a los dos paises (interseccion real de lo que ambos
# datasets tienen construido) -- se sacrifica la riqueza especifica de cada
# uno (water_value/must_run/motor de Espana, embalse de Francia) a proposito,
# para que la comparacion (X)/(Y) vs (Z) aisle el efecto de compartir datos.
# findings.md #103 (2026-09-01): `coal_eur_mwh_th` sin actualizar desde
# 2025-12-26 — excluida de producción, mismo motivo que en
# precio_da_mejor_modelo/model.py (recupera 2026 completo en la evaluación).
COMMON_FEATURES = [
    "forecast_demanda_mw", "forecast_eolica_mw", "forecast_solar_mw",
    "ttf_eur_mwh", "eua_eur_t",
    "nuclear_capacity_avail_mw", "gas_capacity_avail_mw", "coal_capacity_avail_mw",
    "analog_price_mean",
    "price_lag_24h", "price_lag_48h", "price_lag_168h", "price_ema_7d", "price_ema_28d",
    "price_boundary_prev_day",
    "hour_of_day", "period_in_hour", "period_of_day", "day_of_week", "month",
    "is_holiday", "is_weekend",
]


def load_spain() -> pd.DataFrame:
    df = pd.read_parquet(SPAIN_DIR / "dataset.parquet")
    df = df.rename(columns={"analog_price_mean_v5": "analog_price_mean"})
    df["is_holiday"] = df["is_holiday"].astype(int)
    df["is_weekend"] = df["is_weekend"].astype(int)
    df["country"] = "ES"
    return df[["period_start_utc", "price_real", "country"] + COMMON_FEATURES]


def load_france() -> pd.DataFrame:
    df = pd.read_parquet(FRANCE_DIR / "dataset.parquet")
    df["is_holiday"] = df["is_holiday"].astype(int)
    df["is_weekend"] = df["is_weekend"].astype(int)
    df["country"] = "FR"
    return df[["period_start_utc", "price_real", "country"] + COMMON_FEATURES]


def main() -> None:
    lines: list[str] = []

    def log(s: str = "") -> None:
        print(s)
        lines.append(s)

    log("=" * 72)
    log("Modelo conjunto España+Francia — features comunes")
    log("=" * 72)

    es = load_spain()
    fr = load_france()
    log(f"España: {len(es):,} periodos. Francia: {len(fr):,} periodos.")

    # -----------------------------------------------------------------
    log("\n-- (X) España sola, SOLO features comunes (referencia justa) --")
    es["pred_solo"] = walk_forward(es, COMMON_FEATURES, "price_real", "ES solo", log)

    log("\n-- (Y) Francia sola, SOLO features comunes --")
    fr["pred_solo"] = walk_forward(fr, COMMON_FEATURES, "price_real", "FR solo", log)

    # -----------------------------------------------------------------
    log("\n-- (Z) Pooled: ambos paises juntos, con indicador de pais --")
    pooled = pd.concat([es, fr], ignore_index=True)
    pooled["is_spain"] = (pooled["country"] == "ES").astype(int)
    feats_pooled = COMMON_FEATURES + ["is_spain"]
    pooled["pred_pooled"] = walk_forward(pooled, feats_pooled, "price_real", "pooled ES+FR", log)

    pooled_es = pooled[pooled["country"] == "ES"].set_index("period_start_utc")["pred_pooled"]
    pooled_fr = pooled[pooled["country"] == "FR"].set_index("period_start_utc")["pred_pooled"]
    es["pred_pooled"] = es["period_start_utc"].map(pooled_es)
    fr["pred_pooled"] = fr["period_start_utc"].map(pooled_fr)

    # -----------------------------------------------------------------
    log("\n-- Comparación ESPAÑA: solo vs. pooled, mismo periodo --")
    mask_es = es["pred_solo"].notna() & es["pred_pooled"].notna()
    report(es, "pred_solo", "(X) España sola (features comunes)", log, mask=mask_es)
    report(es, "pred_pooled", "(Z) España, modelo pooled ES+FR", log, mask=mask_es)

    log("\n-- Comparación FRANCIA: solo vs. pooled, mismo periodo --")
    mask_fr = fr["pred_solo"].notna() & fr["pred_pooled"].notna()
    report(fr, "pred_solo", "(Y) Francia sola (features comunes)", log, mask=mask_fr)
    report(fr, "pred_pooled", "(Z) Francia, modelo pooled ES+FR", log, mask=mask_fr)

    es.to_parquet(OUTPUT_DIR / "predicciones_es.parquet", index=False)
    fr.to_parquet(OUTPUT_DIR / "predicciones_fr.parquet", index=False)
    (OUTPUT_DIR / "model_summary.txt").write_text("\n".join(lines), encoding="utf-8")
    log("\nGuardado.")


if __name__ == "__main__":
    main()
