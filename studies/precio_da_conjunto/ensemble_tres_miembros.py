"""
Fase 27b (2026-09-02) — mezcla de TRES miembros para España: agrupado +
por-hora + redes (findings.md #110). No sustituye nada: añade
`pred_red_pooled` (`redes_espana.py`) al ensemble causal ya en producción
(`ensemble_pooled_hora.py`, dos miembros) y vuelve a aplicar encima el gate
de hurdle ya validado (Fase 24b), que es la arquitectura del campeón 10,53.

El peso se sigue eligiendo mes a mes con SOLO el histórico de test ya
observado (L8) — misma regla que con dos miembros, ahora sobre una rejilla
del símplex de tres pesos en pasos de 0,1 (66 combinaciones).

Uso:
    .venv\Scripts\python.exe -m studies.precio_da_conjunto.ensemble_tres_miembros

Requiere haber corrido antes ensemble_pooled_hora.py y redes_espana.py.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from etl.common import logging_config  # noqa: F401
from studies.precio_da_mejor_modelo.model import report
from studies.precio_da_mejor_modelo.hurdle_mas_ensemble import walk_forward_zero_gate
from studies.precio_da_mejor_modelo.hurdle_quantile import ZERO_THRESHOLD

UMBRAL_CALIBRACION = 0.80   # se calibra solo por encima de este cuantil del histórico


def calibrar_cola(d: pd.DataFrame, col: str, warmup: int, log) -> pd.Series:
    """Fase 29 — corrección del encogimiento hacia el centro, solo en la cola alta.

    El modelo se entrena minimizando error CUADRÁTICO (cuyo óptimo es la media
    condicional) y se evalúa con el ABSOLUTO (cuyo óptimo es la mediana), así
    que se encoge hacia el centro y se queda corto en los precios altos: −20,6
    EUR/MWh de sesgo por encima de 200. Aquí se aprende, con el histórico ya
    observado de cada mes, una recta `predicción -> precio` por regresión
    cuantílica a la MEDIANA, y se aplica SOLO a los periodos cuya predicción
    supera el cuantil `UMBRAL_CALIBRACION` de ese mismo histórico.

    Por qué solo en la cola y no en todo: aplicarla en todo el rango mejora el
    agregado menos y empeora la zona 0-50, donde no hay encogimiento que
    corregir. Y por qué sin encogimiento hacia la predicción original (lambda):
    elegir ese lambda con el histórico hace que el criterio diga "no calibrar"
    en 17 de 18 meses — la ganancia aparece en los meses de precio alto, que son
    justo los que el pasado no anticipa. La regla del umbral no necesita ese
    parámetro y sí se sostiene en el control independiente (Francia también
    mejora). Detalle en findings.md #114.
    """
    meses = sorted(d["year_month"].unique())
    out = pd.Series(np.nan, index=d.index)
    n_tocados = 0
    for i, m in enumerate(meses):
        if i < warmup:
            continue
        hist = d[d["year_month"].isin(meses[:i])].dropna(subset=[col, "price_real"])
        cur = d[d["year_month"] == m]
        if hist.empty or cur.empty:
            continue
        crudo = cur[col].to_numpy(float)
        umbral = float(hist[col].quantile(UMBRAL_CALIBRACION))
        try:
            ajuste = sm.QuantReg(hist["price_real"].to_numpy(float),
                                 sm.add_constant(hist[col].to_numpy(float))).fit(q=0.5)
            corregido = ajuste.params[0] + ajuste.params[1] * crudo
        except Exception:
            corregido = crudo
        out.loc[cur.index] = np.where(crudo > umbral, corregido, crudo)
        n_tocados += int((crudo > umbral).sum())
    log(f"  calibración de cola: {n_tocados:,} periodos corregidos "
        f"(por encima del P{int(UMBRAL_CALIBRACION * 100)} del histórico de cada mes)")
    return out


ROOT = Path(__file__).resolve().parents[2]
ES_OUTPUT = ROOT / "studies" / "precio_da_mejor_modelo" / "output"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
WARMUP_MONTHS = 6
STEP = 0.1


def _grid() -> list[tuple[float, float, float]]:
    n = int(round(1 / STEP))
    return [(i / n, j / n, (n - i - j) / n)
            for i in range(n + 1) for j in range(n + 1 - i)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--redes", default=str(OUTPUT_DIR / "predicciones_redes.parquet"),
                    help="parquet(s) con las predicciones de las redes, separados por coma")
    ap.add_argument("--miembro", default="pred_red_pooled+pred_red_es",
                    help="columna del tercer miembro; varias unidas por '+' se promedian")
    args = ap.parse_args()

    lines: list[str] = []

    def log(s: str = "") -> None:
        print(s, flush=True)
        lines.append(s)

    log("=" * 72)
    log("Fase 27b — ensemble de tres miembros (agrupado + por-hora + redes) + gate de hurdle")
    log("=" * 72)

    dos = pd.read_parquet(OUTPUT_DIR / "predicciones_ensemble_conjunto.parquet")[
        ["period_start_utc", "country", "price_real", "pred_pooled", "pred_por_hora", "pred_ensemble"]
    ]
    # El tercer miembro puede ser una columna suelta o el PROMEDIO de varias
    # (variantes de red que solo se diferencian en la configuración): combinarlas
    # es en sí mismo un ensemble y da mejor resultado que cualquiera por separado
    # (Fase 27, ver evaluar_miembro_redes.py).
    redes = None
    for f in args.redes.split(","):
        parte = pd.read_parquet(f.strip())
        if redes is None:
            redes = parte
        else:
            nuevas = [c for c in parte.columns if c not in redes.columns]
            redes = redes.merge(parte[["period_start_utc", "country"] + nuevas],
                                on=["period_start_utc", "country"], how="outer")
    partes = [c.strip() for c in args.miembro.split("+")]
    redes["pred_red_pooled"] = redes[partes].mean(axis=1)
    log(f"Tercer miembro: {' + '.join(partes)}")
    d = dos.merge(redes[["period_start_utc", "country", "pred_red_pooled"]],
                  on=["period_start_utc", "country"], how="inner")
    d = d.dropna(subset=["pred_pooled", "pred_por_hora", "pred_red_pooled", "price_real"])
    d = d.sort_values("period_start_utc").copy()
    d["year_month"] = d["period_start_utc"].dt.to_period("M")
    months = sorted(d["year_month"].unique())
    log(f"Periodos con los tres miembros disponibles: {len(d):,} "
        f"({d['country'].value_counts().to_dict()})")

    r_pool = d["pred_pooled"] - d["price_real"]
    r_hora = d["pred_por_hora"] - d["price_real"]
    r_red = d["pred_red_pooled"] - d["price_real"]
    log(f"\nCorrelación de residuos — agrupado/por-hora: {r_pool.corr(r_hora):.4f} · "
        f"agrupado/redes: {r_pool.corr(r_red):.4f} · por-hora/redes: {r_hora.corr(r_red):.4f}")
    log("(cuanto más baja la correlación con las redes, más puede aportar el tercer miembro)")

    grid = _grid()

    def _mejor_peso(hist: pd.DataFrame) -> tuple[float, float, float]:
        best, best_mae = None, np.inf
        for w in grid:
            mix = (w[0] * hist["pred_pooled"] + w[1] * hist["pred_por_hora"]
                   + w[2] * hist["pred_red_pooled"])
            mae = (hist["price_real"] - mix).abs().mean()
            if mae < best_mae:
                best, best_mae = w, mae
        return best

    def _aplicar(cur: pd.DataFrame, w: tuple[float, float, float]) -> pd.Series:
        return (w[0] * cur["pred_pooled"] + w[1] * cur["pred_por_hora"]
                + w[2] * cur["pred_red_pooled"])

    d["pred_ensemble3"] = np.nan
    # Variante: el peso mensual elegido con el histórico DEL PROPIO PAÍS en vez
    # del conjunto ES+FR. El ensemble de dos miembros en producción usa el
    # histórico conjunto; con un tercer miembro cuyo aporte puede ser distinto
    # en cada país, conviene medir si separar el peso ayuda (y, sobre todo, que
    # no degrade al otro país — feedback_no_degradar_modelo_compartido).
    d["pred_ensemble3_pais"] = np.nan
    pesos_log: list[dict] = []
    for i, m in enumerate(months):
        if i < WARMUP_MONTHS:
            continue
        hist = d[d["year_month"].isin(months[:i])]
        cur = d[d["year_month"] == m]
        best = _mejor_peso(hist)
        d.loc[cur.index, "pred_ensemble3"] = _aplicar(cur, best)
        fila = {"mes": str(m), "w_pooled": best[0], "w_hora": best[1], "w_red": best[2]}
        for pais in ("ES", "FR"):
            hist_p = hist[hist["country"] == pais]
            cur_p = cur[cur["country"] == pais]
            if hist_p.empty or cur_p.empty:
                continue
            w_p = _mejor_peso(hist_p)
            d.loc[cur_p.index, "pred_ensemble3_pais"] = _aplicar(cur_p, w_p)
            fila.update({f"w_pooled_{pais}": w_p[0], f"w_hora_{pais}": w_p[1], f"w_red_{pais}": w_p[2]})
        pesos_log.append(fila)

    if pesos_log:
        wl = pd.DataFrame(pesos_log)
        log(f"\nPesos elegidos (media sobre {len(wl)} meses): agrupado={wl['w_pooled'].mean():.2f} · "
            f"por-hora={wl['w_hora'].mean():.2f} · redes={wl['w_red'].mean():.2f}")
        log(f"  peso de las redes: mín={wl['w_red'].min():.1f} máx={wl['w_red'].max():.1f} "
            f"· meses con peso 0: {(wl['w_red'] == 0).sum()}/{len(wl)}")
        for pais in ("ES", "FR"):
            if f"w_red_{pais}" in wl.columns:
                log(f"  peso por país {pais}: agrupado={wl[f'w_pooled_{pais}'].mean():.2f} · "
                    f"por-hora={wl[f'w_hora_{pais}'].mean():.2f} · redes={wl[f'w_red_{pais}'].mean():.2f}")

    log("\n-- Calibración de la cola alta sobre la mezcla de tres (Fase 29) --")
    d["pred_ensemble3_cal"] = calibrar_cola(d, "pred_ensemble3", WARMUP_MONTHS, log)

    valid = d.dropna(subset=["pred_ensemble3", "pred_ensemble3_pais", "pred_ensemble3_cal"]).copy()
    for pais in ("ES", "FR"):
        sub = valid[valid["country"] == pais]
        if sub.empty:
            continue
        log(f"\n-- {pais}: sin gate de hurdle, mismo periodo ({len(sub):,} periodos) --")
        report(sub, "pred_pooled", "  (a) solo agrupado", log)
        report(sub, "pred_por_hora", "  (b) solo por-hora", log)
        report(sub, "pred_red_pooled", "  (c) solo redes", log)
        report(sub, "pred_ensemble", "  (d) ensemble de 2 (producción)", log)
        report(sub, "pred_ensemble3", "  (e) ensemble de 3, peso conjunto ES+FR", log)
        report(sub, "pred_ensemble3_pais", "  (f) ensemble de 3, peso por país", log)
        report(sub, "pred_ensemble3_cal", "  (g) ensemble de 3 + calibración de cola", log)

    # ---- gate de hurdle encima, igual que Fase 24b (arquitectura del campeón)
    log("\n-- Gate de hurdle sobre España (mismo clasificador causal de Fase 7/24) --")
    dataset = pd.read_parquet(ES_OUTPUT / "dataset.parquet")
    dataset["is_holiday"] = dataset["is_holiday"].astype(int)
    dataset["is_weekend"] = dataset["is_weekend"].astype(int)
    gate = walk_forward_zero_gate(dataset, log)

    es = valid[valid["country"] == "ES"].merge(gate, on="period_start_utc", how="inner")
    es = es.dropna(subset=["pred_zero"]).copy()
    es["pred_ens2_hurdle"] = np.where(es["pred_zero"], es["zero_fill"], es["pred_ensemble"])
    es["pred_ens3_hurdle"] = np.where(es["pred_zero"], es["zero_fill"], es["pred_ensemble3"])
    es["pred_ens3_pais_hurdle"] = np.where(es["pred_zero"], es["zero_fill"], es["pred_ensemble3_pais"])
    es["pred_final"] = np.where(es["pred_zero"], es["zero_fill"], es["pred_ensemble3_cal"])

    log(f"\n-- España, arquitectura completa ({len(es):,} periodos) --")
    results = [
        report(es, "pred_ensemble", "(i) ensemble de 2, sin gate", log),
        report(es, "pred_ens2_hurdle", "(ii) ensemble de 2 + gate  <- campeón de 2 miembros (10,50 en su ventana propia)", log),
        report(es, "pred_ensemble3", "(iii) ensemble de 3, sin gate", log),
        report(es, "pred_ens3_hurdle", "(iv) ensemble de 3 + gate, peso conjunto", log),
        report(es, "pred_ens3_pais_hurdle", "(v) ensemble de 3 + gate, peso por país", log),
        report(es, "pred_final", "(vi) + calibración de cola  <- MODELO DE PRODUCCIÓN", log),
    ]

    zero = es[es["price_real"] <= ZERO_THRESHOLD]
    pos = es[es["price_real"] > ZERO_THRESHOLD]
    log("\n-- Control de robustez por tramo (el criterio de selección sigue siendo el MAE agregado) --")
    for tag, sub in (("precio<=0 real", zero), ("precio>0 real", pos)):
        if sub.empty:
            continue
        m2 = (sub["price_real"] - sub["pred_ens2_hurdle"]).abs().mean()
        m3 = (sub["price_real"] - sub["pred_ens3_hurdle"]).abs().mean()
        mf = (sub["price_real"] - sub["pred_final"]).abs().mean()
        log(f"  {tag} (n={len(sub):,}): 2 miembros={m2:.2f} · 3 miembros={m3:.2f} · producción={mf:.2f}")

    p95 = es["price_real"].quantile(0.95)
    alto = es[es["price_real"] >= p95]
    log(f"  precio >= P95 ({p95:.1f} EUR/MWh, n={len(alto):,}): "
        f"2 miembros={(alto['price_real'] - alto['pred_ens2_hurdle']).abs().mean():.2f} · "
        f"3 miembros={(alto['price_real'] - alto['pred_ens3_hurdle']).abs().mean():.2f} · "
        f"producción={(alto['price_real'] - alto['pred_final']).abs().mean():.2f}")

    pd.DataFrame(results).to_parquet(OUTPUT_DIR / "ensemble_tres_miembros_summary.parquet", index=False)
    es.to_parquet(OUTPUT_DIR / "predicciones_fase27b.parquet", index=False)
    if pesos_log:
        pd.DataFrame(pesos_log).to_csv(OUTPUT_DIR / "ensemble_tres_miembros_pesos.csv", index=False)
    (OUTPUT_DIR / "ensemble_tres_miembros_summary.txt").write_text("\n".join(lines), encoding="utf-8")
    log("\nGuardado.")


if __name__ == "__main__":
    main()
