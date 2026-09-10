"""
Backtest de España con el valor del agua CAUSAL (findings.md #146, 2026-09-10).

El problema: `merit_order/model.py::build_water_value_feature` resuelve la DP de
C3 una sola vez con TODA la serie semanal y aplica esa superficie a todas las
filas, así que en el walk-forward una fila de test de 2025 lleva un valor del
agua calculado con precios y aportaciones de 2025-2026. De esa variable
dependen también el precio simulado del motor (la hidráulica entra como tramo
de coste) y los análogos v4/v5, que la usan para emparejar.

Lo que hace este script es lo que habría hecho la operación real: para cada
mes de test, la superficie se estima SOLO con las semanas ya cerradas y
publicadas antes de que empiece el mes, y con ella se recalculan, para TODAS
las filas de ese paso (entrenamiento y test), el valor del agua, el despacho y
el dataset de España entero (mismo `build()` de producción, cambiando solo las
columnas del motor). Después se repiten exactamente los bucles de producción
— agrupado ES+FR, por hora, las dos variantes de redes, filtro de precio<=0,
pesos de la mezcla y calibración de la cola — cada paso con su dataset.

Francia no cambia: sus filas en el modelo conjunto solo llevan variables
comunes y el embalse en bruto (`load_france_full`), ninguna depende de la DP.

Uso:
    python -m studies.precio_da_conjunto.agua_causal validar      # reproduce producción con la superficie completa
    python -m studies.precio_da_conjunto.agua_causal datasets     # un dataset por mes de corte (reanudable)
    python -m studies.precio_da_conjunto.agua_causal walkforward  # backtest causal y comparación
"""

from __future__ import annotations

import contextlib
import io
import shutil
import sys
import tempfile
import time
from pathlib import Path

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd

from etl.common import logging_config  # noqa: F401 - fuerza stdout a UTF-8
import studies.precio_da_mejor_modelo.build_dataset as es_build
from studies.c3_valor_agua.model import OUTPUT_DIR as C3_OUTPUT, load_scenarios, solve_dp, water_value_surface
from studies.merit_order.build_dataset import (
    DATA_DIR, GAS_HEAT_RATE_RANGE, COAL_HEAT_RATE_RANGE, HEAT_RATE_GAS, EMISSION_FACTOR_GAS,
    EMISSION_FACTOR_COAL, _build_unit_tiers, _load_generation_units,
)
from studies.merit_order.model import (
    OUTPUT_DIR as MO_OUTPUT, _filling_and_week, adjust_tiers_for_outage, hourly_unavailable_mw,
)
from studies.precio_da_mejor_modelo.model import LGB_PARAMS, INITIAL_TRAIN_MONTHS, report
from studies.precio_da_mejor_modelo.hurdle_quantile import FEATURES as GATE_FEATURES, ZERO_THRESHOLD, best_threshold_f1
from studies.precio_da_conjunto.model import COMMON_FEATURES
from studies.precio_da_conjunto.model_full import SPAIN_ONLY, FRANCE_ONLY, load_france_full
from studies.precio_da_conjunto.redes_espana import _fit_predict_ensemble
from studies.precio_da_conjunto.ensemble_pooled_hora import WARMUP_MONTHS as WARMUP_2, PESOS_CANDIDATOS, MIN_ROWS_PER_GROUP
from studies.precio_da_conjunto.ensemble_tres_miembros import WARMUP_MONTHS as WARMUP_3, _grid, calibrar_cola

OUTPUT_DIR = Path(__file__).resolve().parent / "output"
CAUSAL_DIR = OUTPUT_DIR / "agua_causal"
ES_DATASET = es_build.OUTPUT_DIR / "dataset.parquet"
SEMILLAS = list(range(42, 47))   # las mismas que el walk-forward de producción
# Redes ajustadas en paralelo. Con 5 (lo de producción) el proceso se quedó sin
# memoria en un equipo de 16 GB: cada proceso hijo recarga pandas/sklearn. El
# número de procesos no cambia el resultado (semillas fijas, mismo promedio).
JOBS_REDES = 3
# Una semana del embalse se da por conocida cuando ha terminado Y ha pasado otra
# semana más: ENTSO-E publica el llenado semanal (A72) días después de cerrarla.
RETRASO_SEMANA = pd.Timedelta(days=14)
COLS_MOTOR = ["water_value_eur_mwh", "price_simulado_final", "tecnologia_marginal_final"]


class Motor:
    """Lo que no depende de la superficie del agua, calculado una sola vez."""

    def __init__(self) -> None:
        self.base = pd.read_parquet(MO_OUTPUT / "predicciones_tecnologia.parquet")
        self.filling, self.iso_week, self.valid = _filling_and_week(self.base)
        con = duckdb.connect(str(DATA_DIR / "spain.duckdb"), read_only=True)
        try:
            units = _load_generation_units(con)
            horas = pd.DatetimeIndex(self.base["hour_utc"])
            self.gas_unavail = hourly_unavailable_mw(con, ["B04"], horas).to_numpy()
            self.coal_unavail = hourly_unavailable_mw(con, ["B02", "B03", "B05", "B06"], horas).to_numpy()
        finally:
            con.close()
        self.gas_tiers = _build_unit_tiers(units, "B04", GAS_HEAT_RATE_RANGE)
        self.coal_tiers = _build_unit_tiers(units, "B05", COAL_HEAT_RATE_RANGE)
        self.weekly = pd.read_parquet(C3_OUTPUT / "weekly_series.parquet")
        self.filas = self.base[["demand_to_cover_mw", "must_run_final_mw", "cogeneracion_resto_clim",
                                "capacity_year", "ttf_eur_mwh", "eua_eur_t", "coal_eur_mwh_th"]].to_dict("records")

    def superficie(self, corte: pd.Timestamp | None):
        """`corte=None` reproduce producción (toda la serie, sin filtrar)."""
        w = self.weekly if corte is None else self.weekly[self.weekly["week"] + RETRASO_SEMANA <= corte]
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "weekly.parquet"
            w.to_parquet(p, index=False)
            _, esc, max_s, turb = load_scenarios(p)
        V, _, grid = solve_dp(esc, max_s, turb, lambda s: None)
        return water_value_surface(V, grid), grid, turb * 1000.0 / 168.0, len(w)

    def valor_agua(self, surf: np.ndarray, grid: np.ndarray) -> np.ndarray:
        wv = np.full(len(self.base), np.nan)
        v = self.valid
        wv[v] = np.array([np.interp(f, grid, surf[w - 1]) for f, w in zip(self.filling[v], self.iso_week[v])])
        return wv

    def _fila(self, r, water_value_h, gas_unavail_h, coal_unavail_h, turbine_cap_mw):
        """Copia literal de `dispatch_row_combined` (merit_order/model.py::main),
        que es una función local y no se puede importar."""
        remaining = r["demand_to_cover_mw"] - r["must_run_final_mw"]
        if pd.isna(remaining):
            return np.nan, "sin_datos"
        if remaining <= 0:
            return 0.0, "renovable_nuclear"
        gas_tiers = adjust_tiers_for_outage(self.gas_tiers.get(r["capacity_year"], []), gas_unavail_h)
        coal_tiers = adjust_tiers_for_outage(self.coal_tiers.get(r["capacity_year"], []), coal_unavail_h)
        tiers = []
        cogen_qty, ttf, eua, coal_th = r["cogeneracion_resto_clim"], r["ttf_eur_mwh"], r["eua_eur_t"], r["coal_eur_mwh_th"]
        if not pd.isna(cogen_qty) and cogen_qty > 0 and not pd.isna(ttf) and not pd.isna(eua):
            tiers.append((HEAT_RATE_GAS * (ttf + EMISSION_FACTOR_GAS * eua), cogen_qty, "cogeneracion"))
        if not pd.isna(ttf) and not pd.isna(eua):
            for hr, qty in gas_tiers:
                tiers.append((hr * (ttf + EMISSION_FACTOR_GAS * eua), qty, "gas"))
        if not pd.isna(coal_th) and not pd.isna(eua):
            for hr, qty in coal_tiers:
                tiers.append((hr * (coal_th + EMISSION_FACTOR_COAL * eua), qty, "carbon"))
        if not pd.isna(water_value_h):
            tiers.append((water_value_h, turbine_cap_mw, "hidraulica_gestionable"))
        tiers.sort(key=lambda t: t[0])
        for cost, qty, label in tiers:
            if remaining <= qty:
                return cost, label
            remaining -= qty
        if tiers:
            return tiers[-1][0], "shortfall_" + tiers[-1][2]
        return np.nan, "sin_tiers"

    def motor(self, corte: pd.Timestamp | None) -> tuple[pd.DataFrame, int]:
        surf, grid, tcap, n_sem = self.superficie(corte)
        wv = self.valor_agua(surf, grid)
        res = [self._fila(r, wv[i], self.gas_unavail[i], self.coal_unavail[i], tcap) for i, r in enumerate(self.filas)]
        out = pd.DataFrame(res, columns=["price_simulado_final", "tecnologia_marginal_final"])
        out.insert(0, "water_value_eur_mwh", wv)
        out.insert(0, "hour_utc", self.base["hour_utc"].to_numpy())
        return out, n_sem

    def dataset(self, corte: pd.Timestamp | None) -> tuple[pd.DataFrame, int]:
        """Dataset de España completo con el motor de este corte: el `build()`
        de producción, cambiando solo lo que sale de la superficie del agua."""
        motor, n_sem = self.motor(corte)
        original, salida = es_build._load_merit_order_base, es_build.OUTPUT_DIR

        def base_con_motor():
            df = original()
            m = motor.set_index("hour_utc").loc[df["hour_utc"], COLS_MOTOR]
            for c in COLS_MOTOR:
                df[c] = m[c].to_numpy()
            return df

        with tempfile.TemporaryDirectory() as tmp:
            es_build._load_merit_order_base, es_build.OUTPUT_DIR = base_con_motor, Path(tmp)
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    df = es_build.build()
            finally:
                es_build._load_merit_order_base, es_build.OUTPUT_DIR = original, salida
        return df, n_sem


# ---------------------------------------------------------------------------
# Walk-forward de producción, un paso por mes, cada paso con su dataset
# ---------------------------------------------------------------------------

TODAS = list(dict.fromkeys(COMMON_FEATURES + SPAIN_ONLY + FRANCE_ONLY + ["is_spain"]))
POR_HORA = [f for f in TODAS if f not in ("hour_of_day", "period_of_day")]
SOLO_ES = [f for f in TODAS if f not in FRANCE_ONLY + ["is_spain"]]


def _es_para_conjunto(df: pd.DataFrame) -> pd.DataFrame:
    """Lo mismo que `model_full.load_spain_full`, sobre un dataset en memoria."""
    df = df.rename(columns={"analog_price_mean_v5": "analog_price_mean"})
    df["is_holiday"] = df["is_holiday"].astype(int)
    df["is_weekend"] = df["is_weekend"].astype(int)
    df["country"] = "ES"
    return df[["period_start_utc", "price_real", "country"] + COMMON_FEATURES + SPAIN_ONLY].copy()


def _es_para_filtro(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["is_holiday"] = df["is_holiday"].astype(int)
    df["is_weekend"] = df["is_weekend"].astype(int)
    return df


def _meses(df: pd.DataFrame, requeridas: list[str]) -> list[pd.Period]:
    d = df.dropna(subset=requeridas + ["price_real"])
    return sorted(d["period_start_utc"].dt.to_period("M").unique())


def paso(mes: pd.Period, es_raw: pd.DataFrame, fr: pd.DataFrame, hacer_n1: bool, hacer_n2: bool,
         hacer_filtro: bool) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Un mes de test de los tres bucles de producción con el dataset `es_raw`."""
    pooled = pd.concat([_es_para_conjunto(es_raw), fr], ignore_index=True)
    pooled["is_spain"] = (pooled["country"] == "ES").astype(int)
    d = pooled.dropna(subset=COMMON_FEATURES + ["price_real"]).copy()
    d["ym"] = d["period_start_utc"].dt.to_period("M")
    train, test = d[d["ym"] < mes], d[d["ym"] == mes]
    out = test[["period_start_utc", "country", "price_real"]].copy()

    if hacer_n1:
        reg = lgb.LGBMRegressor(**LGB_PARAMS)
        reg.fit(train[TODAS], train["price_real"])
        out["pred_pooled"] = reg.predict(test[TODAS])
        out["pred_por_hora"] = np.nan
        for h, test_h in test.groupby("hour_of_day"):
            train_h = train[train["hour_of_day"] == h]
            if len(train_h) < MIN_ROWS_PER_GROUP:
                continue
            reg_h = lgb.LGBMRegressor(**LGB_PARAMS)
            reg_h.fit(train_h[POR_HORA], train_h["price_real"])
            out.loc[test_h.index, "pred_por_hora"] = reg_h.predict(test_h[POR_HORA])
        out["pred_red_n1"] = _fit_predict_ensemble(train[TODAS], train["price_real"].to_numpy(float),
                                                   test[TODAS], SEMILLAS, n_jobs=JOBS_REDES)
    out["pred_red_n2"] = np.nan
    if hacer_n2:
        tr_es, te_es = train[train["country"] == "ES"], test[test["country"] == "ES"]
        if len(te_es):
            out.loc[te_es.index, "pred_red_n2"] = _fit_predict_ensemble(
                tr_es[SOLO_ES], tr_es["price_real"].to_numpy(float), te_es[SOLO_ES], SEMILLAS, n_jobs=JOBS_REDES)

    filtro = None
    if hacer_filtro:
        g = _es_para_filtro(es_raw).dropna(subset=GATE_FEATURES + ["price_real"]).copy()
        g["ym"] = g["period_start_utc"].dt.to_period("M")
        g["is_zero"] = (g["price_real"] <= ZERO_THRESHOLD).astype(int)
        tr, te = g[g["ym"] < mes], g[g["ym"] == mes]
        if len(te) and len(tr) and tr["is_zero"].nunique() >= 2:
            clf = lgb.LGBMClassifier(**LGB_PARAMS)
            clf.fit(tr[GATE_FEATURES], tr["is_zero"])
            thr = best_threshold_f1(clf.predict_proba(tr[GATE_FEATURES])[:, 1], tr["is_zero"].to_numpy())
            zf = tr.loc[tr["is_zero"] == 1, "price_real"].mean()
            filtro = pd.DataFrame({"period_start_utc": te["period_start_utc"].to_numpy(),
                                   "pred_zero": clf.predict_proba(te[GATE_FEATURES])[:, 1] >= thr,
                                   "zero_fill": 0.0 if pd.isna(zf) else zf})
    return out, filtro


def ensamblar(preds: pd.DataFrame, filtro: pd.DataFrame, fechas_es: pd.Series, log) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pesos de dos miembros, pesos de tres, calibración de la cola y filtro:
    la misma lógica de `ensemble_pooled_hora.py` y `ensemble_tres_miembros.py`."""
    d = preds.dropna(subset=["pred_pooled", "pred_por_hora"]).sort_values("period_start_utc").copy()
    d["year_month"] = d["period_start_utc"].dt.to_period("M")
    meses = sorted(d["year_month"].unique())
    d["pred_ensemble"] = np.nan
    for i, m in enumerate(meses):
        if i < WARMUP_2:
            continue
        hist, cur = d[d["year_month"].isin(meses[:i])], d[d["year_month"] == m]
        maes = [(hist["price_real"] - (w * hist["pred_pooled"] + (1 - w) * hist["pred_por_hora"])).abs().mean()
                for w in PESOS_CANDIDATOS]
        w = PESOS_CANDIDATOS[int(np.argmin(maes))]
        d.loc[cur.index, "pred_ensemble"] = w * cur["pred_pooled"] + (1 - w) * cur["pred_por_hora"]
    d = d.dropna(subset=["pred_ensemble"])

    d["pred_red_pooled"] = d[["pred_red_n1", "pred_red_n2"]].mean(axis=1)
    d = d.dropna(subset=["pred_pooled", "pred_por_hora", "pred_red_pooled", "price_real"]).copy()
    meses = sorted(d["year_month"].unique())
    grid = _grid()

    def mejor(h):
        maes = [(h["price_real"] - (w[0] * h["pred_pooled"] + w[1] * h["pred_por_hora"] + w[2] * h["pred_red_pooled"])).abs().mean()
                for w in grid]
        return grid[int(np.argmin(maes))]

    def aplicar(c, w):
        return w[0] * c["pred_pooled"] + w[1] * c["pred_por_hora"] + w[2] * c["pred_red_pooled"]

    d["pred_ensemble3"] = np.nan
    d["pred_ensemble3_pais"] = np.nan
    pesos = []
    for i, m in enumerate(meses):
        if i < WARMUP_3:
            continue
        hist, cur = d[d["year_month"].isin(meses[:i])], d[d["year_month"] == m]
        w = mejor(hist)
        d.loc[cur.index, "pred_ensemble3"] = aplicar(cur, w)
        fila = {"mes": str(m), "w_pooled": w[0], "w_hora": w[1], "w_red": w[2]}
        for pais in ("ES", "FR"):
            hp, cp = hist[hist["country"] == pais], cur[cur["country"] == pais]
            if hp.empty or cp.empty:
                continue
            wp = mejor(hp)
            d.loc[cp.index, "pred_ensemble3_pais"] = aplicar(cp, wp)
            fila.update({f"w_pooled_{pais}": wp[0], f"w_hora_{pais}": wp[1], f"w_red_{pais}": wp[2]})
        pesos.append(fila)
    d["pred_ensemble3_cal"] = calibrar_cola(d, "pred_ensemble3", WARMUP_3, log)
    valid = d.dropna(subset=["pred_ensemble3", "pred_ensemble3_pais", "pred_ensemble3_cal"])

    gate = pd.DataFrame({"period_start_utc": fechas_es.to_numpy(), "pred_zero": False, "zero_fill": np.nan})
    if filtro is not None and len(filtro):
        gate = gate.set_index("period_start_utc")
        f = filtro.set_index("period_start_utc")
        gate.loc[f.index, "pred_zero"] = f["pred_zero"].to_numpy()
        gate.loc[f.index, "zero_fill"] = f["zero_fill"].to_numpy()
        gate = gate.reset_index()
    es = valid[valid["country"] == "ES"].merge(gate, on="period_start_utc", how="inner")
    es["pred_final"] = np.where(es["pred_zero"], es["zero_fill"], es["pred_ensemble3_cal"])
    return es, pd.DataFrame(pesos)


# ---------------------------------------------------------------------------

def _log_factory(lines: list[str]):
    def log(s: str = "") -> None:
        print(s, flush=True)
        lines.append(s)
    return log


def _meses_de_test() -> tuple[list, list, list]:
    """Los meses de test de cada bucle, tal como los recorre producción."""
    es_full = pd.read_parquet(ES_DATASET)
    fr = load_france_full()
    pooled = pd.concat([_es_para_conjunto(es_full), fr], ignore_index=True)
    m_pool = _meses(pooled, COMMON_FEATURES)[INITIAL_TRAIN_MONTHS:]
    m_es = _meses(_es_para_conjunto(es_full), COMMON_FEATURES)[INITIAL_TRAIN_MONTHS:]
    m_gate = _meses(_es_para_filtro(es_full), GATE_FEATURES)[INITIAL_TRAIN_MONTHS:]
    return m_pool, m_es, m_gate


def validar(log) -> None:
    log("-- Validación: con la superficie completa hay que reproducir producción --")
    t0 = time.time()
    motor = Motor()
    m, _ = motor.motor(None)
    b = motor.base
    for c in COLS_MOTOR:
        if c == "tecnologia_marginal_final":
            ok = (m[c].astype(str).to_numpy() == b[c].astype(str).to_numpy()).mean()
        else:
            ok = np.isclose(m[c].to_numpy(float), b[c].to_numpy(float), equal_nan=True, atol=1e-9).mean()
        log(f"  motor · {c}: {ok:.4%} idéntico")
    ds, _ = motor.dataset(None)
    ref = pd.read_parquet(ES_DATASET)
    malas = []
    for c in ref.columns:
        a, x = ref[c], ds[c]
        if pd.api.types.is_numeric_dtype(a) and not pd.api.types.is_bool_dtype(a):
            ok = np.isclose(a.to_numpy(float), x.to_numpy(float), equal_nan=True, atol=1e-9).all()
        else:
            ok = (a.astype(str).to_numpy() == x.astype(str).to_numpy()).all()
        if not ok:
            malas.append(c)
    log(f"  dataset: {len(ds):,} filas · columnas distintas de producción: {malas or 'ninguna'} ({time.time() - t0:.0f}s)")

    m_pool, m_es, m_gate = _meses_de_test()
    mes = m_pool[20]
    out, _ = paso(mes, ds, load_france_full(), True, mes in m_es, False)
    prod = pd.read_parquet(OUTPUT_DIR / "predicciones_ensemble_conjunto.parquet")
    redes = pd.read_parquet(OUTPUT_DIR / "predicciones_redes.parquet")
    j = out.merge(prod[["period_start_utc", "country", "pred_pooled", "pred_por_hora"]], on=["period_start_utc", "country"],
                  suffixes=("", "_prod")).merge(redes[["period_start_utc", "country", "pred_red_pooled", "pred_red_es"]],
                                                on=["period_start_utc", "country"])
    for a, b2 in [("pred_pooled", "pred_pooled_prod"), ("pred_por_hora", "pred_por_hora_prod"),
                  ("pred_red_n1", "pred_red_pooled"), ("pred_red_n2", "pred_red_es")]:
        ok = np.isclose(j[a], j[b2], equal_nan=True, atol=1e-6).mean()
        log(f"  paso {mes} · {a}: {ok:.4%} idéntico a producción ({len(j):,} filas)")


def datasets(log) -> None:
    m_pool, m_es, m_gate = _meses_de_test()
    meses = sorted(set(m_pool) | set(m_es) | set(m_gate))
    CAUSAL_DIR.mkdir(parents=True, exist_ok=True)
    log(f"-- {len(meses)} meses de corte: {meses[0]} → {meses[-1]} --")
    motor = Motor()
    for mes in meses:
        destino = CAUSAL_DIR / f"dataset_{mes}.parquet"
        if destino.exists():
            continue
        t0 = time.time()
        corte = mes.to_timestamp()
        ds, n_sem = motor.dataset(corte)
        ds = ds[ds["period_start_utc"] < (mes + 1).to_timestamp()]
        ds.to_parquet(destino, index=False)
        log(f"  {mes}: {n_sem} semanas de historia · valor del agua medio del mes "
            f"{ds.loc[ds['period_start_utc'] >= corte, 'water_value_eur_mwh'].mean():.1f} EUR/MWh ({time.time() - t0:.0f}s)")


def walkforward(log) -> None:
    m_pool, m_es, m_gate = _meses_de_test()
    meses = sorted(set(m_pool) | set(m_es) | set(m_gate))
    fr = load_france_full()
    partes, filtros = [], []
    t0 = time.time()
    for mes in meses:
        # Cada mes se guarda al terminar: si el proceso se corta (le pasó por
        # falta de memoria), al relanzar se retoma donde iba.
        f_paso, f_filtro = CAUSAL_DIR / f"paso_{mes}.parquet", CAUSAL_DIR / f"filtro_{mes}.parquet"
        if f_paso.exists():
            out = pd.read_parquet(f_paso)
            filtro = pd.read_parquet(f_filtro) if f_filtro.exists() else None
        else:
            es_raw = pd.read_parquet(CAUSAL_DIR / f"dataset_{mes}.parquet")
            t1 = time.time()
            out, filtro = paso(mes, es_raw, fr, mes in m_pool, mes in m_es, mes in m_gate)
            if filtro is not None:
                filtro.to_parquet(f_filtro, index=False)
            out.to_parquet(f_paso, index=False)
            del es_raw
            log(f"  {mes}: {len(out):,} filas ({time.time() - t1:.0f}s, acumulado {(time.time() - t0) / 60:.1f} min)")
        partes.append(out)
        if filtro is not None:
            filtros.append(filtro)
    preds = pd.concat(partes, ignore_index=True)
    filtro = pd.concat(filtros, ignore_index=True) if filtros else None
    fechas_es = pd.read_parquet(ES_DATASET, columns=["period_start_utc"])["period_start_utc"]
    es, pesos = ensamblar(preds, filtro, fechas_es, log)

    es.to_parquet(OUTPUT_DIR / "predicciones_agua_causal.parquet", index=False)
    pesos.to_csv(OUTPUT_DIR / "agua_causal_pesos.csv", index=False)

    report(es, "pred_final", "\n  España, valor del agua causal", log)
    if not (OUTPUT_DIR / "predicciones_fase27b.parquet").exists():
        return
    log("\n-- España: valor del agua causal frente a producción (superficie completa) --")
    prod = pd.read_parquet(OUTPUT_DIR / "predicciones_fase27b.parquet")[["period_start_utc", "price_real", "pred_final"]]
    j = es.merge(prod, on="period_start_utc", suffixes=("", "_prod"))
    log(f"  periodos: causal {len(es):,} · producción {len(prod):,} · comunes {len(j):,}")
    report(es, "pred_final", "  causal, ventana propia", log)
    mae_c = (j["price_real"] - j["pred_final"]).abs().mean()
    mae_p = (j["price_real_prod"] - j["pred_final_prod"]).abs().mean()
    log(f"  mismos periodos: producción {mae_p:.3f} · causal {mae_c:.3f} · diferencia {mae_c - mae_p:+.3f} EUR/MWh")
    j["mes"] = j["period_start_utc"].dt.to_period("M")
    t = j.groupby("mes").apply(lambda g: pd.Series({
        "prod": (g["price_real_prod"] - g["pred_final_prod"]).abs().mean(),
        "causal": (g["price_real"] - g["pred_final"]).abs().mean()}), include_groups=False)
    t["dif"] = t["causal"] - t["prod"]
    log(t.to_string(float_format=lambda x: f"{x:.2f}"))


def main() -> int:
    modo = sys.argv[1] if len(sys.argv) > 1 else "validar"
    lines: list[str] = []
    log = _log_factory(lines)
    if modo == "datasets" and "--rehacer" in sys.argv[2:] and CAUSAL_DIR.exists():
        # Los datasets de cada corte dependen de los datos del día: en un
        # refresco hay que rehacerlos, no reutilizar los de la vez anterior.
        shutil.rmtree(CAUSAL_DIR)
    {"validar": validar, "datasets": datasets, "walkforward": walkforward}[modo](log)
    OUTPUT_DIR.mkdir(exist_ok=True)
    (OUTPUT_DIR / f"agua_causal_{modo}.txt").write_text("\n".join(lines), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
