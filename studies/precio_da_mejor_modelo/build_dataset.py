"""
"El mejor modelo posible de precio DA" — ver DESIGN.md. Reúne en un solo
dataset D-1-seguro todo lo ya validado en el proyecto (merit_order,
principalmente).

Fase 3 (2026-08-31, pedida por el usuario): el mercado español liquida
realmente en periodos de 15 MINUTOS desde el 2025-10-01 (reforma de
armonización europea del Market Time Unit, ya documentada en
`omie_spot_prices.resolution_minutes`) — antes de esa fecha, en periodos
HORARIOS. El dataset se reconstruye a la resolución NATIVA real de cada
periodo (1 fila = 1 hora hasta 2025-09-30, 1 fila = 1 cuarto de hora desde
2025-10-01) en vez de forzar todo a horario. La mayoría de features
(previsión D-1 de e·sios, disponibilidad térmica, etc.) SIGUEN siendo
horarias incluso tras la reforma — comprobado en vivo, no asumido — así que
se reparten (broadcast) a los 4 cuartos de su hora; solo el precio objetivo y
lo derivado de él (lags, análogos) usa la resolución nativa real.

Uso:
    .venv\\Scripts\\python.exe -m studies.precio_da_mejor_modelo.build_dataset

Produce studies/precio_da_mejor_modelo/output/dataset.parquet.
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from studies.merit_order.build_dataset import DATA_DIR, _build_unit_tiers, _load_generation_units, GAS_HEAT_RATE_RANGE, COAL_HEAT_RATE_RANGE
from studies.merit_order.model import hourly_unavailable_mw

ROOT = Path(__file__).resolve().parents[2]
MERIT_ORDER_OUTPUT = ROOT / "studies" / "merit_order" / "output"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"

# L3 — feature de día análogo: k días históricos (ESTRICTAMENTE anteriores,
# nunca futuros) más parecidos en variables previstas (demanda D-1, tipo de
# día). ANALOG_MIN_HISTORY_DAYS evita buscar análogos con muy poco historial
# detrás (los primeros ~60 días no tienen suficientes candidatos representativos).
ANALOG_K = 5
ANALOG_MIN_HISTORY_DAYS = 60


# ---------------------------------------------------------------------------
# Features horarias (sin cambios de fondo respecto a Fase 0/1/2 — se reparten
# a resolución nativa más abajo, ver _expand_to_native)
# ---------------------------------------------------------------------------

def _load_merit_order_base() -> pd.DataFrame:
    """Columnas D-1-seguras del dataset ya construido y validado en
    merit_order (Fase 1 final) — ver DESIGN.md §3, reutilizadas tal cual en
    vez de reconstruirlas. Grano HORARIO — merit_order no tiene (ni necesita)
    resolución de 15 minutos, el mecanismo de despacho que modela no cambia
    dentro de una misma hora."""
    df = pd.read_parquet(MERIT_ORDER_OUTPUT / "predicciones_tecnologia.parquet")
    keep = [
        "hour_utc",
        "forecast_demanda_mw", "forecast_eolica_mw", "forecast_solar_mw",
        "water_value_eur_mwh", "ttf_eur_mwh", "eua_eur_t", "coal_eur_mwh_th",
        "capacity_year", "cogeneracion_resto_clim", "solar_termica_clim",
        "termica_renovable_clim", "net_import_mw_clim",
        "demand_to_cover_mw", "must_run_final_mw",
        "price_simulado_final", "tecnologia_marginal_final",
    ]
    return df[keep].copy()


def _add_real_availability(df: pd.DataFrame, log) -> pd.DataFrame:
    """Capacidad térmica/nuclear real EXPLÍCITA (L5), a grano horario (los
    eventos de indisponibilidad no tienen resolución más fina que la hora en
    la fuente)."""
    con = duckdb.connect(str(DATA_DIR / "spain.duckdb"), read_only=True)
    try:
        units = _load_generation_units(con)
        gas_tiers_by_year = _build_unit_tiers(units, "B04", GAS_HEAT_RATE_RANGE)
        coal_tiers_by_year = _build_unit_tiers(units, "B05", COAL_HEAT_RATE_RANGE)
        hour_index = pd.DatetimeIndex(df["hour_utc"])
        nuclear_unavail = hourly_unavailable_mw(con, ["B14"], hour_index).to_numpy()
        gas_unavail = hourly_unavailable_mw(con, ["B04"], hour_index).to_numpy()
        coal_unavail = hourly_unavailable_mw(con, ["B02", "B03", "B05", "B06"], hour_index).to_numpy()
        nuclear_capacity = con.execute(
            "SELECT capacity_year, capacity_mw FROM entsoe_installed_capacity WHERE psr_type = 'B14'"
        ).fetchdf().set_index("capacity_year")["capacity_mw"]
    finally:
        con.close()

    df["nuclear_capacity_avail_mw"] = df["capacity_year"].map(nuclear_capacity).to_numpy() - nuclear_unavail
    df["gas_capacity_total_mw"] = df["capacity_year"].map(
        {y: sum(q for _, q in t) for y, t in gas_tiers_by_year.items()}
    )
    df["gas_capacity_avail_mw"] = df["gas_capacity_total_mw"] - gas_unavail
    df["coal_capacity_total_mw"] = df["capacity_year"].map(
        {y: sum(q for _, q in t) for y, t in coal_tiers_by_year.items()}
    )
    df["coal_capacity_avail_mw"] = df["coal_capacity_total_mw"] - coal_unavail
    log(f"  Disponibilidad real: nuclear media={df['nuclear_capacity_avail_mw'].mean():.0f} MW, "
        f"gas media={df['gas_capacity_avail_mw'].mean():.0f} MW, carbón media={df['coal_capacity_avail_mw'].mean():.0f} MW")
    return df


def _build_hourly_features(log) -> pd.DataFrame:
    log("Cargando base D-1-segura de merit_order (grano horario)...")
    df = _load_merit_order_base()
    log("Añadiendo disponibilidad térmica/nuclear real explícita (L5)...")
    df = _add_real_availability(df, log)
    return df


# ---------------------------------------------------------------------------
# Resolución nativa (Fase 3): precio real de OMIE tal cual se liquida —
# horario hasta 2025-09-30, cuartos de hora desde 2025-10-01
# ---------------------------------------------------------------------------

def _load_native_price(con: duckdb.DuckDBPyConnection, log) -> pd.DataFrame:
    df = con.execute(
        "SELECT delivery_start_utc AS period_start_utc, resolution_minutes, "
        "price_eur_mwh_es AS price_real, price_eur_mwh_pt AS price_pt "
        "FROM omie_spot_prices ORDER BY 1"
    ).fetchdf()
    df["period_start_utc"] = pd.to_datetime(df["period_start_utc"])
    # period_in_hour: 0 siempre en la era horaria (resolution_minutes=60);
    # 0/1/2/3 en la era de cuartos (resolution_minutes=15) — el minuto UTC ya
    # refleja correctamente el cuarto local incluso en cambios de hora
    # (el offset de DST europeo es siempre un número entero de horas).
    df["period_in_hour"] = (df["period_start_utc"].dt.minute // 15).astype(int)
    n_hourly = (df["resolution_minutes"] == 60).sum()
    n_quarter = (df["resolution_minutes"] == 15).sum()
    log(f"  Precio nativo cargado: {n_hourly:,} periodos horarios (hasta 2025-09-30) + "
        f"{n_quarter:,} periodos de 15 min (desde 2025-10-01) = {len(df):,} filas")
    return df


def _expand_to_native(hourly_df: pd.DataFrame, native_price: pd.DataFrame) -> pd.DataFrame:
    """Reparte (broadcast) cada feature horaria a los periodos nativos de esa
    hora — comprobado en vivo (2026-08-31) que la previsión D-1 de e·sios
    SIGUE siendo horaria tras la reforma del 2025-10-01, así que dentro de
    una misma hora los 4 cuartos comparten el mismo valor de esas features;
    solo el precio real (y lo derivado de él más abajo) varía de verdad
    cuarto a cuarto."""
    native_price["hour_key"] = native_price["period_start_utc"].dt.floor("h")
    df = native_price.merge(hourly_df, left_on="hour_key", right_on="hour_utc", how="left")
    df = df.drop(columns=["hour_key", "hour_utc"])
    df["hour_of_day"] = df["period_start_utc"].dt.hour
    df["day_of_week"] = df["period_start_utc"].dt.dayofweek
    df["month"] = df["period_start_utc"].dt.month
    df["period_of_day"] = df["hour_of_day"] * 4 + df["period_in_hour"]  # 0-95, consistente en ambas eras
    return df


def _add_holidays(df: pd.DataFrame) -> pd.DataFrame:
    con = duckdb.connect(str(DATA_DIR / "spain.duckdb"), read_only=True)
    try:
        holidays = con.execute(
            "SELECT holiday_date FROM holiday_calendar WHERE location_key = 'spain'"
        ).fetchdf()
    finally:
        con.close()
    holiday_dates = set(pd.to_datetime(holidays["holiday_date"]).dt.date)
    df["date"] = df["period_start_utc"].dt.date
    df["is_holiday"] = df["date"].isin(holiday_dates)
    df["is_weekend"] = df["day_of_week"] >= 5
    df["is_dia_no_laborable"] = df["is_holiday"] | df["is_weekend"]
    return df


def _build_analog_feature_v2(df: pd.DataFrame, log) -> pd.DataFrame:
    """L3 v2 (demanda + ratio renovable), adaptada a resolución nativa: el
    emparejamiento de días sigue siendo diario (sin cambios), pero la
    consulta de precio de los días análogos ahora se hace por
    `period_of_day` (0-95) en vez de `hour_of_day` (0-23) — para que un
    cuarto de las 14:15 se compare con el mismo cuarto de las 14:15 de los
    días análogos, no con toda la hora 14 mezclada."""
    daily = df.groupby("date").agg(
        forecast_demanda_dia_mw=("forecast_demanda_mw", "mean"),
        forecast_eolica_dia_mw=("forecast_eolica_mw", "mean"),
        forecast_solar_dia_mw=("forecast_solar_mw", "mean"),
        is_dia_no_laborable=("is_dia_no_laborable", "first"),
    ).reset_index().sort_values("date").reset_index(drop=True)
    daily["ratio_renovable_dia"] = (
        (daily["forecast_eolica_dia_mw"] + daily["forecast_solar_dia_mw"]) / daily["forecast_demanda_dia_mw"]
    )

    n = len(daily)
    demanda = daily["forecast_demanda_dia_mw"].to_numpy()
    ratio_ren = daily["ratio_renovable_dia"].to_numpy()
    tipo_dia = daily["is_dia_no_laborable"].to_numpy()

    demanda_series = pd.Series(demanda)
    ratio_series = pd.Series(ratio_ren)
    demanda_mean = demanda_series.expanding().mean().shift(1).to_numpy()
    demanda_std = demanda_series.expanding().std().shift(1).to_numpy()
    ratio_mean = ratio_series.expanding().mean().shift(1).to_numpy()
    ratio_std = ratio_series.expanding().std().shift(1).to_numpy()

    analog_dates_per_day: dict = {}
    for i in range(n):
        if i < ANALOG_MIN_HISTORY_DAYS:
            analog_dates_per_day[daily["date"].iloc[i]] = []
            continue
        candidates = np.arange(0, i)
        same_type = candidates[tipo_dia[candidates] == tipo_dia[i]]
        pool = same_type if len(same_type) >= ANALOG_K else candidates
        z_demanda = (demanda[pool] - demanda_mean[i]) / (demanda_std[i] or 1.0)
        z_ratio = (ratio_ren[pool] - ratio_mean[i]) / (ratio_std[i] or 1.0)
        target_z_demanda = (demanda[i] - demanda_mean[i]) / (demanda_std[i] or 1.0)
        target_z_ratio = (ratio_ren[i] - ratio_mean[i]) / (ratio_std[i] or 1.0)
        dist = np.sqrt((z_demanda - target_z_demanda) ** 2 + (z_ratio - target_z_ratio) ** 2)
        nearest_idx = pool[np.argsort(dist)[:ANALOG_K]]
        analog_dates_per_day[daily["date"].iloc[i]] = daily["date"].iloc[nearest_idx].tolist()

    log(f"  Análogos v2 (demanda + ratio renovable, por period_of_day) calculados para "
        f"{n - ANALOG_MIN_HISTORY_DAYS:,}/{n:,} días")

    price_by_date_period = df.set_index(["date", "period_of_day"])["price_real"].to_dict()
    analog_price_v2 = np.full(len(df), np.nan)
    for idx, (d, p) in enumerate(zip(df["date"], df["period_of_day"])):
        analog_dates = analog_dates_per_day.get(d, [])
        if not analog_dates:
            continue
        vals = []
        for ad in analog_dates:
            v = price_by_date_period.get((ad, p))
            if v is None:
                # El día análogo es de la era horaria (sin cuartos) y p no es
                # el cuarto :00 — se cae al precio de esa hora completa como
                # mejor aproximación disponible, en vez de perder el dato.
                v = price_by_date_period.get((ad, (p // 4) * 4))
            if v is not None and not pd.isna(v):
                vals.append(v)
        if vals:
            analog_price_v2[idx] = float(np.mean(vals))

    df["analog_price_mean_v2"] = analog_price_v2
    df["analog_price_dev"] = df["forecast_demanda_mw"] - df.groupby("date")["forecast_demanda_mw"].transform("mean")
    return df


ANALOG_MIN_HISTORY_PERIODS = 60  # ~60 periodos del mismo period_of_day (≈60 días)


def _build_analog_generic(df: pd.DataFrame, dims: list[str], out_col: str, log, label: str,
                           borrow_parent_hour_history: bool = False,
                           type_col: str = "is_dia_no_laborable") -> pd.DataFrame:
    """Núcleo compartido de los análogos "por periodo" (v3, v4, ...): busca
    los ANALOG_K periodos HISTÓRICOS (estrictamente anteriores, mismo
    period_of_day, mismo tipo de día) más parecidos en `dims` (z-score
    expanding causal, solo con datos hasta el periodo anterior), y devuelve
    el precio real medio de esos análogos. Generalización de
    `_build_analog_feature_v3` para poder añadir más dimensiones de
    emparejamiento (v4) sin duplicar el bucle.

    `borrow_parent_hour_history` (v5, hallazgo 2026-08-31): los cuartos
    :15/:30/:45 solo existen desde el 2025-10-01 (335 periodos históricos
    propios) frente a ~1.339 de los cuartos :00 (heredan toda la era horaria
    desde 2023) — verificado que por eso su correlación era notablemente
    peor (0,843 vs 0,916). Cuando está activo, el fondo de candidatos de un
    cuarto no-:00 se AMPLÍA con el histórico horario (pre-reforma) de su
    HORA MADRE (period_of_day redondeado a :00) — no se inventa un precio de
    cuarto que no existió, se usa esa historia solo para tener más contexto
    económico (demanda/ratio/ttf/valor del agua) al buscar analogías; la
    predicción sigue siendo únicamente para las filas que son de verdad ese
    cuarto."""
    df = df.sort_values("period_start_utc").reset_index(drop=True)
    analog_price = np.full(len(df), np.nan)
    n_groups_done = 0
    for pod, g_own in df.groupby("period_of_day"):
        if borrow_parent_hour_history and pod % 4 != 0:
            parent_pod = (pod // 4) * 4
            parent_hist = df[(df["period_of_day"] == parent_pod) & (df["resolution_minutes"] == 60)]
            g = pd.concat([parent_hist, g_own]).sort_values("period_start_utc")
        else:
            g = g_own.sort_values("period_start_utc")
        g = g.reset_index()  # 'index' = posición original en df (para escribir analog_price)
        is_target = (g["period_of_day"] == pod).to_numpy()
        orig_idx = g["index"].to_numpy()
        n = len(g)
        tipo_dia = g[type_col].to_numpy()
        price = g["price_real"].to_numpy()

        dim_arrays = {}
        for dim in dims:
            v = g[dim].to_numpy()
            s = pd.Series(v)
            m = s.expanding().mean().shift(1).to_numpy()
            sd = s.expanding().std().shift(1).to_numpy()
            dim_arrays[dim] = (v, m, sd)

        for i in range(n):
            if not is_target[i] or i < ANALOG_MIN_HISTORY_PERIODS:
                continue
            candidates = np.arange(0, i)  # estrictamente anteriores (mismo period_of_day, + historia de la hora madre si aplica)
            same_type = candidates[tipo_dia[candidates] == tipo_dia[i]]
            pool = same_type if len(same_type) >= ANALOG_K else candidates
            if len(pool) == 0:
                continue
            dist2 = np.zeros(len(pool))
            skip = False
            for dim in dims:
                v, m, sd = dim_arrays[dim]
                if pd.isna(v[i]) or pd.isna(m[i]):
                    skip = True
                    break
                std_i = sd[i] or 1.0
                z_pool = np.nan_to_num((v[pool] - m[i]) / std_i, nan=0.0)
                z_i = (v[i] - m[i]) / std_i
                dist2 += (z_pool - z_i) ** 2
            if skip:
                continue
            dist = np.sqrt(dist2)
            nearest = pool[np.argsort(dist)[:ANALOG_K]]
            vals = price[nearest]
            vals = vals[~np.isnan(vals)]
            if len(vals):
                analog_price[orig_idx[i]] = float(vals.mean())
        n_groups_done += 1

    df[out_col] = analog_price
    log(f"  Análogos {label} calculados sobre {n_groups_done} period_of_day distintos, "
        f"{df[out_col].notna().sum():,}/{len(df):,} periodos con dato (dims={dims})")
    return df


def _build_analog_feature_v3(df: pd.DataFrame, log) -> pd.DataFrame:
    """Fase 4 — mejora de v2 pedida por el usuario: en vez de emparejar por
    la MEDIA DEL DÍA (v2) y luego consultar el precio de ese cuarto en los
    días análogos, se empareja directamente por las condiciones previstas de
    ESE MISMO period_of_day (su propia demanda/ratio renovable, no la media
    del día) — un día con una hora puntual rara pero promedio diario normal
    se le escapaba a v2, aquí no. Probado en vivo antes de adoptar: sube la
    correlación de +0,719 (v2) a +0,804."""
    df["ratio_renovable_periodo"] = (df["forecast_eolica_mw"] + df["forecast_solar_mw"]) / df["forecast_demanda_mw"]
    return _build_analog_generic(
        df, ["forecast_demanda_mw", "ratio_renovable_periodo"], "analog_price_mean_v3", log, "v3 (demanda+ratio)"
    )


def _build_analog_feature_v4(df: pd.DataFrame, log) -> pd.DataFrame:
    """Fase 5 — más dimensiones de emparejamiento sobre v3: añade el precio
    del gas (`ttf_eur_mwh`, D-2-seguro) y el valor del agua
    (`water_value_eur_mwh`) — el mismo nivel de demanda/renovable puede dar
    precios muy distintos según el régimen de coste de combustible o cuánto
    "vale" retener agua esa semana. Probado en vivo antes de adoptar: sube la
    correlación de +0,804 (v3) a +0,901."""
    return _build_analog_generic(
        df, ["forecast_demanda_mw", "ratio_renovable_periodo", "ttf_eur_mwh", "water_value_eur_mwh"],
        "analog_price_mean_v4", log, "v4 (demanda+ratio+ttf+water_value)",
    )


def _build_analog_feature_v5(df: pd.DataFrame, log) -> pd.DataFrame:
    """Fase 6 — arregla la debilidad real de v4 en los cuartos :15/:30/:45
    (correlación 0,843 frente a 0,916 de los cuartos :00, verificado en vivo:
    solo 335 periodos históricos propios desde el 2025-10-01, frente a 1.339
    de los cuartos :00 que heredan toda la era horaria) — mismas 4
    dimensiones que v4, pero con `borrow_parent_hour_history=True`."""
    return _build_analog_generic(
        df, ["forecast_demanda_mw", "ratio_renovable_periodo", "ttf_eur_mwh", "water_value_eur_mwh"],
        "analog_price_mean_v5", log, "v5 (=v4 + historia de la hora madre para cuartos no-:00)",
        borrow_parent_hour_history=True,
    )


def _add_price_lags(df: pd.DataFrame, log) -> pd.DataFrame:
    """Features autorregresivas — reescritas para resolución nativa (pedido
    explícito del usuario, 2026-08-31): el lag debe ser del MISMO PERIODO del
    día anterior, no de "24 filas antes" — un desplazamiento por número de
    filas se rompe justo en el cambio de resolución (2025-10-01: pasa de 24
    filas/día a 96) y dentro de toda la era de 15 minutos (24 filas serían
    solo 6 horas, no 24). Se usa un lookup por TIMESTAMP EXACTO en su lugar,
    correcto en las dos eras y en la propia transición entre ellas."""
    df = df.sort_values("period_start_utc").reset_index(drop=True)
    price_by_time = df.set_index("period_start_utc")["price_real"]

    # Fase 4 — probado en vivo (comparación de correlación simple) antes de
    # comprometerse: 48h llena el hueco entre 24h y 168h (corr +0,759, entre
    # el +0,855 de 24h y el +0,714 de 168h); la EMA bate a la media móvil
    # simple en todos los horizontes probados (7/14/28 días) — se adopta EMA
    # en vez de rolling simple.
    for hours in (24, 48, 168):
        lag_time = df["period_start_utc"] - pd.Timedelta(hours=hours)
        df[f"price_lag_{hours}h"] = lag_time.map(price_by_time)

    daily_price = df.groupby("date")["price_real"].mean().sort_index()
    ema7 = daily_price.ewm(span=7).mean().shift(1)
    ema28 = daily_price.ewm(span=28).mean().shift(1)
    df["price_ema_7d"] = df["date"].map(ema7)
    df["price_ema_28d"] = df["date"].map(ema28)

    # Fase 10 — planteado por el usuario (2026-08-31): un lag corto (<24h)
    # NO tiene sentido en general, porque todos los periodos de un día D se
    # deciden a la vez en una única subasta — no hay "3 horas antes, del
    # mismo día" ya conocido para predecir "ahora". Pero SÍ tiene sentido en
    # la FRONTERA entre días: el último periodo de D-1 (23:45, o 23:00 en la
    # era horaria) se fijó en la subasta de D-2, así que está totalmente
    # conocido mucho antes de que cierre la subasta de D — no hay fuga. Y por
    # continuidad física/económica alrededor de medianoche cabe esperar que
    # se parezca a los primeros periodos de D. Se construye como una única
    # feature por día D (constante para todos sus periodos), no como un lag
    # genérico de "1 periodo antes" — eso sí sería un lag intra-día inválido.
    last_period_prev_day = df.groupby("date")["price_real"].last()
    last_period_prev_day.index = pd.to_datetime(last_period_prev_day.index) + pd.Timedelta(days=1)
    df["price_boundary_prev_day"] = pd.to_datetime(df["date"]).map(last_period_prev_day)

    log(f"  Lags de precio (por timestamp exacto): nulos lag_24h={df['price_lag_24h'].isna().sum()}, "
        f"lag_48h={df['price_lag_48h'].isna().sum()}, lag_168h={df['price_lag_168h'].isna().sum()}, "
        f"ema_7d={df['price_ema_7d'].isna().sum()}, ema_28d={df['price_ema_28d'].isna().sum()}, "
        f"boundary_prev_day={df['price_boundary_prev_day'].isna().sum()}")
    return df


def _france_nuclear_avail_hourly(log) -> pd.DataFrame:
    """Disponibilidad nuclear francesa real (capacidad − indisponibilidad),
    mismo criterio D-1-seguro que la nuclear española de merit_order (máximo
    por central entre eventos que solapan, no suma directa — mismo hallazgo
    de duplicados por revisión-como-mRID-nuevo, ver findings.md #89).
    `nuclear_outage_events` de `france.duckdb` no tiene columna `psr_type`
    (toda la tabla es nuclear) — se reimplementa el barrido en vez de forzar
    la reutilización de `hourly_unavailable_mw` (pensada para
    `generation_outage_events` de España, con `psr_type`)."""
    con = duckdb.connect(str(ROOT / "data" / "france.duckdb"), read_only=True)
    try:
        events = con.execute(
            "SELECT unit_resource_id, event_start_utc, event_end_utc, unavailable_mw "
            "FROM nuclear_outage_events WHERE unavailable_mw IS NOT NULL"
        ).fetchdf()
        capacity = con.execute(
            "SELECT capacity_year, capacity_mw FROM entsoe_installed_capacity WHERE psr_type = 'B14'"
        ).fetchdf().set_index("capacity_year")["capacity_mw"]
    finally:
        con.close()

    start = pd.Timestamp("2023-01-01")
    end = pd.Timestamp("2026-09-01")
    grid = pd.date_range(start, end, freq="h")
    n = len(grid)
    total_unavail = np.zeros(n)
    for _, g in events.groupby("unit_resource_id"):
        arr = np.zeros(n)
        for _, e in g.iterrows():
            s = max(pd.Timestamp(e["event_start_utc"]).floor("h"), grid[0])
            t = min(pd.Timestamp(e["event_end_utc"]).floor("h"), grid[-1])
            if s >= t:
                continue
            i0 = int((s - grid[0]) / pd.Timedelta(hours=1))
            i1 = int((t - grid[0]) / pd.Timedelta(hours=1))
            arr[i0:i1] = np.maximum(arr[i0:i1], e["unavailable_mw"])
        total_unavail += arr

    out = pd.DataFrame({"hour_utc": grid})
    out["capacity_year"] = out["hour_utc"].dt.year
    out["nuclear_capacity_fr_avail_mw"] = out["capacity_year"].map(capacity).to_numpy() - total_unavail
    log(f"  Disponibilidad nuclear FR: media={out['nuclear_capacity_fr_avail_mw'].mean():.0f} MW "
        f"({len(events)} eventos)")
    return out[["hour_utc", "nuclear_capacity_fr_avail_mw"]]


def _add_interconnection_features(df: pd.DataFrame, log) -> pd.DataFrame:
    """L11/hallazgo del usuario (2026-08-31): "no sé si estamos teniendo en
    cuenta interconexiones con Francia y Portugal" — no se tenían más allá de
    `net_import_mw_clim` (climatología). Todo lo de abajo ya estaba cargado
    en el proyecto (france.duckdb, interconnections.duckdb,
    omie_spot_prices.price_eur_mwh_pt) — no es una fuente nueva, es usar lo
    que ya había.

    OJO con la circularidad: el precio francés/portugués del MISMO día se fija
    en la MISMA subasta acoplada (EUPHEMIA) que el español — no está
    disponible antes, usarlo tal cual sería tan circular como usar el propio
    price_real de hoy. Por eso aquí solo se usan: (a) precios franceses/
    portugueses REZAGADOS (mismo criterio que los lags propios — ya
    publicados y públicos mucho antes del cierre de la subasta de hoy), (b)
    NTC de "mañana" (ya publicada en el momento de la carga diaria, ver
    docstring de `etl/sources/entsoe_ntc.py`), y (c) disponibilidad nuclear
    francesa real (D-1-conocida, mismo criterio que la española)."""
    con = duckdb.connect(str(ROOT / "data" / "france.duckdb"), read_only=True)
    try:
        fr_price = con.execute(
            "SELECT interval_start_utc AS t, price_eur_mwh AS price_fr FROM entsoe_day_ahead_prices"
        ).fetchdf()
    finally:
        con.close()
    con = duckdb.connect(str(ROOT / "data" / "interconnections.duckdb"), read_only=True)
    try:
        # Dos filas por hora (una por sentido) — se usa Francia->España (capacidad de
        # IMPORTAR a España), la dirección más directamente relevante para el precio.
        ntc = con.execute(
            "SELECT interval_start_utc AS t, ntc_mw FROM ntc_estimated "
            "WHERE border_key = 'ES-FR' AND from_area = 'france' AND to_area = 'spain'"
        ).fetchdf()
    finally:
        con.close()
    fr_price["t"] = pd.to_datetime(fr_price["t"])
    ntc["t"] = pd.to_datetime(ntc["t"])
    fr_avail = _france_nuclear_avail_hourly(log)

    # Lags de precio FR (mismo criterio de timestamp exacto que los propios)
    price_fr_by_time = fr_price.set_index("t")["price_fr"]
    df["price_lag_24h_fr"] = (df["period_start_utc"] - pd.Timedelta(hours=24)).map(price_fr_by_time)
    df["price_lag_168h_fr"] = (df["period_start_utc"] - pd.Timedelta(hours=168)).map(price_fr_by_time)

    # Lags de precio PT (ya viene en omie_spot_prices, mismo periodo nativo)
    price_pt_by_time = df.set_index("period_start_utc")["price_pt"]
    df["price_lag_24h_pt"] = (df["period_start_utc"] - pd.Timedelta(hours=24)).map(price_pt_by_time)
    df["price_lag_168h_pt"] = (df["period_start_utc"] - pd.Timedelta(hours=168)).map(price_pt_by_time)

    # NTC de mañana y disponibilidad nuclear FR: horarias, se reparten al cuarto (broadcast)
    hour_key = df["period_start_utc"].dt.floor("h")
    df["ntc_es_fr_mw"] = hour_key.map(ntc.set_index("t")["ntc_mw"])
    df["nuclear_capacity_fr_avail_mw"] = hour_key.map(fr_avail.set_index("hour_utc")["nuclear_capacity_fr_avail_mw"])

    log(f"  Interconexiones ES-FR/ES-PT añadidas: nulos lag24_fr={df['price_lag_24h_fr'].isna().sum()}, "
        f"lag24_pt={df['price_lag_24h_pt'].isna().sum()}, ntc={df['ntc_es_fr_mw'].isna().sum()}, "
        f"nuclear_fr={df['nuclear_capacity_fr_avail_mw'].isna().sum()}")
    return df


# Fase 14 — margen de reserva del sistema: capacidad ya de turbinado hidráulico
# máximo (misma constante que C3/merit_order usan como tope de la DP de
# valor del agua) — fija, no depende de la hora.
HYDRO_TURBINE_CAP_MW = 7519.6


def _add_reserve_margin(df: pd.DataFrame, log) -> pd.DataFrame:
    """Fase 14 — hipótesis propuesta por el usuario (2026-08-31), razonada
    antes de construir nada: la imagen especular de A2 (que ya estableció que
    el precio CERO es casi determinista vía un umbral simple renovable/
    demanda). Aquí el umbral equivalente para el TRAMO ALTO sería la
    ESCASEZ — cuánto margen le queda al sistema por encima de la demanda
    prevista, sumando TODA la capacidad D-1-conocida (nuclear/gas/carbón
    reales, turbinado hidráulico máximo, previsión eólica/solar, y el resto
    de climatologías ya usadas en merit_order) menos la demanda prevista.
    Cuando ese margen es estrecho, el sistema entra en la parte convexa de la
    curva de oferta (un generador puede pujar por encima de coste marginal
    sabiendo que es imprescindible) — el motor de merit_order nunca modela
    esto (DESIGN.md ya lo señala: "sin oferta estratégica"), así que es
    endeble ver si dárselo al modelo ML como feature explícita ayuda donde el
    motor mecanicista no puede por diseño.

    Comprobado en vivo antes de integrar (no en este docstring, ver
    RESULTS.md Fase 14): el margen correlaciona -0,72 con el precio real
    (validación de sentido) y, más importante, el SESGO del modelo actual
    (Fase 10) es monótono por decil de margen — subestima cuando el margen
    es estrecho (-3,37 en el decil más ajustado), sobreestima cuando es
    holgado (+3,78 en el más ajustado) — justo la dirección que la
    intuición operativa del usuario predecía, y una variable que el árbol no
    tiene ya lista de fábrica (es una suma de 9 columnas, cada una ya en el
    dataset por separado, pero no como combinación única)."""
    df["reserve_margin_mw"] = (
        df["nuclear_capacity_avail_mw"] + df["gas_capacity_avail_mw"] + df["coal_capacity_avail_mw"]
        + HYDRO_TURBINE_CAP_MW
        + df["forecast_eolica_mw"] + df["forecast_solar_mw"]
        + df["cogeneracion_resto_clim"] + df["termica_renovable_clim"] + df["solar_termica_clim"]
        - df["forecast_demanda_mw"]
    )
    log(f"  Margen de reserva: media={df['reserve_margin_mw'].mean():.0f} MW, "
        f"rango=[{df['reserve_margin_mw'].min():.0f}, {df['reserve_margin_mw'].max():.0f}] MW, "
        f"correlación con precio real={df['reserve_margin_mw'].corr(df['price_real']):.3f}")
    return df


def _add_supply_curve_climatology(df: pd.DataFrame, log) -> pd.DataFrame:
    """Fase 15 — pedido explícito del usuario (2026-09-01): "hay que
    abordarlo desde el modelo de ofertas de España... cómo operan los
    distintos participantes". Ver findings.md #107: con el valor SIN LAG
    (ex-post) de la curva real de OMIE (`omie_supply_curve_summary`,
    etl/sources/omie_supply_curve.py), la pendiente de la curva en el margen
    (`slope_next500_eur_per_mw`) y la holgura de oferta vista desde la puja
    (`unused_sell_headroom_mw`) explican una parte real del sesgo en el
    régimen de escasez extrema que ni la mezcla tecnológica ni el margen
    físico (`reserve_margin_mw`) capturaban — pero esa serie es EX-POST
    (el fichero de un día de entrega no se conoce hasta después de que
    cierre SU PROPIA subasta), así que no es usable en producción tal cual.

    D-1-seguridad, verificado en vivo (2026-09-01): el campo "Fecha Emisión"
    del fichero de OMIE es INCONSISTENTE — en fechas recientes a veces
    muestra una marca posterior a la propia fecha de entrega (probable
    artefacto de cómo el portal sirve/regenera peticiones de fechas
    recientes), pero en fechas ya asentadas (semanas atrás) es siempre
    D-1 ~13:30-13:50, justo después del cierre de SU PROPIA subasta — igual
    que el precio oficial. Se asume ese patrón estable como el real: la
    curva de un día de entrega X se conoce el propio día X-1 poco después
    de mediodía, nunca antes. Por tanto, la curva del día D-1 (conocida el
    propio D-1 después de mediodía) SÍ está disponible con margen antes de
    que cierre la subasta del día D (mediodía D-1 también) — un lag de 1
    día natural es seguro. Se usa la MEDIA de los últimos 3 días naturales
    (mismo cuarto de hora) en vez de un solo lag=1 día: comprobado en vivo,
    la correlación del valor sin lag con la media de proxies se aplana a
    partir de 3 días (0,177 con lag=1 día solo vs. 0,231 con media de 3 días
    para la pendiente; ~0,80 en ambos casos para la holgura) — no hay
    ganancia real en ventanas más largas.

    IMPORTANTE: esta es la primera versión D-1-segura, sin validar todavía
    con el protocolo walk-forward estándar del proyecto (12 meses de
    calentamiento) — solo hay ~11 meses de histórico de curva. Ver
    test_supply_curve_feature.py para la prueba preliminar."""
    df = df.sort_values("period_start_utc").reset_index(drop=True)

    con = duckdb.connect(str(DATA_DIR / "spain.duckdb"), read_only=True)
    try:
        curve = con.execute(
            "SELECT period_start_utc, slope_next500_eur_per_mw, unused_sell_headroom_mw FROM omie_supply_curve_summary"
        ).fetchdf()
    finally:
        con.close()

    for col in ["slope_next500_eur_per_mw", "unused_sell_headroom_mw"]:
        series_by_time = curve.set_index("period_start_utc")[col]
        lags = []
        for days in (1, 2, 3):
            lag_time = df["period_start_utc"] - pd.Timedelta(days=days)
            lags.append(lag_time.map(series_by_time))
        df[f"{col}_clim"] = pd.concat(lags, axis=1).mean(axis=1)

    log(f"  Curva de oferta (D-1-segura, media de 3 días naturales, mismo cuarto de hora): "
        f"nulos slope_clim={df['slope_next500_eur_per_mw_clim'].isna().sum()}, "
        f"nulos headroom_clim={df['unused_sell_headroom_mw_clim'].isna().sum()}")
    return df


def build() -> pd.DataFrame:
    lines: list[str] = []

    def log(s: str = "") -> None:
        print(s)
        lines.append(s)

    hourly_df = _build_hourly_features(log)

    con = duckdb.connect(str(DATA_DIR / "spain.duckdb"), read_only=True)
    try:
        native_price = _load_native_price(con, log)
    finally:
        con.close()

    log("Repartiendo features horarias a la resolución nativa de cada periodo...")
    df = _expand_to_native(hourly_df, native_price)

    log("Añadiendo festivos/fin de semana...")
    df = _add_holidays(df)

    log("Construyendo el análogo v2 (demanda + ratio renovable, L3) en resolución nativa...")
    df = _build_analog_feature_v2(df, log)

    log("Construyendo el análogo v3 (emparejado por periodo, no por día — idea del usuario)...")
    df = _build_analog_feature_v3(df, log)

    log("Construyendo el análogo v4 (+ttf +valor del agua)...")
    df = _build_analog_feature_v4(df, log)

    log("Construyendo el análogo v5 (=v4 + historia de la hora madre para cuartos no-:00)...")
    df = _build_analog_feature_v5(df, log)

    log("Añadiendo lags de precio por timestamp exacto (mismo periodo, no misma fila)...")
    df = _add_price_lags(df, log)

    log("Añadiendo interconexiones ES-FR/ES-PT (hallazgo del usuario, Fase 9)...")
    df = _add_interconnection_features(df, log)

    log("Añadiendo margen de reserva del sistema (hipótesis del usuario, Fase 14)...")
    df = _add_reserve_margin(df, log)

    log("Añadiendo curva real de oferta de OMIE, D-1-segura (Fase 15, findings.md #107)...")
    df = _add_supply_curve_climatology(df, log)

    df = df.sort_values("period_start_utc").reset_index(drop=True)

    # EL DÍA DE MERCADO ES LOCAL, NO UTC. Corrige una fuga que afectaba al
    # 93,4% de las filas (`price_boundary_prev_day` entregaba un precio del
    # propio día de mercado) y además pone el calendario en hora local. En el
    # modelo privado equivalente esto vale +0,684 EUR/MWh: las cifras SIN esta
    # corrección son optimistas. Ver el docstring de `dia_mercado`.
    log("Pasando frontera, EMAs y calendario al DÍA DE MERCADO local (dia_mercado)...")
    from studies.precio_da_mejor_modelo.dia_mercado import aplicar as _dia_mercado
    df = _dia_mercado(df, log)

    # Las diez de la revisión de 2026-09-13. Ninguna añade un dato nuevo: son
    # combinaciones y transformaciones de series que ya están en el dataset
    # (ver el docstring de `features_v3`). Va al final a propósito, porque el
    # componente de largo plazo necesita la serie de precio ya montada.
    log("Añadiendo LTSC, variación de previsiones y tensión francesa (features_v3)...")
    from studies.precio_da_mejor_modelo.features_v3 import anadir as _anadir_v3
    df, _ = _anadir_v3(df, log)

    summary_path = OUTPUT_DIR / "build_summary.txt"
    OUTPUT_DIR.mkdir(exist_ok=True)
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    return df


def main() -> None:
    df = build()
    OUTPUT_DIR.mkdir(exist_ok=True)
    out_path = OUTPUT_DIR / "dataset.parquet"
    df.to_parquet(out_path, index=False)
    print(f"\nDataset construido: {len(df):,} periodos, {df['period_start_utc'].min()} -> {df['period_start_utc'].max()}")
    print(f"  De ellos, resolución 15 min: {(df['resolution_minutes']==15).sum():,}  "
          f"horaria: {(df['resolution_minutes']==60).sum():,}")
    print(f"Guardado en {out_path}")
    print(f"\nColumnas: {df.columns.tolist()}")
    print("\nNulos por columna:")
    print(df.isna().sum())


if __name__ == "__main__":
    main()
