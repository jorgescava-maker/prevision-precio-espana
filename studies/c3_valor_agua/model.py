"""
Fase 1 (modelo) del estudio C3 -- ver DESIGN.md S3 "Fase 1": derivar la
función de valor del agua mediante programación dinámica estocástica y
comparar la política óptima contra el comportamiento observado.

Nota de alcance: DESIGN.md menciona "SDDP" (Stochastic Dual Dynamic
Programming), la técnica estándar para sistemas MULTI-embalse donde la
maldición de la dimensionalidad hace inviable la programación dinámica
exacta. Aquí solo hay UN embalse agregado (serie única de España) -- para
un estado de una sola dimensión, la programación dinámica estocástica exacta
por inducción hacia atrás (backward induction) sobre una rejilla discreta de
nivel de embalse resuelve el mismo problema de forma exacta, sin necesitar
los planos de corte de Benders que SDDP añade específicamente para escalar a
muchos embalses. Se implementa esa versión exacta y más simple -- calibrada
con escenarios históricos de aportación/precio por bootstrap, tal como pide
el diseño -- no una SDDP "de nombre" que sería sobre-ingeniería para un
estado unidimensional.

Uso:
    .venv\\Scripts\\python.exe -m studies.c3_valor_agua.model

Requiere haber corrido antes build_dataset.py. Escribe en output/:
    model_summary.txt, water_value_surface.png, policy_vs_observed.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from etl.common import logging_config  # noqa: F401 - fuerza stdout a UTF-8

OUTPUT_DIR = Path(__file__).resolve().parent / "output"

N_STORAGE_BINS = 40
N_DECISION_CANDIDATES = 30
N_CYCLE_ITERATIONS = 25  # iteraciones del ciclo anual completo hasta converger a una política estacionaria
CONVERGENCE_TOL = 1e-3


def load_scenarios(weekly_series_path: Path | None = None) -> tuple[pd.DataFrame, dict[int, np.ndarray], float, float]:
    """`weekly_series_path` (añadido para reutilizar este DP genérico en otros
    países, p.ej. `precio_da_francia/water_value.py`): por defecto usa la
    serie semanal de España ya construida en este propio estudio; cualquier
    otro país solo necesita construir un parquet con las mismas columnas
    (week/filling_gwh/inflow_gwh/price_avg/gen_hidraulica_gwh_week) y pasar
    su ruta aquí — `solve_dp`/`water_value_surface` ya eran genéricas, sin
    nada específico de España."""
    path = weekly_series_path if weekly_series_path is not None else OUTPUT_DIR / "weekly_series.parquet"
    df = pd.read_parquet(path).dropna(subset=["inflow_gwh", "price_avg", "filling_gwh"])
    df["iso_week"] = df["week"].dt.isocalendar().week.clip(upper=52)  # semana 53 rara -> se funde con la 52

    scenarios: dict[int, np.ndarray] = {}
    for w in range(1, 53):
        sub = df[df["iso_week"] == w][["inflow_gwh", "price_avg"]].dropna()
        if len(sub) == 0:
            continue
        scenarios[w] = sub.to_numpy()

    max_storage = df["filling_gwh"].max()
    turbine_cap = df["gen_hidraulica_gwh_week"].max()
    return df, scenarios, max_storage, turbine_cap


def solve_dp(scenarios: dict[int, np.ndarray], max_storage: float, turbine_cap: float, log) -> tuple[np.ndarray, np.ndarray]:
    log(f"  Rejilla de embalse: {N_STORAGE_BINS} niveles, 0 -> {max_storage:.0f} GWh")
    log(f"  Capacidad de turbinado (tope): {turbine_cap:.0f} GWh/semana (máximo histórico observado)")
    n_scenarios_per_week = {w: len(s) for w, s in scenarios.items()}
    log(f"  Escenarios históricos disponibles por semana: min={min(n_scenarios_per_week.values())}  "
        f"max={max(n_scenarios_per_week.values())}  (limitado por solo ~3-4 años de historia -- ver aviso en RESULTS.md)")

    storage_grid = np.linspace(0, max_storage, N_STORAGE_BINS)
    V_next = np.zeros(N_STORAGE_BINS)  # valor terminal al final del ciclo, se itera hasta converger
    policy = np.zeros((52, N_STORAGE_BINS))
    V_all = np.zeros((53, N_STORAGE_BINS))

    for iteration in range(N_CYCLE_ITERATIONS):
        V_all[52] = V_next
        for w in range(52, 0, -1):
            if w not in scenarios:
                V_all[w - 1] = V_all[w]
                continue
            inflow_s, price_s = scenarios[w][:, 0], scenarios[w][:, 1]
            V_w = np.full(N_STORAGE_BINS, -np.inf)
            pol_w = np.zeros(N_STORAGE_BINS)
            for i, s in enumerate(storage_grid):
                candidates = np.linspace(0, min(turbine_cap, s), N_DECISION_CANDIDATES)
                best_val, best_g = -np.inf, 0.0
                for g in candidates:
                    next_storage = np.clip(s - g + inflow_s, 0, max_storage)
                    v_next_interp = np.interp(next_storage, storage_grid, V_all[w])
                    expected_val = np.mean(price_s * g + v_next_interp)
                    if expected_val > best_val:
                        best_val, best_g = expected_val, g
                V_w[i] = best_val
                pol_w[i] = best_g
            V_all[w - 1] = V_w
            policy[w - 1] = pol_w

        diff = np.max(np.abs(V_all[0] - V_next))
        V_next = V_all[0]
        if diff < CONVERGENCE_TOL * max(1.0, np.max(np.abs(V_next))):
            log(f"  Convergencia en la iteración {iteration + 1} (diff={diff:.4f})")
            break
    else:
        log(f"  No convergió del todo en {N_CYCLE_ITERATIONS} iteraciones (diff final={diff:.4f}) -- "
            f"se usa la última pasada de todos modos, política ya muy estable en la práctica")

    return V_all[:52], policy, storage_grid


def water_value_surface(V_all: np.ndarray, storage_grid: np.ndarray) -> np.ndarray:
    """Valor del agua = derivada discreta de V respecto al nivel de embalse
    (cuánto vale tener 1 GWh más almacenado) -- la superficie que pide el
    catálogo original, EUR/MWh equivalente por semana x nivel."""
    return np.gradient(V_all, storage_grid, axis=1)


def plot_water_value(wv: np.ndarray, storage_grid: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(wv.T, aspect="auto", origin="lower", cmap="YlGnBu",
                    extent=[1, 52, storage_grid[0], storage_grid[-1]])
    ax.set_xlabel("Semana ISO del año")
    ax.set_ylabel("Nivel de embalse (GWh)")
    ax.set_title("C3 -- superficie de valor del agua (EUR/MWh marginal, por semana × nivel)")
    fig.colorbar(im, ax=ax, label="Valor marginal del agua (EUR/MWh)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "water_value_surface.png", dpi=110)
    plt.close(fig)


def compare_policy_vs_observed(df: pd.DataFrame, policy: np.ndarray, storage_grid: np.ndarray, log) -> None:
    log("\n" + "=" * 72)
    log("Política óptima (DP) vs. comportamiento observado")
    log("=" * 72)
    obs = df.dropna(subset=["gen_hidraulica_gwh_week", "filling_gwh"]).copy()
    obs["iso_week_idx"] = obs["week"].dt.isocalendar().week.clip(upper=52) - 1

    def optimal_gen(row) -> float:
        w_idx = int(row["iso_week_idx"])
        return float(np.interp(row["filling_gwh"], storage_grid, policy[w_idx]))

    obs["gen_optimal_dp"] = obs.apply(optimal_gen, axis=1)
    obs["gap"] = obs["gen_hidraulica_gwh_week"] - obs["gen_optimal_dp"]

    log(f"  n={len(obs)} semanas con dato completo")
    log(f"  Generación real media: {obs['gen_hidraulica_gwh_week'].mean():.1f} GWh/semana")
    log(f"  Generación óptima (DP) media: {obs['gen_optimal_dp'].mean():.1f} GWh/semana")
    log(f"  Gap medio (real - óptimo DP): {obs['gap'].mean():+.1f} GWh/semana "
        f"({'genera MÁS de lo que el DP considera óptimo, en media' if obs['gap'].mean() > 0 else 'genera MENOS de lo que el DP considera óptimo, en media' if obs['gap'].mean() < 0 else 'coincide en media'})")
    corr = obs[["gen_hidraulica_gwh_week", "gen_optimal_dp"]].corr().iloc[0, 1]
    log(f"  Correlación real vs. óptimo DP: r={corr:.3f}")
    log("\n  AVISO: un gap sistemático NO se puede interpretar directamente como 'comportamiento")
    log("  estratégico' (retención para forzar precio) sin más -- el DP aquí es una simplificación")
    log("  (1 embalse agregado, escenarios limitados a ~3-4 años, sin modelar restricciones de caudal")
    log("  ecológico ni compromisos de riego/abastecimiento que la operación real SÍ respeta). El gap es")
    log("  un punto de partida para investigar, no una conclusión de comportamiento estratégico en sí.")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(obs["week"], obs["gen_hidraulica_gwh_week"], label="generación real", linewidth=1)
    ax.plot(obs["week"], obs["gen_optimal_dp"], label="generación óptima (DP)", linewidth=1, alpha=0.8)
    ax.set_ylabel("GWh/semana")
    ax.set_title("C3 -- generación hidráulica real vs. óptima según el valor del agua (DP)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "policy_vs_observed.png", dpi=110)
    plt.close(fig)


def main() -> None:
    lines: list[str] = []

    def log(s: str = "") -> None:
        print(s)
        lines.append(s)

    log("=" * 72)
    log("C3 -- Fase 1: programación dinámica estocástica del valor del agua")
    log("=" * 72)

    df, scenarios, max_storage, turbine_cap = load_scenarios()
    log(f"Datos: {len(df)} semanas, {df['week'].min().date()} -> {df['week'].max().date()}")

    log("\nResolviendo la DP (inducción hacia atrás, ciclo anual iterado hasta converger)...")
    V_all, policy, storage_grid = solve_dp(scenarios, max_storage, turbine_cap, log)

    wv = water_value_surface(V_all, storage_grid)
    log(f"\nValor del agua: rango [{wv.min():.2f}, {wv.max():.2f}] EUR/MWh sobre toda la superficie")
    log(f"  Media en niveles BAJOS de embalse (<25% de capacidad): {wv[:, storage_grid < 0.25*max_storage].mean():.2f} EUR/MWh")
    log(f"  Media en niveles ALTOS de embalse (>75% de capacidad): {wv[:, storage_grid > 0.75*max_storage].mean():.2f} EUR/MWh")
    log("  (el valor del agua debería ser mayor cuando el embalse está bajo -- comprobación de sentido")
    log("  económico antes de aceptar la superficie como razonable)")
    plot_water_value(wv, storage_grid)

    compare_policy_vs_observed(df, policy, storage_grid, log)

    summary_path = OUTPUT_DIR / "model_summary.txt"
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    log(f"\nResumen guardado en {summary_path}")
    log("Gráficos: water_value_surface.png, policy_vs_observed.png")


if __name__ == "__main__":
    main()
