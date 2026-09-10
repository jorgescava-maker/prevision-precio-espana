"""
Evalúa las previsiones diarias publicadas en `previsiones/diarias/` contra el
precio real de OMIE, en cuanto OMIE lo publica (hacia las 13:00 de la víspera
del día de entrega).

Escribe:
  resultados/evaluacion_diaria.csv   una fila por día evaluado
  resultados/README.md               resumen (lo enlaza el README principal)

Como referencia de contexto se incluye la previsión ingenua habitual en la
literatura: el precio del mismo periodo del día anterior. No es un listón que
haya que superar, solo ayuda a leer si un día fue difícil.

Uso:
    python -m scripts.evaluar
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from etl.common import logging_config  # noqa: F401 - fuerza stdout a UTF-8

ROOT =Path(__file__).resolve().parents[1]
PREVISIONES = ROOT / "previsiones" / "diarias"
RESULTADOS = ROOT / "resultados"
DB = ROOT / "data" / "spain.duckdb"


def _precio_real() -> pd.DataFrame:
    con = duckdb.connect(str(DB), read_only=True)
    try:
        return con.execute(
            "SELECT delivery_date, delivery_start_utc AS period_start_utc, price_eur_mwh_es AS precio_real "
            "FROM omie_spot_prices WHERE delivery_date >= '2026-01-01' ORDER BY 2"
        ).fetchdf()
    finally:
        con.close()


def _fmt(x: float) -> str:
    return f"{x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def main() -> int:
    ficheros = sorted(PREVISIONES.glob("*.csv"))
    if not ficheros:
        print("No hay previsiones publicadas todavía.")
        return 0
    prev = pd.concat([pd.read_csv(f, parse_dates=["inicio_utc"]).assign(fichero=f.name) for f in ficheros])
    real = _precio_real()

    # Ingenua: el mismo periodo local del día anterior (se empareja por
    # posición dentro del día, así los cambios de hora no desalinean nada).
    real["pos"] = real.groupby("delivery_date").cumcount()
    ayer = real[["delivery_date", "pos", "precio_real"]].copy()
    ayer["delivery_date"] = ayer["delivery_date"] + pd.Timedelta(days=1)
    real = real.merge(ayer.rename(columns={"precio_real": "ingenua"}), on=["delivery_date", "pos"], how="left")

    d = prev.merge(real, left_on="inicio_utc", right_on="period_start_utc", how="inner")
    if d.empty:
        print("Ninguna previsión tiene todavía precio real publicado.")
        return 0

    filas = []
    for dia, g in d.groupby("dia_entrega"):
        filas.append({
            "dia_entrega": dia,
            "periodos": len(g),
            "mae": (g["precio_real"] - g["precio_previsto"]).abs().mean(),
            "mae_ingenua": (g["precio_real"] - g["ingenua"]).abs().mean(),
            "media_real": g["precio_real"].mean(),
            "media_prevista": g["precio_previsto"].mean(),
            "emitida_utc": g["emitida_utc"].iloc[0],
        })
    ev = pd.DataFrame(filas).sort_values("dia_entrega")
    RESULTADOS.mkdir(exist_ok=True)
    ev.to_csv(RESULTADOS / "evaluacion_diaria.csv", index=False, float_format="%.3f")

    # MAE agregado por PERIODO (no media de medias diarias), igual que el backtest.
    err = (d["precio_real"] - d["precio_previsto"]).abs()
    err_ing = (d["precio_real"] - d["ingenua"]).abs()
    ultimos = ev.tail(30)
    lineas = [
        "# Resultados de la previsión publicada",
        "",
        "Se actualiza sola cada día. Cada previsión se publica antes del cierre de la",
        "subasta (12:00, hora peninsular) y se evalúa cuando OMIE publica el precio real.",
        "",
        f"- Días evaluados: **{len(ev)}** ({ev['dia_entrega'].min()} → {ev['dia_entrega'].max()})",
        f"- MAE por periodo, todo el histórico publicado: **{_fmt(err.mean())} EUR/MWh**"
        f" (ingenua del día anterior: {_fmt(err_ing.mean())})",
        f"- MAE de la media diaria: **{_fmt((ev['media_real'] - ev['media_prevista']).abs().mean())} EUR/MWh**",
        "",
        "## Últimos 30 días",
        "",
        "| Día | Periodos | MAE | Ingenua | Media real | Media prevista |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, r in ultimos.iloc[::-1].iterrows():
        lineas.append(f"| {r['dia_entrega']} | {r['periodos']} | {_fmt(r['mae'])} | {_fmt(r['mae_ingenua'])} | "
                      f"{_fmt(r['media_real'])} | {_fmt(r['media_prevista'])} |")
    lineas += ["", "Detalle completo: [`evaluacion_diaria.csv`](evaluacion_diaria.csv)."]
    (RESULTADOS / "README.md").write_text("\n".join(lineas) + "\n", encoding="utf-8")
    print(f"{len(ev)} días evaluados · MAE {err.mean():.2f} · ingenua {err_ing.mean():.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
