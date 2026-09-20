# Resultados de la previsión publicada

Se actualiza sola cada día. Cada previsión se publica antes del cierre de la
subasta (12:00, hora peninsular) y se evalúa cuando OMIE publica el precio real.

- Días evaluados: **8** (2026-09-12 → 2026-09-19)
- MAE por periodo, todo el histórico publicado: **16,45 EUR/MWh** (ingenua del día anterior: 31,58)
- MAE de la media diaria: **5,75 EUR/MWh**

## Por día

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="grafico_diario_oscuro.svg">
  <img src="grafico_diario.svg" alt="Precio medio de cada día, previsto frente a real, y error medio absoluto de cada día frente al de la previsión ingenua">
</picture>

## Por hora

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="grafico_horario_oscuro.svg">
  <img src="grafico_horario.svg" alt="Precio cuarto a cuarto de la última semana, previsto frente a real, y error medio absoluto por hora del día">
</picture>

Arriba, la última semana cuarto a cuarto; abajo, en qué horas del día se
equivoca más, con todos los días evaluados.

## Últimos 30 días

| Día | Periodos | MAE | Ingenua | Media real | Media prevista |
|---|---:|---:|---:|---:|---:|
| 2026-09-19 | 96 | 11,73 | 21,04 | 110,35 | 112,68 |
| 2026-09-18 | 96 | 20,88 | 21,90 | 126,25 | 138,28 |
| 2026-09-17 | 96 | 26,28 | 35,97 | 127,68 | 130,06 |
| 2026-09-16 | 96 | 21,13 | 62,74 | 104,64 | 124,10 |
| 2026-09-15 | 96 | 18,05 | 19,60 | 167,38 | 168,16 |
| 2026-09-14 | 96 | 12,38 | 56,09 | 177,48 | 172,50 |
| 2026-09-13 | 96 | 8,52 | 8,81 | 121,39 | 124,28 |
| 2026-09-12 | 96 | 12,67 | 26,48 | 118,28 | 117,13 |

Detalle completo: [`evaluacion_diaria.csv`](evaluacion_diaria.csv) (por día) y [`evaluacion_periodos.csv`](evaluacion_periodos.csv) (por periodo).
