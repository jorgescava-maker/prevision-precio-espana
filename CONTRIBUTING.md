# Cómo proponer una mejora

Toda propuesta es bienvenida: una variable nueva, un dato que no se está
usando, una forma distinta de entrenar, un error en el código o una idea sobre
cómo funciona el mercado que el modelo ignora. No hace falta traerla medida.

## Las dos formas

1. **Una idea** → abre un
   [issue de propuesta](../../issues/new?template=propuesta.yml). Cuenta qué
   cambiarías y por qué crees que ayudaría. Si tiene detrás un mecanismo del
   mercado (una regla de OMIE, cómo oferta la hidráulica, una norma de REE),
   mejor todavía: la mitad de las mejoras de este modelo han salido de
   entender el mecanismo, no de probar a ciegas.
2. **Una mejora medida** → abre un pull request con el código y la salida de
   `scripts/comparar.py` contra la referencia.

## Cómo se decide

**El criterio es el MAE agregado por periodo** sobre los periodos de la
referencia (`previsiones/walkforward/espana_referencia.csv`). Si lo baja, es
una mejora y se adopta. No hay más requisitos.

Los desgloses por mes, el MAE de la media diaria y el test de Diebold-Mariano
que imprime `comparar.py` sirven para **entender** el resultado, no son
listones que haya que superar.

Para que los números sean comparables:

- **Sin fuga.** Cada variable tiene que ser conocida antes de las 12:00 del día
  anterior al de entrega, que es cuando cierra la subasta. Los precios del
  propio día, de España o de cualquier país acoplado, no valen: se casan a la
  vez.
- **Walk-forward mensual.** Se entrena con todo lo anterior a cada mes y se
  predice ese mes. Todo lo que se estime (umbrales, pesos, calibraciones) se
  estima solo con meses ya observados. **Esto incluye las variables que salen
  de otro modelo ajustado**: si una variable se calcula con algo estimado
  sobre la historia (una superficie, una curva, una normalización), en cada
  paso hay que reestimarlo con lo anterior a ese mes, y recalcular con ello
  la variable también para las filas de entrenamiento. El valor del agua es
  el ejemplo de este repositorio: ver `studies/precio_da_conjunto/agua_causal.py`.
- **Mismos periodos.** Si tu variante deja periodos sin predecir, dilo:
  un MAE sobre menos filas no es comparable. `comparar.py` lo avisa.
- **La referencia no se reentrena.** Se compara contra la predicción de
  producción ya guardada, no contra una versión que tú vuelvas a correr.

## Para reproducir el modelo

```
python -m venv .venv
.venv\Scripts\activate          # o source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env          # y rellena los dos tokens
python -m scripts.construir_bases
python -m scripts.reconstruir   # datasets + walk-forward completo
```

Si solo quieres experimentar con variables, no hace falta montar las bases:
descarga los datasets de la última release y completa las columnas de materias
primas con `python -m scripts.completar_materias_primas`.

## Estilo

El código y los comentarios están en español. Los comentarios explican **por
qué** se hace algo, sobre todo cuando la razón es un dato o un mecanismo del
mercado; lo que hace el código ya se lee en el código.
