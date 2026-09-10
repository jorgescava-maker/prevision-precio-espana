"""
Diccionario de datos vivo: documentación de cada columna + estadísticos recalculados
sobre los datos reales de la propia base. Se guarda DENTRO de cada .duckdb (tabla
data_dictionary) para que la documentación viaje siempre junto con los datos y nunca
quede desincronizada de lo que realmente contiene la tabla.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import duckdb


@dataclass
class ColumnDoc:
    description: str   # qué representa la columna, en lenguaje de negocio
    source: str         # de qué campo/cálculo exacto sale el valor
    dtype: str           # tipo SQL declarado en la tabla
    kind: str            # "continuous" | "categorical" | "identifier" | "timestamp"


def ensure_schema(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS data_dictionary (
            table_name     VARCHAR NOT NULL,
            column_name    VARCHAR NOT NULL,
            description    VARCHAR NOT NULL,
            source         VARCHAR NOT NULL,
            dtype          VARCHAR NOT NULL,
            variable_kind  VARCHAR NOT NULL,
            min_value      DOUBLE,
            max_value      DOUBLE,
            mean_value     DOUBLE,
            median_value   DOUBLE,
            n_non_null     BIGINT,
            n_null         BIGINT,
            computed_at    TIMESTAMP NOT NULL,
            PRIMARY KEY (table_name, column_name)
        )
        """
    )


def refresh(con: duckdb.DuckDBPyConnection, table_name: str, column_docs: dict[str, ColumnDoc]) -> None:
    """Recalcula min/max/media/mediana sobre los datos reales de `table_name` y
    reescribe la ficha de cada columna. Llamar una vez tras cada carga (no por fila)."""
    ensure_schema(con)
    now = datetime.utcnow()
    total_rows = con.execute(f"SELECT count(*) FROM {table_name}").fetchone()[0]

    records = []
    for col, doc in column_docs.items():
        if doc.kind == "continuous" and total_rows > 0:
            min_v, max_v, mean_v, median_v, n_non_null = con.execute(
                f"SELECT min({col}), max({col}), avg({col}), median({col}), count({col}) FROM {table_name}"
            ).fetchone()
        else:
            min_v = max_v = mean_v = median_v = None
            n_non_null = con.execute(f"SELECT count({col}) FROM {table_name}").fetchone()[0]
        n_null = total_rows - n_non_null

        records.append(
            (table_name, col, doc.description, doc.source, doc.dtype, doc.kind,
             min_v, max_v, mean_v, median_v, n_non_null, n_null, now)
        )

    con.execute("DELETE FROM data_dictionary WHERE table_name = ?", [table_name])
    con.executemany(
        """
        INSERT INTO data_dictionary
            (table_name, column_name, description, source, dtype, variable_kind,
             min_value, max_value, mean_value, median_value, n_non_null, n_null, computed_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        records,
    )


def print_report(con: duckdb.DuckDBPyConnection, table_name: str) -> None:
    cursor = con.execute(
        """
        SELECT column_name, description, source, dtype, variable_kind,
               min_value, max_value, mean_value, median_value, n_non_null, n_null
        FROM data_dictionary WHERE table_name = ? ORDER BY column_name
        """,
        [table_name],
    )
    cols = [c[0] for c in cursor.description]
    rows = cursor.fetchall()
    if not rows:
        print(f"(sin diccionario de datos para {table_name} — ejecuta refresh() primero)")
        return

    print(f"\n=== Diccionario de datos: {table_name} ===")
    for values in rows:
        r = dict(zip(cols, values))
        print(f"\n· {r['column_name']}  [{r['dtype']} / {r['variable_kind']}]")
        print(f"    Significado : {r['description']}")
        print(f"    Origen      : {r['source']}")
        if r["variable_kind"] == "continuous" and r["min_value"] is not None:
            print(
                f"    min={r['min_value']:.4g}  max={r['max_value']:.4g}  "
                f"media={r['mean_value']:.4g}  mediana={r['median_value']:.4g}  "
                f"(n={r['n_non_null']}, nulos={r['n_null']})"
            )
        else:
            print(f"    n={r['n_non_null']}, nulos={r['n_null']}")
