"""Conexión y esquema común a todas las bases de datos por país."""

from pathlib import Path

import duckdb

from etl.common import catalog
from etl.common.catalog import ColumnDoc

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

INGESTION_LOG_COLUMNS = {
    "run_id": ColumnDoc(
        description="Identificador secuencial autoincremental de cada ejecución de ingesta registrada.",
        source="Generado por la secuencia ingestion_log_seq al insertar la fila.",
        dtype="BIGINT", kind="identifier",
    ),
    "source": ColumnDoc(
        description="Nombre corto de la fuente/pipeline que generó el registro (p.ej. 'omie_spot').",
        source="Constante fijada por el script de ingesta que escribe el log.",
        dtype="VARCHAR", kind="categorical",
    ),
    "target_table": ColumnDoc(
        description="Tabla de datos que recibió (o debía recibir) las filas de esta ejecución.",
        source="Constante fijada por el script de ingesta que escribe el log.",
        dtype="VARCHAR", kind="categorical",
    ),
    "delivery_date": ColumnDoc(
        description="Día de entrega (calendario del mercado) al que se refiere la ejecución de ingesta.",
        source="Parámetro de entrada de la función run_for_date().",
        dtype="DATE", kind="identifier",
    ),
    "rows_expected": ColumnDoc(
        description="Número de periodos de mercado que se esperaban para esa fecha según calendario y resolución vigente en origen.",
        source="Calculado por la función expected_period_count() de cada fuente antes de comparar con lo realmente parseado.",
        dtype="INTEGER", kind="continuous",
    ),
    "rows_loaded": ColumnDoc(
        description="Número de filas efectivamente insertadas en la tabla destino en esta ejecución.",
        source="Recuento de registros pasados a la sentencia INSERT del pipeline.",
        dtype="INTEGER", kind="continuous",
    ),
    "status": ColumnDoc(
        description="Resultado de la validación de esta carga: OK (sin incidencias), WARN (cargado con advertencias) o ERROR (no se pudo cargar).",
        source="Derivado por la función validate() de cada fuente.",
        dtype="VARCHAR", kind="categorical",
    ),
    "message": ColumnDoc(
        description="Detalle textual de las incidencias de validación detectadas (vacío/„sin incidencias“ si no hubo ninguna).",
        source="Concatenación de los mensajes devueltos por validate().",
        dtype="VARCHAR", kind="categorical",
    ),
    "started_at": ColumnDoc(
        description="Marca temporal UTC de inicio de la ejecución de ingesta.",
        source="datetime.utcnow() al entrar en run_for_date().",
        dtype="TIMESTAMP", kind="timestamp",
    ),
    "finished_at": ColumnDoc(
        description="Marca temporal UTC de fin de la ejecución de ingesta.",
        source="datetime.utcnow() al escribir el registro en ingestion_log.",
        dtype="TIMESTAMP", kind="timestamp",
    ),
}


def connect(country_code: str) -> duckdb.DuckDBPyConnection:
    """Abre (o crea) la base de datos DuckDB de un país y asegura el esquema común."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    db_path = DATA_DIR / f"{country_code}.duckdb"
    con = duckdb.connect(str(db_path))
    ensure_common_schema(con)
    return con


def document_ingestion_log(con: duckdb.DuckDBPyConnection) -> None:
    catalog.refresh(con, "ingestion_log", INGESTION_LOG_COLUMNS)


def ensure_common_schema(con: duckdb.DuckDBPyConnection) -> None:
    """Tabla de auditoría de cada ejecución de ingesta, compartida por todas las fuentes."""
    con.execute("CREATE SEQUENCE IF NOT EXISTS ingestion_log_seq START 1")
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS ingestion_log (
            run_id          BIGINT PRIMARY KEY DEFAULT nextval('ingestion_log_seq'),
            source          VARCHAR NOT NULL,
            target_table    VARCHAR NOT NULL,
            delivery_date   DATE NOT NULL,
            rows_expected   INTEGER,
            rows_loaded     INTEGER,
            status          VARCHAR NOT NULL,   -- OK | WARN | ERROR
            message         VARCHAR,
            started_at      TIMESTAMP NOT NULL,
            finished_at     TIMESTAMP
        )
        """
    )
