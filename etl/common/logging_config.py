"""Configuración de logging compartida: consola + fichero diario por país."""

import logging
import sys
from datetime import date
from pathlib import Path

LOGS_DIR = Path(__file__).resolve().parents[2] / "logs"

# La consola de Windows (cp1252/cp850) rompe tildes y símbolos si no se fuerza UTF-8.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8")
        except Exception:
            pass


def get_logger(name: str, country_code: str) -> logging.Logger:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # ya configurado (evita duplicar handlers en re-imports)

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")

    file_handler = logging.FileHandler(
        LOGS_DIR / f"{country_code}_{date.today().isoformat()}.log", encoding="utf-8"
    )
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)

    return logger
