"""
Clase base abstracta para los preprocesadores de fuente.

Cada dataset tiene sus propias particularidades en el formato bruto 
(distintas columnas de etiqueta, distintos formatos de archivo, valores 
que representan tráfico normal, etc.). Este módulo provee:

  1. Una interfaz común (SourcePreprocessor) que todos los preprocesadores
     deben implementar.
  2. Helpers reutilizables para operaciones frecuentes (concatenar CSVs,
     guardar en chunks, etc.) siguiendo las mejores prácticas.

La responsabilidad de esta capa es EXCLUSIVAMENTE:
    CSV bruto → CSV estandarizado (columnas label / category / sub_category)
    
El Train/Test split y el resto del pipeline ocurren DESPUÉS (scripts 01-04).
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator

import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    datefmt="%H:%M:%S",
)


class SourcePreprocessor(ABC):
    """
    Interfaz base para los preprocesadores específicos de cada dataset.

    Subclases deben implementar:
        - transform_chunk(chunk): transforma un chunk de DataFrame crudo
          al esquema estándar y lo retorna.
    
    Los campos del esquema estándar de SALIDA son los definidos en config.json:
        label         – etiqueta binaria (normal / anomaly_label)
        category_col  – categoría de ataque o 'normal'
        sub_category_col – subcategoría / detalle del ataque
        numerical_cols – columnas numéricas sin modificar (serán procesadas
                         por 02_preprocessing.py)
        categorical_cols – columnas categóricas (serán codificadas en 02)
    """

    # random_state global del proyecto para reproducibilidad
    RANDOM_STATE: int = 42

    def __init__(self, dataset_name: str, config: dict):
        """
        Args:
            dataset_name: Clave del dataset en config.json (ej. 'DS2OS').
            config: Diccionario completo retornado por load_config().
        """
        self.name = dataset_name
        self.info = config["DATASETS"][dataset_name]
        self.root = config["ROOT"]
        self.chunksize = config["CHUNKSIZE"]

        raw_path = self.info["path"].lstrip(os.sep)
        self.output_path = Path(self.root) / raw_path

    # ------------------------------------------------------------------
    # Interfaz pública
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Orquesta la lectura, transformación y guardado del dataset."""
        logger.info("=" * 60)
        logger.info("Iniciando preprocesamiento de fuente: %s", self.name)
        logger.info("Salida esperada: %s", self.output_path)
        logger.info("=" * 60)

        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        first_chunk = True
        for i, chunk in enumerate(self._iter_raw_chunks()):
            logger.info("  Procesando chunk %d (%d filas)…", i + 1, len(chunk))
            processed = self.transform_chunk(chunk)
            if processed is None or processed.empty:
                logger.warning("  Chunk %d vacío tras transformación. Se omite.", i + 1)
                continue

            mode = "w" if first_chunk else "a"
            header = first_chunk
            processed.to_csv(self.output_path, mode=mode, index=False, header=header)
            first_chunk = False

        logger.info("✅ Preprocesamiento de %s completado → %s", self.name, self.output_path)

    # ------------------------------------------------------------------
    # Métodos abstractos
    # ------------------------------------------------------------------

    @abstractmethod
    def transform_chunk(self, chunk: pd.DataFrame) -> pd.DataFrame:
        """
        Transforma un chunk del CSV crudo al esquema estándar del proyecto.

        Reglas:
         - NO aplicar escalado, imputación ni codificación (eso es del paso 02).
         - Solo renombrar columnas, crear label/category/sub_category,
           filtrar filas inválidas y castear tipos mínimos necesarios.
         - Retornar None o DataFrame vacío para omitir el chunk.
        """

    @abstractmethod
    def _iter_raw_chunks(self) -> Iterator[pd.DataFrame]:
        """
        Itera sobre los archivos crudos retornando chunks de DataFrame.
        La implementación por defecto usa un CSV único; subclases pueden
        sobrescribir este método para múltiples archivos, formatos especiales, etc.
        """

    # ------------------------------------------------------------------
    # Helpers protegidos reutilizables
    # ------------------------------------------------------------------

    def _read_csv_chunks(self, path: Path | str, **kwargs) -> Iterator[pd.DataFrame]:
        """Lee un CSV en chunks usando el CHUNKSIZE del proyecto."""
        yield from pd.read_csv(path, chunksize=self.chunksize, low_memory=False, **kwargs)

    def _read_full_csv(self, path: Path | str, **kwargs) -> pd.DataFrame:
        """Lee un CSV completo en memoria (solo para datasets pequeños)."""
        return pd.read_csv(path, low_memory=False, **kwargs)

    def _standardize_label(
        self, series: pd.Series, normal_val: str, anomaly_val: str
    ) -> pd.Series:
        """
        Mapea cualquier columna de etiquetas a los valores estándar
        del proyecto: normal_label y anomaly_label definidos en config.json.
        Filas sin correspondencia se marcan como NaN (serán eliminadas en 02).
        """
        mapping = {
            str(normal_val): str(self.info["normal_label"]),
            str(anomaly_val): str(self.info["anomaly_label"]),
        }
        return series.astype(str).map(mapping)
