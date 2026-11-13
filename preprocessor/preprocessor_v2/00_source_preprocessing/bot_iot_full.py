"""
Preprocesador de fuente para el dataset Bot-IoT Full.

Transformaciones específicas:
  - El dataset original está fragmentado en 74 archivos CSV consecutivos
    (UNSW_2018_IoT_Botnet_Dataset_1.csv … _74.csv) más el de nombres de columnas.
  - Se concatenan todos en un único 'complete.csv' iterando por chunks
    para no saturar la memoria.
  - Las columnas label (attack), category y subcategory ya existen en el CSV
    crudo y no requieren transformación, solo validación.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pandas as pd

from base_preprocessor import SourcePreprocessor


class BotIoTFullPreprocessor(SourcePreprocessor):

    def _iter_raw_chunks(self) -> Iterator[pd.DataFrame]:
        source_dir = (
            Path(self.root)
            / "Datasets"
            / "Sources"
            / "Bot-IoT"
            / "CSV"
            / "Entire Dataset"
        )
        # El primer archivo contiene los nombres de columna
        header_file = source_dir / "UNSW_2018_IoT_Botnet_Dataset_Feature_Names.csv"
        part_files = sorted(
            source_dir.glob("UNSW_2018_IoT_Botnet_Dataset_[0-9]*.csv")
        )
        all_files = [header_file] + part_files

        for csv_file in all_files:
            if not csv_file.exists():
                continue
            yield from self._read_csv_chunks(csv_file)

    def transform_chunk(self, chunk: pd.DataFrame) -> pd.DataFrame:
        """
        El dataset Bot-IoT ya tiene las columnas estandarizadas:
          - 'attack'      → label (0/1)
          - 'category'    → categoría de ataque
          - 'subcategory' → subcategoría

        Solo se elimina la columna 'subcategory' duplicada si existe,
        renombrándola al estándar 'sub_category'.
        """
        df = chunk.copy()

        # Renombrar subcategory → sub_category si aplica
        if "subcategory" in df.columns and "sub_category" not in df.columns:
            df.rename(columns={"subcategory": "sub_category"}, inplace=True)

        return df
