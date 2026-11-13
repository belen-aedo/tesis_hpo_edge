"""
Preprocesador de fuente para N-BaIoT.

Transformaciones específicas:
  - El CSV ya tiene las columnas numéricas y una columna 'Label'
    con los valores 'benign' / 'abnormal'.
  - Se añaden columnas 'Cat' y 'Sub_Cat' para mantener el esquema estándar
    del proyecto (en N-BaIoT no existen subcategorías diferenciadas,
    por lo que Cat y Sub_Cat replican el valor del Label).
"""

from __future__ import annotations

from typing import Iterator

import pandas as pd

from base_preprocessor import SourcePreprocessor


class NBaIoTPreprocessor(SourcePreprocessor):

    def _iter_raw_chunks(self) -> Iterator[pd.DataFrame]:
        source_path = (
            self.output_path.parent / "full" / "final.csv"
        )
        yield from self._read_csv_chunks(source_path)

    def transform_chunk(self, chunk: pd.DataFrame) -> pd.DataFrame:
        df = chunk.copy()

        label_col = self.info["label"]         # 'Label'
        cat_col   = self.info["category_col"]  # 'Cat'
        sub_col   = self.info["sub_category_col"]  # 'Sub_Cat'

        # N-BaIoT no tiene categorías explícitas; se replica el label
        if cat_col not in df.columns:
            df[cat_col] = df[label_col]
        if sub_col not in df.columns:
            df[sub_col] = df[label_col]

        return df
