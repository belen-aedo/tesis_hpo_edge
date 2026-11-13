"""
Preprocesador de fuente para CICIoTDataset2023.

Transformaciones específicas:
  - El CSV tiene un label multiclase 'Label' (tipo de ataque o 'benign').
  - Se crean columnas 'Cat' y 'Sub_Cat' derivadas del Label:
      * Cat     → 'benign' si benign, 'abnormal' en caso contrario
      * Sub_Cat → el valor original de 'Label' (tipo específico de ataque)
  - La columna 'Protocol Type' es categórica y se conserva tal cual
    para la codificación posterior en 02_preprocessing.py.
"""

from __future__ import annotations

from typing import Iterator

import pandas as pd

from base_preprocessor import SourcePreprocessor


class CICIoTDataset2023Preprocessor(SourcePreprocessor):

    def _iter_raw_chunks(self) -> Iterator[pd.DataFrame]:
        # El CSV preprocesado original puede ser muy grande (múltiples subcarpetas).
        # Aquí cargamos desde la ruta configurada.
        source_path = (
            self.output_path.parent / "raw_combined.csv"
        )
        if not source_path.exists():
            # Fallback: el notebook original lo llamaba preprocessed.csv
            source_path = self.output_path.parent / "preprocessed_raw.csv"
        yield from self._read_csv_chunks(source_path)

    def transform_chunk(self, chunk: pd.DataFrame) -> pd.DataFrame:
        df = chunk.copy()

        label_col = self.info["label"]             # 'Label'
        normal_lbl = self.info["normal_label"]      # 'benign'
        cat_col = self.info["category_col"]         # 'Cat'
        sub_col = self.info["sub_category_col"]     # 'Sub_Cat'

        # Sub_Cat = tipo específico del Label original
        df[sub_col] = df[label_col].astype(str)

        # Cat = binario normal vs anómalo
        df[cat_col] = df[label_col].apply(
            lambda x: normal_lbl if str(x).lower() == str(normal_lbl).lower() else self.info["anomaly_label"]
        )

        return df
